import logging
import numpy as np
import time

from dataclasses import asdict, dataclass

from afe.ir.tensor_type import ScalarType
from afe.ir.quantization_conv import block_quantize_weight_tensor

from sima_utils.logging.sima_logger import sima_log_info, sima_log_warning

from sima_lmm.gguf.gguf_conversion import GgufModel
from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel
from sima_lmm.model.base import (
    BaseModel, EvalMode, FileGenMode, FileGenPrecision, LoraGenMode, GenConfiguration
)
from sima_lmm.model.language_pre_model import LanguagePreModel
from sima_lmm.model.language_post_model import LanguagePostModel
from sima_lmm.model.language_moe_router_model import LanguageMoeRouterModel
from sima_lmm.model.language_moe_weightedsum_model import LanguageMoeWeightedSumModel
from sima_lmm.model.language_cache_model import LanguageCacheModel
from sima_lmm.model.language_conv_model import LanguageConvModel
from sima_lmm.model.language_conv_post_model import LanguageConvPostModel
from sima_lmm.model.language_draft_fc_model import LanguageDraftFCModel
from sima_lmm.model.language_per_layer_model import LanguagePerLayerModel
from sima_lmm.utils import calc_freq_real_imag, round_up_to
from sima_lmm.config.layer_id import LayerID
from sima_lmm.config.vlm_config import LlmArchType, VlmArchType, PipelineConfig


bfloat16 = ScalarType.numpy_type(ScalarType.bfloat16)


def quantize_embedding_rows(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize each vocabulary row and return MLA dynamic-dequant scales."""
    assert embeddings.ndim == 2
    vocab_size, hidden_size = embeddings.shape

    # The weight quantizer treats the final flattened axis as output channels. Put the
    # vocabulary axis there so per-channel quantization becomes per-vocabulary-row.
    channel_layout = embeddings.T.reshape(1, 1, hidden_size, 1, vocab_size)
    quantized, dequant_steps = block_quantize_weight_tensor(
        channel_layout, per_channel=True, bits=8, c_block_size=None
    )
    quantized = quantized.reshape(hidden_size, vocab_size).T

    # block_quantize_weight_tensor returns absmax / 127. DynamicDequant expects absmax.
    scales = np.asarray(dequant_steps, dtype=np.float32).reshape(vocab_size) * 127.0
    scales[np.count_nonzero(embeddings, axis=1) == 0] = 0.0
    return quantized, scales.reshape(vocab_size, 1).astype(bfloat16)


@dataclass
class LanguageModel(BaseModel):
    """Language model implementation.

    The language model consists of a stack of transformer layers and some layers after the last
    transformer layer to obtain the output probability or next token index.

    The implementation assumes a transformer is broken up into 3 parts.
    1. PreCacheModel: Pre cache model implements the transformer layer up to the batched
        matrix-multiply in the self-attention block.
    2. CacheModel: Cache model implements the self-attention block of a transformer layer without
        the qkv projection.
    3. PostCacheModel: Post cache model implements the transformer layer after the self-attention
        block, including the layers after the last transformer layer.
    """
    def __post_init__(self):
        if self.cfg.pipeline_cfg.input_token_group_offsets:
            self.cfg.pipeline_cfg.input_token_group_offsets.sort()

    def gen_files(
        self,
        gen_mode: FileGenMode,
        *,
        gen_config: GenConfiguration,
        log_level: int = logging.NOTSET,
        num_processes: int = 1,
        resume: bool = False,
    ):
        """
        Generates files based on the provided file generation mode.

        Args:
            gen_mode: File generation mode.
            gen_config: Generation configuration of precision and lora for each layer.
                Layer IDs can be obtained using VlmConfig.get_layer_ids.
                The precision dict is a map from layer ID to precision for each layer to be
                processed. Unrecognized layer IDs will be ignored.  Layers that are not in the map
                will not be processed.
                The lora dict is a map from layer ID to lora graph mode for each layer.
            log_level: Logging level.
            resume: Generate the files if missing.
        """
        # Create a list of all models to compile
        model_list = list()
        num_tokens = self.cfg.pipeline_cfg.input_token_group_size
        single_model_num_tokens = self._single_model_num_tokens
        precision = gen_config["precision"]
        lora_mode = gen_config.get("lora", None)

        for layer_id, curr_precision in precision.items():
            part_model = None
            match layer_id.part:
                case "group_pre":
                    part_model = self._get_part_model(
                        "pre", num_tokens, layer_idx=layer_id.part_idx
                    )
                case "single_pre":
                    part_model = self._get_part_model(
                        "pre", single_model_num_tokens, layer_idx=layer_id.part_idx
                    )
                case "group_post":
                    part_model = self._get_part_model(
                        "post", num_tokens, layer_idx=layer_id.part_idx
                    )
                case "single_post":
                    part_model = self._get_part_model(
                        "post", single_model_num_tokens, layer_idx=layer_id.part_idx
                    )
                case "group_cache":
                    part_model = self._get_part_model(
                        "cache", num_tokens, token_idx=layer_id.part_idx
                    )
                case "single_cache":
                    part_model = self._get_part_model(
                        "cache", single_model_num_tokens, token_idx=layer_id.part_idx
                    )
                case "group_sliding_cache":
                    part_model = self._get_part_model(
                        "sliding_cache", num_tokens, token_idx=layer_id.part_idx
                    )
                case "single_sliding_cache":
                    part_model = self._get_part_model(
                        "sliding_cache", 1, token_idx=layer_id.part_idx
                    )
                case "group_conv":
                    part_model = self._get_part_model(
                        "conv_fused", num_tokens, layer_idx=layer_id.part_idx
                    )
                case "single_conv":
                    part_model = self._get_part_model(
                        "conv_fused", 1, layer_idx=layer_id.part_idx
                    )
                case "conv_post_final":
                    part_model = self._get_part_model(
                        "conv_post_final", 1, layer_idx=layer_id.part_idx
                    )
                case "group_draft_fc":
                    part_model = self._get_part_model(
                        "draft_fc", num_tokens, layer_idx=layer_id.part_idx
                    )
                case "single_draft_fc":
                    part_model = self._get_part_model(
                        "draft_fc", single_model_num_tokens, layer_idx=layer_id.part_idx
                    )
                case "group_per_layer":
                    part_model = self._get_part_model("per_layer", num_tokens)
                case "single_per_layer":
                    part_model = self._get_part_model("per_layer", 1)
                case "group_router":
                    part_model = self._get_part_model(
                        "router", num_tokens, layer_idx=layer_id.part_idx
                    )
                case "single_router":
                    part_model = self._get_part_model(
                        "router", single_model_num_tokens, layer_idx=layer_id.part_idx
                    )
                case "group_expert":
                    part_model = self._get_part_model(
                        "post", num_tokens, layer_idx=layer_id.part_idx,
                        expert_idx=layer_id.expert_idx,
                    )
                case "single_expert":
                    part_model = self._get_part_model(
                        "post", single_model_num_tokens, layer_idx=layer_id.part_idx,
                        expert_idx=layer_id.expert_idx,
                    )
                case "group_weightedsum":
                    part_model = self._get_part_model(
                        "moe_weightedsum", num_tokens, layer_idx=layer_id.part_idx
                    )
                case "single_weightedsum":
                    part_model = self._get_part_model(
                        "moe_weightedsum", single_model_num_tokens, layer_idx=layer_id.part_idx
                    )
                case _:
                    # Not a part of this model
                    continue
            curr_cfg = {"precision": curr_precision}
            if lora_mode:
                curr_cfg["lora"] = lora_mode[layer_id]
            model_list.append((part_model, curr_cfg))

        # Finished creating model_list.  Compile these models.
        self.gen_files_from_model_list(model_list, gen_mode, num_processes, log_level, resume)

    def run_model(
        self,
        eval_mode: EvalMode,
        ifms: list[np.ndarray],
        embeddings_tensor: np.ndarray | None = None,
        embedding_scales: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        assert self.vlm_helper is not None
        expected_num_ifms = 2 if self.cfg.pipeline_cfg.quantize_embeddings else 1
        assert len(ifms) == expected_num_ifms
        input_embeds = ifms[0]
        if embeddings_tensor is None:
            embeddings_tensor, embedding_scales = self.get_embeddings_tensor()
        if self.cfg.pipeline_cfg.quantize_embeddings:
            assert embedding_scales is not None
            input_embedding_scales = ifms[1]

        # Obtain sliding window config.
        swa_enable = self.cfg.lm_cfg.attn_cfg.swa_enable
        sliding_window = self.cfg.lm_cfg.attn_cfg.sliding_window
        layer_types = self.cfg.lm_cfg.layer_types

        # Initialize rope.
        global_freq_real, global_freq_imag = self.calc_freq_real_imag(False)
        if swa_enable:
            local_freq_real, local_freq_imag = self.calc_freq_real_imag(True)

        # Initialize cache.
        num_kv_heads = self.cfg.lm_cfg.attn_cfg.num_key_value_heads
        head_dim = self.cfg.lm_cfg.attn_cfg.head_dim
        quantize_kv_cache = self.cfg.pipeline_cfg.quantize_kv_cache
        kv_dtype = np.int8 if quantize_kv_cache else np.float32
        cache_key = [
            np.zeros((1, num_kv_heads, self.cfg.pipeline_cfg.max_num_tokens, head_dim), dtype=kv_dtype)
            for _ in range(self.cfg.lm_cfg.num_hidden_layers)
        ]
        cache_val = [
            np.zeros((1, num_kv_heads, self.cfg.pipeline_cfg.max_num_tokens, head_dim), dtype=kv_dtype)
            for _ in range(self.cfg.lm_cfg.num_hidden_layers)
        ]
        if quantize_kv_cache:
            cache_key_scale = [
                np.zeros((1, num_kv_heads, self.cfg.pipeline_cfg.max_num_tokens, 1), dtype=np.float32)
                for _ in range(self.cfg.lm_cfg.num_hidden_layers)
            ]
            cache_val_scale = [
                np.zeros((1, num_kv_heads, self.cfg.pipeline_cfg.max_num_tokens, 1), dtype=np.float32)
                for _ in range(self.cfg.lm_cfg.num_hidden_layers)
            ]

        # Determine group models.
        token_group_offsets = self.cfg.pipeline_cfg.input_token_group_offsets
        num_input_tokens = input_embeds.shape[2]
        group_token_idx_list = list()
        padded_token_size = 1

        token_idx_set = set(range(self.cfg.pipeline_cfg.max_num_tokens))
        if token_group_offsets:
            for i in token_group_offsets:
                padded_token_size = i + self.cfg.pipeline_cfg.input_token_group_size
                tmp_token_idx_set = (
                    token_idx_set - set(range(i, min(num_input_tokens, padded_token_size)))
                )
                if len(tmp_token_idx_set) != len(token_idx_set):
                    token_idx_set = tmp_token_idx_set
                    token_idx_set.add(i)
                    group_token_idx_list.append(i)
                if num_input_tokens <= padded_token_size:
                    break
        token_idx_list = sorted(list(token_idx_set))

        # Pad the input embedding if needed and determine which token indices to run.
        if (padding := padded_token_size - num_input_tokens) > 0:
            input_embeds = np.pad(input_embeds, ((0, 0), (0, 0), (0, padding), (0, 0)))
            if self.cfg.pipeline_cfg.quantize_embeddings:
                input_embedding_scales = np.pad(
                    input_embedding_scales, ((0, 0), (0, 0), (0, padding), (0, 0))
                )

        # For paligemma, the original implementation requires all the input tokens to be processed
        # together (attention mask = 0) to obtain the matching output.
        if (
            self.cfg.model_type == VlmArchType.VLM_PALIGEMMA
            and padded_token_size < num_input_tokens
        ):
            sima_log_warning(
                f"Number of input tokens ({num_input_tokens}) is greater than the maximum allowed"
                f" number of tokens ({padded_token_size}) for PaliGemma."
                "\nPlease increase the estimated_max_num_query_tokens in config_pipeline()."
            )

        # Run language model.
        perf_cnt_begin = time.perf_counter_ns()

        new_tokens = list()
        post_ofms = None
        new_token = None
        for token_idx in token_idx_list:
            if token_idx in group_token_idx_list:
                num_tokens = self.cfg.pipeline_cfg.input_token_group_size
                next_token_idx = min(num_input_tokens, token_idx + num_tokens)
            else:
                num_tokens = 1
                next_token_idx = token_idx + 1
            sima_log_info("Processing token no. %d-%d", token_idx, next_token_idx)

            use_input_tokens = token_idx < num_input_tokens
            if not use_input_tokens:
                perf_cnt_begin = time.perf_counter_ns()

            for layer_idx in range(self.cfg.lm_cfg.num_hidden_layers):
                is_global = (layer_types[layer_idx] == "full_attention")
                pre_ifms = list()
                if layer_idx == 0:
                    if use_input_tokens:
                        pre_ifms.append(input_embeds[..., token_idx:token_idx + num_tokens, :])
                    else:
                        assert new_token is not None
                        pre_ifms.append(
                            np.expand_dims(embeddings_tensor[new_token], axis=(0, 1, 2))
                        )
                else:
                    assert post_ofms is not None
                    pre_ifms.append(post_ofms[0])
                if self.cfg.pipeline_cfg.quantize_embeddings and layer_idx == 0:
                    if use_input_tokens:
                        pre_ifms.append(
                            input_embedding_scales[
                                ..., token_idx:token_idx + num_tokens, :
                            ]
                        )
                    else:
                        pre_ifms.append(
                            np.expand_dims(embedding_scales[new_token], axis=(0, 1, 2))
                        )
                if is_global:
                    pre_ifms.append(global_freq_real[..., token_idx:token_idx + num_tokens, :])
                    pre_ifms.append(global_freq_imag[..., token_idx:token_idx + num_tokens, :])
                else:
                    pre_ifms.append(local_freq_real[..., token_idx:token_idx + num_tokens, :])
                    pre_ifms.append(local_freq_imag[..., token_idx:token_idx + num_tokens, :])
                pre_model = self._get_part_model("pre", num_tokens, layer_idx=layer_idx)
                pre_ofms = pre_model.run_model(eval_mode, pre_ifms)

                # Update cache.
                if quantize_kv_cache:
                    cache_key[layer_idx][..., token_idx:token_idx + num_tokens, :] = pre_ofms[1]
                    cache_key_scale[layer_idx][..., token_idx:token_idx + num_tokens, :] = pre_ofms[2]
                    cache_val[layer_idx][..., token_idx:token_idx + num_tokens, :] = pre_ofms[3]
                    cache_val_scale[layer_idx][..., token_idx:token_idx + num_tokens, :] = pre_ofms[4]
                else:
                    cache_key[layer_idx][..., token_idx:token_idx + num_tokens, :] = pre_ofms[1]
                    cache_val[layer_idx][..., token_idx:token_idx + num_tokens, :] = pre_ofms[2]

                # For multi-token models, there is no need to run the cache and post models for the
                # last hidden layer since pre model already updates the cache.
                if num_tokens > 1 and layer_idx == self.cfg.lm_cfg.num_hidden_layers - 1:
                    if token_idx + num_tokens >= num_input_tokens:
                        # The last input token is included. Run the cache and post model for the
                        # last input token to generate the first output token.
                        idx = num_input_tokens - 1 - token_idx
                        pre_ifms[0] = pre_ifms[0][..., idx:idx + 1, :]
                        if self.cfg.pipeline_cfg.quantize_embeddings and layer_idx == 0:
                            pre_ifms[1] = pre_ifms[1][..., idx:idx + 1, :]
                        pre_ofms[0] = pre_ofms[0][..., idx:idx + 1, :]
                        num_tokens = 1
                        token_idx = num_input_tokens - 1
                    else:
                        break

                if is_global:
                    token_idx_begin = 0
                else:
                    token_idx_begin = max(0, token_idx + num_tokens - sliding_window)
                if self.cfg.pipeline_cfg.future_token_mask_size > 1 and num_tokens == 1:
                    aligned_token_idx = min(
                        round_up_to(token_idx+1, self.cfg.pipeline_cfg.future_token_mask_size) - 1,
                        self.cfg.pipeline_cfg.max_num_tokens - 1
                    )
                else:
                    aligned_token_idx = token_idx
                cache_ifms = [
                    pre_ofms[0],
                    cache_key[layer_idx][..., token_idx_begin:aligned_token_idx + num_tokens, :],
                ]
                if quantize_kv_cache:
                    cache_ifms.append(
                        cache_key_scale[layer_idx][..., token_idx_begin:aligned_token_idx + num_tokens, :]
                    )
                if self.cfg.model_type == VlmArchType.VLM_PALIGEMMA and num_tokens > 1:
                    assert is_global
                    mask = np.zeros((1, 1, num_tokens, token_idx + num_tokens), dtype=np.float32)
                    mask[0, 0, :, num_input_tokens - token_idx:] = np.finfo(np.float32).min
                    cache_ifms.append(mask)
                elif self.cfg.pipeline_cfg.future_token_mask_size > 1 and num_tokens == 1:
                    mask = np.full(
                        (1, 1, 1, aligned_token_idx + 1 - token_idx_begin),
                        np.finfo(np.float32).min,
                        dtype=np.float32
                    )
                    mask[..., :token_idx + 1 - token_idx_begin] = 0
                    cache_ifms.append(mask)
                if self.cfg.lm_cfg.arch == LlmArchType.GPT_OSS:
                    # Per-head attention sink logit, slotted right before cached_values.
                    base_name = self.hf_model.language_model_param_base_name
                    sinks = self.get_hf_param(
                        f"{base_name}.layers.{layer_idx}.self_attn.sinks"
                    ).astype(np.float32)
                    num_attn_heads = self.cfg.lm_cfg.attn_cfg.num_attention_heads
                    cache_ifms.append(
                        np.broadcast_to(
                            sinks.reshape(1, 1, 1, num_attn_heads),
                            (1, 1, num_tokens, num_attn_heads),
                        ).copy()
                    )
                cache_ifms.append(
                    cache_val[layer_idx][..., token_idx_begin:aligned_token_idx + num_tokens, :]
                )
                if quantize_kv_cache:
                    cache_ifms.append(
                        cache_val_scale[layer_idx][..., token_idx_begin:aligned_token_idx + num_tokens, :]
                    )

                cache_model = self._get_part_model(
                    "cache", num_tokens, token_idx=aligned_token_idx - token_idx_begin
                )
                cache_ofms = cache_model.run_model(eval_mode, cache_ifms)

                if self.cfg.lm_cfg.moe_cfg is not None:
                    post_ofms = self._run_moe_post_onnx(
                        eval_mode, num_tokens, layer_idx, pre_ifms[0], cache_ofms[0]
                    )
                else:
                    post_ifms = [pre_ifms[0]]
                    if self.cfg.pipeline_cfg.quantize_embeddings and layer_idx == 0:
                        post_ifms.append(pre_ifms[1])
                    post_ifms.append(cache_ofms[0])
                    post_model = self._get_part_model("post", num_tokens, layer_idx=layer_idx)
                    post_ofms = post_model.run_model(eval_mode, post_ifms)

            # Download the argmax output.
            if num_input_tokens <= next_token_idx:
                if self.cfg.lm_cfg.lm_head_num_splits == 1 and not self.cfg.pipeline_cfg.return_logits:
                    new_token = post_ofms[0].item()
                else:
                    lm_output = np.concatenate(post_ofms, axis=-1)
                    new_token = np.argmax(lm_output)
                if self.cfg.pipeline_cfg.return_logits:
                    new_tokens.append(lm_output)
                else:
                    new_tokens.append(new_token)
                perf_cnt_delta = (time.perf_counter_ns() - perf_cnt_begin) * 1e-9

                if self.cfg.pipeline_cfg.return_logits:
                    sima_log_info("Got logit: %s in %ds", lm_output, perf_cnt_delta)
                else:
                    sima_log_info("Got token: %d in %fs", new_token, perf_cnt_delta)

                if new_token in self.vlm_helper.stop_tokens:
                    break

        # Return the generated tokens.
        return np.array([new_tokens])

    def _run_moe_post_onnx(self, eval_mode, num_tokens, layer_idx, hidden, self_attn):
        """MoE post block: router (TopK+softmax on-graph) -> experts -> weighted-sum."""
        moe = self.cfg.lm_cfg.moe_cfg
        num_experts = moe.num_experts

        router_model = self._get_part_model("router", num_tokens, layer_idx=layer_idx)
        values, indices, residual, norm_hidden = router_model.run_model(
            eval_mode, [hidden, self_attn]
        )
        vals = values[0, 0]                    # (num_tokens, top_k)
        idxs = indices[0, 0].astype(np.int64)  # (num_tokens, top_k)

        # Scatter the k routing weights into a dense (num_tokens, num_experts) tensor.
        router_weights = np.zeros((num_tokens, num_experts), dtype=np.float32)
        for t in range(num_tokens):
            router_weights[t, idxs[t]] = vals[t]
        rw = router_weights[None, None]  # -> NCHW (1, num_experts, 1, num_tokens)

        # Only run experts at least one token selected; the rest have weight 0.
        activated = set(int(e) for e in idxs.flatten())

        def run_expert(e):
            ex_model = self._get_part_model(
                "post", num_tokens, layer_idx=layer_idx, expert_idx=e
            )
            return ex_model.run_model(eval_mode, [norm_hidden, rw])[0]

        expert_outs = []
        if num_tokens > 1:
            # Group combine consumes all num_experts slots; skipped ones are zeros.
            zero = np.zeros(
                (1, 1, num_tokens, self.cfg.lm_cfg.hidden_size), dtype=np.float32
            )
            for e in range(num_experts):
                expert_outs.append(run_expert(e) if e in activated else zero)
        else:
            for e in sorted(activated):
                expert_outs.append(run_expert(e))

        ws_model = self._get_part_model("moe_weightedsum", num_tokens, layer_idx=layer_idx)
        return ws_model.run_model(eval_mode, expert_outs + [residual])

    def calc_freq_real_imag(self, use_swa: bool) -> tuple[np.ndarray, np.ndarray]:
        if use_swa:
            theta = self.cfg.lm_cfg.rope_cfg.rope_local_base_freq
            # A distinct local base freq (Gemma3) means an unscaled local rope; when it
            # matches rope_theta the sliding layers share the global scaling (gpt-oss YaRN).
            rope_type = (
                "default" if theta != self.cfg.lm_cfg.rope_cfg.rope_theta
                else self.cfg.lm_cfg.rope_cfg.rope_scaling.rope_type
            )
        else:
            theta = self.cfg.lm_cfg.rope_cfg.rope_theta
            rope_type = self.cfg.lm_cfg.rope_cfg.rope_scaling.rope_type
        rope_dimension_count = self.cfg.lm_cfg.rope_cfg.rope_dimension_count
        scaling_cfg = asdict(self.cfg.lm_cfg.rope_cfg.rope_scaling)
        max_num_tokens = self.cfg.pipeline_cfg.max_num_tokens
        idx_base = 1 if self.cfg.model_type == VlmArchType.VLM_PALIGEMMA else 0
        return calc_freq_real_imag(
            max_num_tokens, rope_type, theta, rope_dimension_count, scaling_cfg, idx_base
        )

    def get_embeddings_tensor(
        self,
        weight_name: str | None = None,
        embed_scale: float = 1.0,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        assert self.hf_model, "HF cache needs to be provided to obtain the embeddings tensor."
        if weight_name is None:
            base_name = self.hf_model.language_model_param_base_name
            weight_name = f"{base_name}.embed_tokens.weight"
            if self.cfg.lm_cfg.arch == LlmArchType.GEMMA:
                embed_scale = self.cfg.lm_cfg.hidden_size ** 0.5
        is_draft = (
            self.cfg.lm_cfg.speculative_decoding_cfg is not None
            and self.cfg.lm_cfg.speculative_decoding_cfg.is_draft
        )
        if is_draft and weight_name not in self.hf_model.weight_map:
            return None, None
        if isinstance(self.hf_model, LocalHuggingFaceModel):
            embeddings_tensor = self.hf_model.load_np_param(weight_name)
        else:
            assert isinstance(self.hf_model, GgufModel)
            embeddings_tensor = self.hf_model.load_np_param(weight_name, force_float=True)
        embeddings_tensor = embeddings_tensor.astype(np.float32)
        embeddings_tensor *= embed_scale

        if self.cfg.pipeline_cfg.quantize_embeddings:
            embeddings_tensor = embeddings_tensor.astype(bfloat16).astype(np.float32)
            return quantize_embedding_rows(embeddings_tensor)

        return embeddings_tensor.astype(bfloat16).astype(np.float32), None

    def get_input_embeddings_tensor(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.get_embeddings_tensor()

    def get_per_layer_embeddings_tensor(self) -> tuple[np.ndarray, np.ndarray | None]:
        base_name = self.hf_model.language_model_param_base_name
        embeddings, scale = self.get_embeddings_tensor(
            weight_name=f"{base_name}.embed_tokens_per_layer.weight",
            embed_scale=(
                self.cfg.lm_cfg.hidden_size_per_layer_input ** 0.5
                if self.cfg.lm_cfg.arch == LlmArchType.GEMMA
                else 1.0
            ),
        )
        assert embeddings is not None
        return embeddings, scale

    @property
    def _single_model_num_tokens(self) -> int:
        if self.cfg.lm_cfg.speculative_decoding_cfg is None:
            return 1
        return self.cfg.lm_cfg.speculative_decoding_cfg.speculative_budget

    def _get_part_model(
        self, part: str, num_tokens: int, layer_idx: int | None = None,
        token_idx: int | None = None, expert_idx: int = -1,
    ) -> BaseModel:
        match part:
            case "pre":
                model_name = f"{self.model_name}_n{num_tokens}_pre_layer{layer_idx}"
                assert layer_idx is not None
                return LanguagePreModel(
                    self.cfg, model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                    hf_model=self.hf_model, num_tokens=num_tokens, layer_idx=layer_idx,
                )
            case "post":
                if expert_idx >= 0:
                    model_name = f"{self.model_name}_n{num_tokens}_post_layer{layer_idx}_expert{expert_idx}"
                else:
                    model_name = f"{self.model_name}_n{num_tokens}_post_layer{layer_idx}"
                assert layer_idx is not None
                return LanguagePostModel(
                    self.cfg, model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                    hf_model=self.hf_model, num_tokens=num_tokens, layer_idx=layer_idx,
                    final_softcapping=self.cfg.lm_cfg.final_logit_softcapping,
                    expert_idx=expert_idx,
                )
            case "router":
                model_name = f"{self.model_name}_n{num_tokens}_router_layer{layer_idx}"
                assert layer_idx is not None
                return LanguageMoeRouterModel(
                    self.cfg, model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                    hf_model=self.hf_model, num_tokens=num_tokens, layer_idx=layer_idx,
                )
            case "moe_weightedsum":
                model_name = f"{self.model_name}_n{num_tokens}_moe_weightedsum_layer{layer_idx}"
                assert layer_idx is not None
                return LanguageMoeWeightedSumModel(
                    self.cfg, model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                    hf_model=self.hf_model, num_tokens=num_tokens, layer_idx=layer_idx,
                    final_softcapping=self.cfg.lm_cfg.final_logit_softcapping,
                )
            case "cache":
                model_name = f"{self.model_name}_n{num_tokens}_cache_token{token_idx}"
                assert token_idx is not None
                return LanguageCacheModel(
                    self.cfg, model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                    hf_model=self.hf_model, num_tokens=num_tokens, token_idx=token_idx,
                    logit_softcapping=self.cfg.lm_cfg.attn_logit_softcapping
                )
            case "sliding_cache":
                model_name = f"{self.model_name}_n{num_tokens}_sliding_cache_token{token_idx}"
                assert token_idx is not None
                return LanguageCacheModel(
                    self.cfg, model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                    hf_model=self.hf_model, num_tokens=num_tokens, token_idx=token_idx,
                    logit_softcapping=self.cfg.lm_cfg.attn_logit_softcapping,
                    layer_type="sliding_attention"
                )
            case "conv_fused":
                model_name = f"{self.model_name}_n{num_tokens}_layer{layer_idx}_conv"
                assert layer_idx is not None
                return LanguageConvModel(
                    self.cfg, model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                    hf_model=self.hf_model, num_tokens=num_tokens, layer_idx=layer_idx,
                    final_softcapping=self.cfg.lm_cfg.final_logit_softcapping
                )
            case "conv_post_final":
                model_name = f"{self.model_name}_n{num_tokens}_post_layer{layer_idx}_conv_final"
                assert layer_idx is not None
                return LanguageConvPostModel(
                    self.cfg, model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                    hf_model=self.hf_model, num_tokens=num_tokens, layer_idx=layer_idx,
                    final_softcapping=self.cfg.lm_cfg.final_logit_softcapping
                )
            case "draft_fc":
                model_name = f"{self.model_name}_n{num_tokens}_draft_fc"
                return LanguageDraftFCModel(
                    self.cfg, model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                    hf_model=self.hf_model, num_tokens=num_tokens
                )
            case "per_layer":
                model_name = f"{self.model_name}_n{num_tokens}_per_layer"
                return LanguagePerLayerModel(
                    self.cfg, model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                    hf_model=self.hf_model, num_tokens=num_tokens,
                )
