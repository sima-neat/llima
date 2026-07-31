import logging
import numpy as np
import time

from dataclasses import asdict, dataclass, field

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
from sima_lmm.model.language_cache_model import LanguageCacheModel
from sima_lmm.model.language_conv_model import LanguageConvModel
from sima_lmm.model.language_conv_post_model import LanguageConvPostModel
from sima_lmm.model.language_draft_fc_model import LanguageDraftFCModel
from sima_lmm.model.language_per_layer_model import LanguagePerLayerModel
from sima_lmm.utils import calc_freq_real_imag, round_up_to
from sima_lmm.config.layer_id import LayerID
from sima_lmm.config.vlm_config import (
    AttentionBlockConfig,
    LlmArchType,
    PipelineConfig,
    VlmArchType,
)


bfloat16 = ScalarType.numpy_type(ScalarType.bfloat16)


@dataclass(frozen=True)
class ReferenceRuntimeStep:
    """One normal-generation invocation of the compiled language-model pipeline."""

    num_tokens: int
    token_idx: int
    use_input_tokens: bool
    emits_token: bool


@dataclass(frozen=True)
class ReferenceCachePlan:
    """C++-equivalent cache model and input slice for one attention layer."""

    token_idx_begin: int
    aligned_context: int
    model_token_idx: int
    layer_type: str
    use_future_mask: bool


def _reference_cache_plan(
    pipeline_cfg: PipelineConfig,
    attention_cfg: AttentionBlockConfig,
    layer_type: str,
    token_idx: int,
    num_tokens: int,
) -> ReferenceCachePlan:
    sliding_window = attention_cfg.sliding_window or 0
    token_idx_begin = (
        max(0, token_idx + num_tokens - sliding_window)
        if layer_type == "sliding_attention"
        else 0
    )
    effective_context = token_idx + num_tokens - token_idx_begin
    separate_sliding_cache = (
        layer_type == "sliding_attention"
        and attention_cfg.sliding_head_dim is not None
        and attention_cfg.sliding_head_dim != attention_cfg.head_dim
    )
    sliding_cache_mask_differs = (
        layer_type == "sliding_attention"
        and not separate_sliding_cache
        and (
            pipeline_cfg.get_cache_mask_size(
                "full_attention", sliding_window, is_group=True
            )
            != pipeline_cfg.get_cache_mask_size(
                "sliding_attention", sliding_window, is_group=True
            )
            or pipeline_cfg.get_cache_mask_size(
                "full_attention", sliding_window, is_group=False
            )
            != pipeline_cfg.get_cache_mask_size(
                "sliding_attention", sliding_window, is_group=False
            )
        )
    )
    use_sliding_cache = separate_sliding_cache or (
        sliding_cache_mask_differs and effective_context >= sliding_window
    )
    cache_layer_type = (
        "sliding_attention" if use_sliding_cache else "full_attention"
    )
    cache_mask_size = pipeline_cfg.get_cache_mask_size(
        cache_layer_type, effective_context, is_group=num_tokens > 1
    )
    use_future_mask = cache_mask_size > num_tokens
    aligned_context = (
        min(
            round_up_to(effective_context, cache_mask_size),
            pipeline_cfg.max_num_tokens,
        )
        if use_future_mask
        else effective_context
    )
    return ReferenceCachePlan(
        token_idx_begin=token_idx_begin,
        aligned_context=aligned_context,
        model_token_idx=aligned_context - num_tokens,
        layer_type=cache_layer_type,
        use_future_mask=use_future_mask,
    )


def _reference_runtime_steps(
    pipeline_cfg: PipelineConfig,
    num_input_tokens: int,
    max_new_tokens: int,
) -> list[ReferenceRuntimeStep]:
    """Build the same prefill/decode schedule as the normal C++ runtime."""
    if num_input_tokens <= 0:
        raise ValueError("At least one input token is required")
    if num_input_tokens > pipeline_cfg.max_num_tokens:
        raise ValueError(
            f"Input has {num_input_tokens} tokens, but max_num_tokens is "
            f"{pipeline_cfg.max_num_tokens}"
        )
    max_generated_tokens = pipeline_cfg.max_num_tokens - num_input_tokens + 1
    if max_new_tokens < 0 or max_new_tokens > max_generated_tokens:
        raise ValueError(
            f"max_new_tokens must be between 0 and {max_generated_tokens}, "
            f"got {max_new_tokens}"
        )
    if max_new_tokens == 0:
        return []

    prefill: list[ReferenceRuntimeStep] = []
    offsets = pipeline_cfg.input_token_group_offsets
    group_size = pipeline_cfg.input_token_group_size
    if offsets and group_size > 1:
        for token_idx in offsets:
            if token_idx >= num_input_tokens:
                break
            prefill.append(
                ReferenceRuntimeStep(group_size, token_idx, True, False)
            )
            if token_idx + group_size >= num_input_tokens:
                break
    else:
        prefill = [
            ReferenceRuntimeStep(1, token_idx, True, False)
            for token_idx in range(num_input_tokens)
        ]

    if not prefill:
        raise RuntimeError("No compiled model covers the input-token range")
    prefill[-1] = ReferenceRuntimeStep(
        prefill[-1].num_tokens,
        prefill[-1].token_idx,
        True,
        True,
    )
    decode = [
        ReferenceRuntimeStep(1, num_input_tokens + i, False, True)
        for i in range(max_new_tokens - 1)
    ]
    return [*prefill, *decode]


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
    _embeddings_scale: float | None = field(default=None, init=False, repr=False)
    _per_layer_embeddings_scale: float | None = field(default=None, init=False, repr=False)

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
                case _:
                    # Not a part of this model
                    continue
            curr_cfg = {"precision": curr_precision}
            if lora_mode:
                curr_cfg["lora"] = lora_mode[layer_id]
            model_list.append((part_model, curr_cfg))

        direct_graph_mode = gen_mode in [FileGenMode.SOURCE_TO_FP, FileGenMode.SOURCE_TO_QUANT]
        if direct_graph_mode and self.cfg.pipeline_cfg.quantize_embeddings:
            models_to_generate = [
                model for model, _ in model_list
                if not (resume and model.get_gen_file_name(gen_mode).is_file())
            ]
            input_dequant_models = [
                model for model in models_to_generate
                if (
                    isinstance(model, LanguagePerLayerModel)
                    or (
                        isinstance(model, (LanguagePreModel, LanguagePostModel))
                        and model.layer_idx == 0
                    )
                )
                and model.uses_quantized_input_embeddings
            ]
            if input_dequant_models:
                if self._embeddings_scale is None:
                    self.get_input_embeddings_tensor()
                assert self._embeddings_scale is not None
                self.cfg.pipeline_cfg.embeddings_scale = self._embeddings_scale
                for model in input_dequant_models:
                    model.embeddings_scale = self._embeddings_scale

            per_layer_models = [
                model for model in models_to_generate
                if isinstance(model, LanguagePerLayerModel)
            ]
            if per_layer_models:
                if self._per_layer_embeddings_scale is None:
                    self.get_per_layer_embeddings_tensor()
                assert self._per_layer_embeddings_scale is not None
                for model in per_layer_models:
                    model.per_layer_embeddings_scale = self._per_layer_embeddings_scale

        # Finished creating model_list.  Compile these models.
        self.gen_files_from_model_list(model_list, gen_mode, num_processes, log_level, resume)

    def run_model(
        self,
        eval_mode: EvalMode,
        ifms: list[np.ndarray],
        embeddings_tensor: np.ndarray | None = None,
        *,
        max_new_tokens: int | None = None,
    ) -> list[np.ndarray]:
        """Run the host reference pipeline using compiled ONNX or quantized ``.sima`` parts.

        This follows the normal (non-speculative) C++ prefill/decode schedule and cache-model
        selection. ``max_new_tokens`` limits generation without changing the compiled context
        length.
        """
        assert self.vlm_helper is not None
        assert len(ifms) == 1
        input_embeds = ifms[0]
        if embeddings_tensor is None:
            embeddings_tensor, _ = self.get_embeddings_tensor()
        assert embeddings_tensor is not None

        if self.cfg.lm_cfg.speculative_decoding_cfg is not None:
            raise NotImplementedError(
                "The Python reference runtime currently supports normal generation only"
            )
        unsupported_layer_types = set(self.cfg.lm_cfg.layer_types) - {
            "full_attention", "sliding_attention"
        }
        if unsupported_layer_types:
            raise NotImplementedError(
                "The Python reference runtime does not yet support layer types "
                f"{sorted(unsupported_layer_types)}"
            )
        if self.cfg.lm_cfg.hidden_size_per_layer_input:
            raise NotImplementedError(
                "The Python reference runtime does not yet support per-layer embedding inputs"
            )

        swa_enable = self.cfg.lm_cfg.attn_cfg.swa_enable
        layer_types = self.cfg.lm_cfg.layer_types
        global_freq_real, global_freq_imag = self.calc_freq_real_imag(False)
        if swa_enable:
            local_freq_real, local_freq_imag = self.calc_freq_real_imag(True)

        num_kv_heads = self.cfg.lm_cfg.attn_cfg.num_key_value_heads
        quantize_kv_cache = self.cfg.pipeline_cfg.quantize_kv_cache
        kv_dtype = np.int8 if quantize_kv_cache else np.float32
        cache_key = [
            np.zeros(
                (
                    1,
                    num_kv_heads,
                    self.cfg.pipeline_cfg.max_num_tokens,
                    self.cfg.lm_cfg.attn_cfg.get_head_dim(layer_type),
                ),
                dtype=kv_dtype,
            )
            for layer_type in layer_types
        ]
        cache_val = [
            np.zeros_like(layer_cache)
            for layer_cache in cache_key
        ]
        if quantize_kv_cache:
            cache_key_scale = [
                np.zeros(
                    (1, num_kv_heads, self.cfg.pipeline_cfg.max_num_tokens, 1),
                    dtype=np.float32,
                )
                for _ in range(self.cfg.lm_cfg.num_hidden_layers)
            ]
            cache_val_scale = [
                np.zeros(
                    (1, num_kv_heads, self.cfg.pipeline_cfg.max_num_tokens, 1),
                    dtype=np.float32,
                )
                for _ in range(self.cfg.lm_cfg.num_hidden_layers)
            ]

        num_input_tokens = input_embeds.shape[2]
        if max_new_tokens is None:
            max_new_tokens = self.cfg.pipeline_cfg.max_num_tokens - num_input_tokens + 1
        steps = _reference_runtime_steps(
            self.cfg.pipeline_cfg, num_input_tokens, max_new_tokens
        )
        padded_token_size = max(
            (
                step.token_idx + step.num_tokens
                for step in steps
                if step.use_input_tokens
            ),
            default=num_input_tokens,
        )
        if (padding := padded_token_size - num_input_tokens) > 0:
            input_embeds = np.pad(input_embeds, ((0, 0), (0, 0), (0, padding), (0, 0)))

        if (
            self.cfg.model_type == VlmArchType.VLM_PALIGEMMA
            and padded_token_size < num_input_tokens
        ):
            sima_log_warning(
                f"Number of input tokens ({num_input_tokens}) is greater than the maximum allowed"
                f" number of tokens ({padded_token_size}) for PaliGemma."
                "\nPlease increase the estimated_max_num_query_tokens in config_pipeline()."
            )

        perf_cnt_begin = time.perf_counter_ns()
        new_tokens = list()
        post_ofms = None
        new_token = None
        part_models: dict[tuple[str, int, int | None, int | None], BaseModel] = {}

        def get_part_model(
            part: str,
            num_tokens: int,
            *,
            layer_idx: int | None = None,
            token_idx: int | None = None,
        ) -> BaseModel:
            key = (part, num_tokens, layer_idx, token_idx)
            if key not in part_models:
                part_models[key] = self._get_part_model(
                    part, num_tokens, layer_idx=layer_idx, token_idx=token_idx
                )
            return part_models[key]

        for step in steps:
            token_idx = step.token_idx
            num_tokens = step.num_tokens
            next_token_idx = (
                min(num_input_tokens, token_idx + num_tokens)
                if num_tokens > 1 else token_idx + 1
            )
            sima_log_info("Processing token no. %d-%d", token_idx, next_token_idx)

            if not step.use_input_tokens:
                perf_cnt_begin = time.perf_counter_ns()

            for layer_idx in range(self.cfg.lm_cfg.num_hidden_layers):
                layer_type = layer_types[layer_idx]
                is_global = layer_type == "full_attention"
                pre_ifms = list()
                if layer_idx == 0:
                    if step.use_input_tokens:
                        pre_ifms.append(input_embeds[..., token_idx:token_idx + num_tokens, :])
                    else:
                        assert new_token is not None
                        pre_ifms.append(
                            np.expand_dims(embeddings_tensor[new_token], axis=(0, 1, 2))
                        )
                else:
                    assert post_ofms is not None
                    pre_ifms.append(post_ofms[0])
                if is_global:
                    pre_ifms.append(global_freq_real[..., token_idx:token_idx + num_tokens, :])
                    pre_ifms.append(global_freq_imag[..., token_idx:token_idx + num_tokens, :])
                else:
                    pre_ifms.append(local_freq_real[..., token_idx:token_idx + num_tokens, :])
                    pre_ifms.append(local_freq_imag[..., token_idx:token_idx + num_tokens, :])
                pre_model = get_part_model("pre", num_tokens, layer_idx=layer_idx)
                pre_ofms = pre_model.run_model(eval_mode, pre_ifms)

                if not self.cfg.lm_cfg.is_kv_shared_layer(layer_idx):
                    if quantize_kv_cache:
                        cache_key[layer_idx][..., token_idx:token_idx + num_tokens, :] = pre_ofms[1]
                        cache_key_scale[layer_idx][..., token_idx:token_idx + num_tokens, :] = (
                            pre_ofms[2]
                        )
                        cache_val[layer_idx][..., token_idx:token_idx + num_tokens, :] = pre_ofms[3]
                        cache_val_scale[layer_idx][..., token_idx:token_idx + num_tokens, :] = (
                            pre_ofms[4]
                        )
                    else:
                        cache_key[layer_idx][..., token_idx:token_idx + num_tokens, :] = pre_ofms[1]
                        cache_val[layer_idx][..., token_idx:token_idx + num_tokens, :] = pre_ofms[2]

                post_num_tokens = num_tokens
                post_row_idx = None
                if num_tokens > 1 and layer_idx == self.cfg.lm_cfg.num_hidden_layers - 1:
                    if token_idx + num_tokens >= num_input_tokens:
                        post_row_idx = num_input_tokens - 1 - token_idx
                        post_num_tokens = 1
                    else:
                        break

                cache_plan = _reference_cache_plan(
                    self.cfg.pipeline_cfg,
                    self.cfg.lm_cfg.attn_cfg,
                    layer_type,
                    token_idx,
                    num_tokens,
                )
                token_idx_begin = cache_plan.token_idx_begin
                aligned_context = cache_plan.aligned_context

                kv_source_layer = self.cfg.lm_cfg.get_kv_source_layer(layer_idx)
                cache_ifms = [
                    pre_ofms[0],
                    cache_key[kv_source_layer][
                        ..., token_idx_begin:token_idx_begin + aligned_context, :
                    ],
                ]
                if quantize_kv_cache:
                    cache_ifms.append(
                        cache_key_scale[kv_source_layer][
                            ..., token_idx_begin:token_idx_begin + aligned_context, :
                        ]
                    )
                if self.cfg.model_type == VlmArchType.VLM_PALIGEMMA and num_tokens > 1:
                    mask = np.zeros(
                        (1, 1, num_tokens, aligned_context), dtype=np.float32
                    )
                    mask[0, 0, :, num_input_tokens - token_idx:] = (
                        np.finfo(np.float32).min
                    )
                    cache_ifms.append(mask)
                elif cache_plan.use_future_mask:
                    mask = np.full(
                        (1, 1, num_tokens, aligned_context),
                        np.finfo(np.float32).min,
                        dtype=np.float32,
                    )
                    effective_token_idx = token_idx - token_idx_begin
                    for row in range(num_tokens):
                        mask[..., row, :effective_token_idx + row + 1] = 0
                    cache_ifms.append(mask)
                cache_ifms.append(
                    cache_val[kv_source_layer][
                        ..., token_idx_begin:token_idx_begin + aligned_context, :
                    ]
                )
                if quantize_kv_cache:
                    cache_ifms.append(
                        cache_val_scale[kv_source_layer][
                            ..., token_idx_begin:token_idx_begin + aligned_context, :
                        ]
                    )

                cache_model = get_part_model(
                    (
                        "sliding_cache"
                        if cache_plan.layer_type == "sliding_attention"
                        else "cache"
                    ),
                    num_tokens,
                    token_idx=cache_plan.model_token_idx,
                )
                cache_ofms = cache_model.run_model(eval_mode, cache_ifms)

                post_ifms = [pre_ifms[0], cache_ofms[0]]
                if post_row_idx is not None:
                    post_ifms = [
                        ofm[..., post_row_idx:post_row_idx + 1, :]
                        for ofm in post_ifms
                    ]
                post_model = get_part_model(
                    "post", post_num_tokens, layer_idx=layer_idx
                )
                post_ofms = post_model.run_model(eval_mode, post_ifms)

            if step.emits_token:
                assert post_ofms is not None
                if (
                    self.cfg.lm_cfg.lm_head_num_splits == 1
                    and not self.cfg.pipeline_cfg.return_logits
                ):
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

        return np.array([new_tokens])

    def calc_freq_real_imag(self, use_swa: bool) -> tuple[np.ndarray, np.ndarray]:
        if use_swa:
            theta = self.cfg.lm_cfg.rope_cfg.rope_local_base_freq
            rope_type = "default"
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
    ) -> tuple[np.ndarray | None, float | np.ndarray | None]:
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
            # Reshape to (1, 1, vocab_size, 1, hidden_size)
            embeddings_tensor = embeddings_tensor.reshape(
                1, 1, embeddings_tensor.shape[0], 1, embeddings_tensor.shape[1]
            )
            q_embeddings, scale = block_quantize_weight_tensor(
                embeddings_tensor, per_channel=False, bits=8, c_block_size=None
            )
            q_embeddings = q_embeddings.reshape(q_embeddings.shape[-3], q_embeddings.shape[-1])
            return q_embeddings, scale

        return embeddings_tensor.astype(bfloat16).astype(np.float32), None

    def get_input_embeddings_tensor(self) -> tuple[np.ndarray | None, float | None]:
        embeddings, scale = self.get_embeddings_tensor()
        if scale is not None:
            self._embeddings_scale = float(scale)
        return embeddings, self._embeddings_scale

    def get_per_layer_embeddings_tensor(self) -> tuple[np.ndarray, float | None]:
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
        if scale is not None:
            self._per_layer_embeddings_scale = float(scale)
        return embeddings, self._per_layer_embeddings_scale

    @property
    def _single_model_num_tokens(self) -> int:
        if self.cfg.lm_cfg.speculative_decoding_cfg is None:
            return 1
        return self.cfg.lm_cfg.speculative_decoding_cfg.speculative_budget

    def _get_part_model(
        self, part: str, num_tokens: int, layer_idx: int | None = None,
        token_idx: int | None = None,
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
                model_name = f"{self.model_name}_n{num_tokens}_post_layer{layer_idx}"
                assert layer_idx is not None
                return LanguagePostModel(
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
