import numpy as np
from dataclasses import dataclass

from afe.apis.defines import TensorDRAMLayout
from afe.ir.defines import get_expected_tensor_value

from sima_lmm.model.base import TensorTessellateParameters, LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.model.model_graph import ModelGraph
from sima_lmm.config.vlm_config import LlmArchType, VlmArchType


_BMM2_REDUCTION_SPLIT_THRESHOLD = 2048
_BMM2_REDUCTION_CHUNK_SIZE = 1024


def _get_bmm2_reduction_ranges(context_length: int) -> list[tuple[int, int]]:
    """Return contiguous cache ranges for the second attention BMM."""
    assert context_length > 0
    if context_length <= _BMM2_REDUCTION_SPLIT_THRESHOLD:
        return [(0, context_length)]
    return [
        (start, min(start + _BMM2_REDUCTION_CHUNK_SIZE, context_length))
        for start in range(0, context_length, _BMM2_REDUCTION_CHUNK_SIZE)
    ]


@dataclass
class LanguageCacheModel(LanguagePartBaseModel):
    """Base implementation for the cache model of the language model.

    With support for Sliding Window Attention, a cache model has two flavors,
    depending on layer index: global cache or local cache. Because the cache
    is managed outside the cache model, the difference is reflected by input
    shapes of K and V tensors.

    Attributes:
        num_tokens: Number of tokens. Set to a value greater than 1 to consume multiple input tokens
            in one model.
        token_idx: Token index.
        logit_softcapping: Attention logit soft capping for gemma 2.
    """
    num_tokens: int
    token_idx: int
    logit_softcapping: float | None
    layer_type: str = "full_attention"

    def __post_init__(self):
        assert self.num_tokens >= 1

    @property
    def _is_speculative_decoding(self) -> bool:
        return (self.cfg.lm_cfg.speculative_decoding_cfg is not None
                and self.num_tokens == self.cfg.lm_cfg.speculative_decoding_cfg.speculative_budget)

    @property
    def _is_group_model(self) -> bool:
        speculative_cfg = self.cfg.lm_cfg.speculative_decoding_cfg
        single_num_tokens = (
            speculative_cfg.speculative_budget if speculative_cfg is not None else 1
        )
        return (
            bool(self.cfg.pipeline_cfg.input_token_group_offsets)
            and self.num_tokens == self.cfg.pipeline_cfg.input_token_group_size
            and self.num_tokens != single_num_tokens
        )

    @property
    def _cache_mask_size(self) -> int:
        return self.cfg.pipeline_cfg.get_cache_mask_size(
            self.layer_type, self.context_length, is_group=self._is_group_model
        )

    @property
    def _uses_group_future_token_mask(self) -> bool:
        return (
            self._is_group_model
            and self._cache_mask_size > self.cfg.pipeline_cfg.input_token_group_size
        )

    @property
    def context_length(self) -> int:
        if self._is_speculative_decoding:
            return self.token_idx + 1
        return self.token_idx + self.num_tokens

    @property
    def _head_dim(self) -> int:
        return self.cfg.lm_cfg.attn_cfg.get_head_dim(self.layer_type)

    @property
    def _q_size(self) -> int:
        return self.cfg.lm_cfg.attn_cfg.get_q_size(self.layer_type)

    @property
    def _kv_size(self) -> int:
        return self.cfg.lm_cfg.attn_cfg.get_kv_size(self.layer_type)

    def generate_graph(
        self,
        layer_cfg: LayerConfiguration,
        quantizable: bool,
    ):
        assert (
            self.cfg.lm_cfg.attn_cfg.num_attention_heads % self.cfg.lm_cfg.attn_cfg.num_key_value_heads
            == 0
        )
        quantize_kv_cache = self.cfg.pipeline_cfg.quantize_kv_cache

        # Shape of input key and value tensors.
        kv_tensor_shape = (
            1,
            self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
            self.context_length,
            self._head_dim,
        )
        # Shape of scale tensors for quantized KV cache (per-token)
        kv_scale_shape = (1, self.cfg.lm_cfg.attn_cfg.num_key_value_heads, self.context_length, 1)
        input_shape = (1, self.cfg.lm_cfg.attn_cfg.num_attention_heads, self.num_tokens, self._head_dim)
        output_shape = (1, 1, self.num_tokens, self._q_size)

        # Shape of the result of the first matrix multiply (input * key)
        key_shape = (
            1,
            self.cfg.lm_cfg.attn_cfg.num_attention_heads,
            self.num_tokens,
            self.context_length,
        )

        # Shape of the result of the second matrix multiply ((input * key) * value)
        value_shape = (1, self.cfg.lm_cfg.attn_cfg.num_attention_heads, self.num_tokens, self._head_dim)

        # Shape of the attention mask
        if (
            (self.cfg.model_type == VlmArchType.VLM_PALIGEMMA and self.num_tokens > 1)
            or self._is_speculative_decoding
            or self._uses_group_future_token_mask
        ):
            # Paligemma uses a special attention mask
            # Speculative decoding uses num_tokens > 1 for the target model during decoding.
            attn_shape = (1, 1, self.num_tokens, self.context_length)
        else:
            # Other models use an attention mask with a single value for each token,
            # or they don't use an attention mask
            attn_shape = (1, 1, 1, self.token_idx + 1)

        input_specs = {
            "input": input_shape,
            "cached_keys": kv_tensor_shape,
        }
        if quantize_kv_cache:
            input_specs["cached_keys_scale"] = kv_scale_shape
        if (
            self.cfg.model_type == VlmArchType.VLM_PALIGEMMA
            and self.num_tokens > 1
            or self._cache_mask_size > 1
            and self.num_tokens == 1
            or self._is_speculative_decoding
            or self._uses_group_future_token_mask
        ):
            input_specs["attn_mask"] = attn_shape
        if self.cfg.lm_cfg.arch == LlmArchType.GPT_OSS:
            input_specs["sinks"] = (1, 1, self.num_tokens, self.cfg.lm_cfg.attn_cfg.num_attention_heads)
        input_specs["cached_values"] = kv_tensor_shape
        if quantize_kv_cache:
            input_specs["cached_values_scale"] = kv_scale_shape
        graph = ModelGraph(
            self, input_specs, quantizable,
            input_dtypes={"cached_keys": np.int8, "cached_values": np.int8} if quantize_kv_cache else None,
        )
        inputs = graph.inputs
        mla_input_input = inputs["input"]
        mla_input_cached_keys = inputs["cached_keys"]
        mla_input_cached_values = inputs["cached_values"]
        mla_input_attn_mask = inputs.get("attn_mask")

        # Dequantize KV cache if needed.
        if quantize_kv_cache:
            mla_input_cached_keys = graph.dequant(mla_input_cached_keys, inputs["cached_keys_scale"])
            mla_input_cached_values = graph.dequant(
                mla_input_cached_values, inputs["cached_values_scale"]
            )

        # First multiply (input * key)
        # BatchMatMul repeats the smaller H dimension for GQA.
        bmm1 = graph.matmul(
            mla_input_input, mla_input_cached_keys, transpose_b=True
        )
        assert get_expected_tensor_value(bmm1.get_type().output).shape == key_shape

        if self.logit_softcapping is not None:
            assert self.cfg.lm_cfg.arch == LlmArchType.GEMMA and self.cfg.lm_cfg.model_type == "gemma2"
            bmm1 = graph.softcap(bmm1, self.cfg.lm_cfg.attn_logit_softcapping)

        if self.num_tokens > 1:
            if (
                self.cfg.model_type == VlmArchType.VLM_PALIGEMMA
                or self._is_speculative_decoding
                or self._uses_group_future_token_mask
            ):
                # For paligemma, the attention mask is dynamically determined.
                # Speculative decoding uses num_tokens > 1 during decode time.
                assert mla_input_attn_mask is not None
                bmm1 = graph.add(bmm1, mla_input_attn_mask)
            else:
                # Attention mask is a static constant.
                mask = np.zeros((1, 1, self.num_tokens, self.context_length), dtype=np.float32)
                for i in range(self.num_tokens):
                    for j in range(self.token_idx + i + 1, self.context_length):
                        mask[0, 0, i, j] = np.finfo(np.float32).min
                mask_const = graph.constant(mask)
                bmm1 = graph.add(bmm1, mask_const)
        elif self._cache_mask_size > 1:
            assert mla_input_attn_mask is not None
            bmm1 = graph.add(bmm1, mla_input_attn_mask)

        if self.cfg.lm_cfg.arch == LlmArchType.GPT_OSS:
            sinks = graph.split_concat(
                inputs["sinks"], axis=1, split_axis=3,
                split_block=self.cfg.lm_cfg.attn_cfg.num_attention_heads, split_repeat=1,
            )
            probabilities = graph.softmax(graph.concat([bmm1, sinks], axis=3))
            softmax = graph.slice(probabilities, start=0, stop=self.context_length, axis=3)
        else:
            softmax = graph.softmax(bmm1, 3)

        # Second multiply ((input * key) * value)
        reduction_ranges = _get_bmm2_reduction_ranges(self.context_length)
        if len(reduction_ranges) == 1:
            bmm2 = graph.matmul(softmax, mla_input_cached_values)
        else:
            partial_bmm2 = []
            for start, end in reduction_ranges:
                softmax_slice = graph.slice(softmax, start=start, stop=end, axis=3)
                values_slice = graph.slice(
                    mla_input_cached_values, start=start, stop=end, axis=2
                )
                partial_bmm2.append(
                    graph.matmul(softmax_slice, values_slice)
                )

            while len(partial_bmm2) > 1:
                next_level = [
                    graph.add(lhs, rhs)
                    for lhs, rhs in zip(partial_bmm2[::2], partial_bmm2[1::2])
                ]
                if len(partial_bmm2) % 2:
                    next_level.append(partial_bmm2[-1])
                partial_bmm2 = next_level
            bmm2 = partial_bmm2[0]
        assert get_expected_tensor_value(bmm2.get_type().output).shape == value_shape
        output = graph.merge_heads(bmm2)
        assert get_expected_tensor_value(output.get_type().output).shape == output_shape

        graph.save([output])

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """
        Get the custom tessellate params for model's inputs on the MLA.
        """
        tessellate_params = {}

        # Input order: input, keys, optional key scales/mask/sinks, values, optional value scales.

        # cached_keys
        idx = 1
        # Define DRAM shape to enable strided KV cache access for kv cache outputs.
        dram_shape = (
            1,
            self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
            max(self.context_length, self.cfg.pipeline_cfg.max_num_tokens),
            self._head_dim
        )
        k_cache_tessellate_params = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC16,
            dram_shape=dram_shape
        )
        tessellate_params[idx] = k_cache_tessellate_params
        idx += 1

        # cached_keys_scale
        quantize_kv_cache = self.cfg.pipeline_cfg.quantize_kv_cache
        if quantize_kv_cache:
            scale_dram_shape = (
                1,
                self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
                max(self.context_length, self.cfg.pipeline_cfg.max_num_tokens),
                1
            )
            k_scale_params = TensorTessellateParameters(
                tile_shape=(0, 0, 0, 0),
                enable_mla=True,
                dram_layout=TensorDRAMLayout.HWC16,
                dram_shape=scale_dram_shape
            )
            tessellate_params[idx] = k_scale_params
            idx += 1

        # attn_mask
        if (self.cfg.model_type == VlmArchType.VLM_PALIGEMMA and self.num_tokens > 1) or\
                (self._cache_mask_size > 1 and self.num_tokens == 1) or\
                self._is_speculative_decoding or self._uses_group_future_token_mask:
            attn_mask_tessellate_params = TensorTessellateParameters(
                tile_shape=(0, 0, 0, 0),
                enable_mla=True,
                dram_layout=TensorDRAMLayout.HWC
            )
            tessellate_params[idx] = attn_mask_tessellate_params
            idx += 1

        # Attention sinks precede values but do not use the strided cache layout.
        if self.cfg.lm_cfg.arch == LlmArchType.GPT_OSS:
            idx += 1

        # cached_values
        v_cache_tessellate_params = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC16,
            dram_shape=dram_shape
        )
        tessellate_params[idx] = v_cache_tessellate_params
        idx += 1

        # cached_values_scale
        if quantize_kv_cache:
            v_scale_params = TensorTessellateParameters(
                tile_shape=(0, 0, 0, 0),
                enable_mla=True,
                dram_layout=TensorDRAMLayout.HWC16,
                dram_shape=scale_dram_shape
            )
            tessellate_params[idx] = v_scale_params

        return tessellate_params
