import numpy as np
from dataclasses import dataclass

from afe.apis.defines import gen2_target, TensorDRAMLayout
from afe.backends.backends import Backend
from afe.ir.serializer import save_awesomenet
from afe.ir.defines import Status, get_expected_tensor_value
from afe.ir.node import AwesomeNode
from afe.ir.tensor_type import TensorType, ScalarType

from sima_lmm.model.base import TensorTessellateParameters, LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import SimaBuilder, activation_type
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
    def _future_token_mask_size(self) -> int:
        return self.cfg.pipeline_cfg.get_future_token_mask_size(self.layer_type)

    @property
    def _uses_group_future_token_mask(self) -> bool:
        return (
            not self._is_speculative_decoding
            and self.num_tokens == self.cfg.pipeline_cfg.input_token_group_size
            and self.cfg.pipeline_cfg.uses_group_future_token_mask(self.layer_type)
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

    def gen_onnx_files(self):
        base_name = f"{self.hf_model.language_model_param_base_name}.token.{self.token_idx}"

        self.create_onnx_builder()

        self._onnx_builder.create_input_node(
            "query",
            (
                1,
                self._head_dim,
                self.cfg.lm_cfg.attn_cfg.num_attention_heads,
                self.num_tokens
            )
        )

        kv_cache_shape =  (
            1,
            self._head_dim,
            self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
            self.context_length
        )
        self._onnx_builder.create_input_node(f"cached_keys", kv_cache_shape)
        if (
            (self.cfg.model_type == VlmArchType.VLM_PALIGEMMA and self.num_tokens > 1)
            or self._is_speculative_decoding
            or self._uses_group_future_token_mask
        ):
            # For paligemma, the attention mask is dynamically determined.
            # For speculative decoding, the attention mask is dynamically determined during decode time.
            self._onnx_builder.create_input_node(
                "attn_mask", (1, self.context_length, 1, self.num_tokens)
            )
        elif self._future_token_mask_size > 1 and self.num_tokens == 1:
            # Enable the future attention mask to reduce the total number of cache models.
            self._onnx_builder.create_input_node("attn_mask", (1, self.token_idx + 1, 1, 1))
        self._onnx_builder.create_input_node(f"cached_values", kv_cache_shape)

        output_nodes = self._build_onnx_nodes(base_name, self._onnx_builder.input_nodes)
        output_name = self._onnx_builder.get_node_output_name(output_nodes[0])
        self._onnx_builder.create_output_node(
            output_name, (1, self._q_size, 1, self.num_tokens)
        )
        self._onnx_builder.create_and_save_model()

        # Set to None to deallocate the memory.
        self._onnx_builder = None

    def _build_reshape_kv(self, base_name: str, input_nodes: list[OnnxNode]):
        # Expansion of K or V to match number of attention heads:
        # (1, Head_Dim, n_kv, n_tokens) -> (1, Head_Dim, n_heads, n_tokens).
        attn_heads = self.cfg.lm_cfg.attn_cfg.num_attention_heads
        kv_heads = self.cfg.lm_cfg.attn_cfg.num_key_value_heads
        assert attn_heads % kv_heads == 0
        expansion_factor = attn_heads // kv_heads

        assert len(input_nodes) == 1
        kv_concat_shape = (
            1,
            self._head_dim,
            attn_heads,
            self.context_length
        )
        reshape_kv = self._onnx_builder.build_split_expand_concat(
            f"{base_name}.reshape", input_nodes[0], kv_heads, expansion_factor,
            split_axis=2, concat_axis=2, concat_shape=kv_concat_shape
        )

        return [reshape_kv]

    def _build_onnx_nodes(self, base_name: str, input_nodes: list[OnnxNode]) -> list[OnnxNode]:
        query = input_nodes[0]

        # Expansion of KV to match number of attention heads
        keys = self._build_reshape_kv(f"{base_name}.cached_keys", [input_nodes[1]])
        values = self._build_reshape_kv(f"{base_name}.cached_values", [input_nodes[-1]])

        assert len(keys) == len(values) == 1
        bmm1 = self._onnx_builder.build_op(
            f"{base_name}.bmm1", [query, keys[0]], "Einsum",
            equation="nchw,nchq->nqhw"
        )

        if self.logit_softcapping is not None:
            assert self.cfg.lm_cfg.arch == LlmArchType.GEMMA and self.cfg.lm_cfg.model_type == "gemma2"
            bmm1 = self._onnx_builder.build_logit_softcapping(
                f"{base_name}.softcap", bmm1, self.cfg.lm_cfg.attn_logit_softcapping
            )

        if self.num_tokens > 1:
            if (
                self.cfg.model_type == VlmArchType.VLM_PALIGEMMA
                or self._is_speculative_decoding
                or self._uses_group_future_token_mask
            ):
                # For paligemma, the attention mask is dynamically determined.
                # Speculative decoding uses num_tokens > 1 during decoding.
                bmm1 = self._onnx_builder.build_op(
                    f"{base_name}.masked_bmm1", [bmm1, input_nodes[2]], "Add"
                )
            else:
                mask = np.zeros((1, self.context_length, 1, self.num_tokens), dtype=np.float32)
                for i in range(self.num_tokens):
                    for j in range(self.token_idx + i + 1, self.context_length):
                        mask[0, j, 0, i] = np.finfo(np.float32).min
                bmm1 = self._onnx_builder.build_op(f"{base_name}.masked_bmm1", [bmm1, mask], "Add")
        elif self._future_token_mask_size > 1:
            bmm1 = self._onnx_builder.build_op(
                f"{base_name}.masked_bmm1", [bmm1, input_nodes[2]], "Add"
            )
        softmax = self._onnx_builder.build_op(f"{base_name}.softmax", [bmm1], "Softmax", axis=1)
        reduction_ranges = _get_bmm2_reduction_ranges(self.context_length)
        if len(reduction_ranges) == 1:
            bmm2 = self._onnx_builder.build_op(
                f"{base_name}.bmm2", [softmax, values[0]], "Einsum",
                equation="nchw,nqhc->nqhw"
            )
        else:
            partial_bmm2 = []
            for range_idx, (start, end) in enumerate(reduction_ranges):
                slice_args = [
                    np.array([start], dtype=np.int64),
                    np.array([end], dtype=np.int64),
                ]
                softmax_slice = self._onnx_builder.build_op(
                    f"{base_name}.bmm2.softmax_slice{range_idx}",
                    [softmax, *slice_args, np.array([1], dtype=np.int64)],
                    "Slice",
                )
                values_slice = self._onnx_builder.build_op(
                    f"{base_name}.bmm2.values_slice{range_idx}",
                    [values[0], *slice_args, np.array([3], dtype=np.int64)],
                    "Slice",
                )
                partial_bmm2.append(
                    self._onnx_builder.build_op(
                        f"{base_name}.bmm2.partial{range_idx}",
                        [softmax_slice, values_slice],
                        "Einsum",
                        equation="nchw,nqhc->nqhw",
                    )
                )

            add_level = 0
            while len(partial_bmm2) > 1:
                next_level = [
                    self._onnx_builder.build_op(
                        f"{base_name}.bmm2.add_level{add_level}_pair{pair_idx}",
                        [lhs, rhs],
                        "Add",
                    )
                    for pair_idx, (lhs, rhs) in enumerate(
                        zip(partial_bmm2[::2], partial_bmm2[1::2])
                    )
                ]
                if len(partial_bmm2) % 2:
                    next_level.append(partial_bmm2[-1])
                partial_bmm2 = next_level
                add_level += 1
            bmm2 = partial_bmm2[0]

        reshape_bmm2 = self._onnx_builder.build_split_and_concat(
            f"{base_name}.bmm2.reshape", bmm2,
            self.cfg.lm_cfg.attn_cfg.num_attention_heads,
            split_axis=2, concat_axis=1
        )
        return [reshape_bmm2]

    def gen_model_sdk_files_directly(
        self,
        layer_cfg: LayerConfiguration,
        log_level: int,
        quantizable: bool,
    ):
        base_name = f"{self.hf_model.language_model_param_base_name}.token.{self.token_idx}"
        g = self._build_sima_nodes(base_name, quantizable)
        save_awesomenet(g, self.model_name + (".fp32" if quantizable else ""), str(self.sima_model_sdk_path))

    def _build_sima_nodes(self, base_name: str, quantizable: bool):
        # Expansion of KV to match number of attention heads:
        # (1, Head_Dim, n_kv, n_tokens) -> (1, Head_Dim, n_heads, n_tokens).
        assert (
            self.cfg.lm_cfg.attn_cfg.num_attention_heads
            % self.cfg.lm_cfg.attn_cfg.num_key_value_heads == 0
        )
        quantize_kv_cache = self.cfg.pipeline_cfg.quantize_kv_cache

        # Shape of input key and value tensors.
        kv_tensor_shape = (
            1,
            self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
            self.context_length,
            self._head_dim
        )
        # Shape of scale tensors for quantized KV cache (per-token)
        kv_scale_shape = (
            1,
            self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
            self.context_length,
            1
        )
        input_shape = (
            1,
            self.cfg.lm_cfg.attn_cfg.num_attention_heads,
            self.num_tokens,
            self._head_dim
        )
        output_shape = (1, 1, self.num_tokens, self._q_size)

        # Shape of the result of the first matrix multiply (input * key)
        key_shape = (
            1,
            self.cfg.lm_cfg.attn_cfg.num_attention_heads,
            self.num_tokens,
            self.context_length
        )

        # Shape of the result of the second matrix multiply ((input * key) * value)
        value_shape = (
            1,
            self.cfg.lm_cfg.attn_cfg.num_attention_heads,
            self.num_tokens,
            self._head_dim
        )

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

        # Begin constructing a model graph
        builder = SimaBuilder(Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target)

        model_input_input = builder.create_placeholder_node(
            "input", TensorType(activation_type(quantizable), input_shape)
        )
        kv_dtype = ScalarType.int8 if quantize_kv_cache else activation_type(quantizable)
        model_input_cached_keys = builder.create_placeholder_node(
            "cached_keys", TensorType(kv_dtype, kv_tensor_shape)
        )
        if quantize_kv_cache:
            model_input_cached_keys_scale = builder.create_placeholder_node(
                "cached_keys_scale", TensorType(activation_type(quantizable), kv_scale_shape)
            )
        else:
            model_input_cached_keys_scale = None
        if (self.cfg.model_type == VlmArchType.VLM_PALIGEMMA and self.num_tokens > 1 or
            self._future_token_mask_size > 1 and self.num_tokens == 1 or
            self._is_speculative_decoding or self._uses_group_future_token_mask):
            # Dynamically computed attention mask for paligemma
            # Dynamically computed attention mask for speculative decoding
            # or the model runner's mask to remove the influence of future tokens
            model_input_attn_mask = builder.create_placeholder_node(
                "attn_mask", TensorType(activation_type(quantizable), attn_shape)
            )
        else:
            model_input_attn_mask = None
        model_input_cached_values = builder.create_placeholder_node(
            "cached_values", TensorType(kv_dtype, kv_tensor_shape)
        )
        if quantize_kv_cache:
            model_input_cached_values_scale = builder.create_placeholder_node(
                "cached_values_scale", TensorType(activation_type(quantizable), kv_scale_shape)
            )
        else:
            model_input_cached_values_scale = None

        model_inputs = list(filter(None, [
            model_input_input,
            model_input_cached_keys,
            model_input_cached_keys_scale,
            model_input_attn_mask,
            model_input_cached_values,
            model_input_cached_values_scale
        ]))

        builder.begin_subnet(model_inputs)

        # MLA subgraph inputs are the same as the model inputs, except the node names are different
        mla_input_input = builder.create_placeholder_node(
            "MLA_0/input", TensorType(activation_type(quantizable), input_shape)
        )
        mla_input_cached_keys = builder.create_placeholder_node(
            "MLA_0/cached_keys", TensorType(kv_dtype, kv_tensor_shape)
        )
        if quantize_kv_cache:
            mla_input_cached_keys_scale = builder.create_placeholder_node(
                "MLA_0/cached_keys_scale", TensorType(activation_type(quantizable), kv_scale_shape)
            )
        if model_input_attn_mask is not None:
            mla_input_attn_mask = builder.create_placeholder_node(
                "MLA_0/attn_mask", TensorType(activation_type(quantizable), attn_shape)
            )
        else:
            mla_input_attn_mask = None
        mla_input_cached_values = builder.create_placeholder_node(
            "MLA_0/cached_values", TensorType(kv_dtype, kv_tensor_shape)
        )
        if quantize_kv_cache:
            mla_input_cached_values_scale = builder.create_placeholder_node(
                "MLA_0/cached_values_scale",
                TensorType(activation_type(quantizable), kv_scale_shape)
            )

        # Dequantize KV cache if needed.
        if quantize_kv_cache:
            mla_input_cached_keys = builder.create_dynamic_dequant_node(
                mla_input_cached_keys, mla_input_cached_keys_scale
            )
            mla_input_cached_values = builder.create_dynamic_dequant_node(
                mla_input_cached_values, mla_input_cached_values_scale
            )

        # First multiply (input * key)
        # BatchMatMul repeats the smaller H dimension for GQA.
        bmm1 = builder.create_batch_matmul_node(
            mla_input_input, mla_input_cached_keys, transpose_a=False, transpose_b=True
        )
        assert get_expected_tensor_value(bmm1.get_type().output).shape == key_shape

        if self.logit_softcapping is not None:
            assert self.cfg.lm_cfg.arch == LlmArchType.GEMMA and self.cfg.lm_cfg.model_type == "gemma2"
            soft_cap = self.cfg.lm_cfg.attn_logit_softcapping
            mul_const_1 = builder.create_constant_node(
                np.ndarray([2.0 / soft_cap], dtype=np.float32)
            )
            mul1 = builder.create_mul_node(bmm1, mul_const_1)
            sig = builder.create_sigmoid_node(mul1)
            mul_const_2 = builder.create_constant_node(
                np.ndarray([2.0 * soft_cap], dtype=np.float32)
            )
            mul2 = builder.create_mul_node(sig, mul_const_2)
            sub_const_1 = builder.create_constant_node(np.ndarray([-soft_cap], dtype=np.float32))
            last = builder.create_add_node(mul2, sub_const_1)
            bmm1 = last

        if self.num_tokens > 1:
            if (
                self.cfg.model_type == VlmArchType.VLM_PALIGEMMA
                or self._is_speculative_decoding
                or self._uses_group_future_token_mask
            ):
                # For paligemma, the attention mask is dynamically determined.
                # Speculative decoding uses num_tokens > 1 during decode time.
                assert mla_input_attn_mask is not None
                bmm1 = builder.create_add_node(bmm1, mla_input_attn_mask)
            else:
                # Attention mask is a static constant.
                mask = np.zeros((1, 1, self.num_tokens, self.context_length), dtype=np.float32)
                for i in range(self.num_tokens):
                    for j in range(self.token_idx + i + 1, self.context_length):
                        mask[0, 0, i, j] = np.finfo(np.float32).min
                mask_const = builder.create_constant_node(
                    mask.astype(ScalarType.numpy_type(activation_type(quantizable)))
                )
                bmm1 = builder.create_add_node(bmm1, mask_const)
        elif self._future_token_mask_size > 1:
            assert mla_input_attn_mask is not None
            bmm1 = builder.create_add_node(bmm1, mla_input_attn_mask)

        softmax = builder.create_softmax_node(bmm1, 3)

        # Second multiply ((input * key) * value)
        reduction_ranges = _get_bmm2_reduction_ranges(self.context_length)
        if len(reduction_ranges) == 1:
            bmm2 = builder.create_batch_matmul_node(
                softmax, mla_input_cached_values, transpose_a=False, transpose_b=False
            )
        else:
            partial_bmm2 = []
            for start, end in reduction_ranges:
                softmax_slice = builder.create_slice_node(softmax, [start], [end], [1], [3])
                values_slice = builder.create_slice_node(
                    mla_input_cached_values, [start], [end], [1], [2]
                )
                partial_bmm2.append(
                    builder.create_batch_matmul_node(
                        softmax_slice, values_slice, transpose_a=False, transpose_b=False
                    )
                )

            while len(partial_bmm2) > 1:
                next_level = [
                    builder.create_add_node(lhs, rhs)
                    for lhs, rhs in zip(partial_bmm2[::2], partial_bmm2[1::2])
                ]
                if len(partial_bmm2) % 2:
                    next_level.append(partial_bmm2[-1])
                partial_bmm2 = next_level
            bmm2 = partial_bmm2[0]
        assert get_expected_tensor_value(bmm2.get_type().output).shape == value_shape
        output = builder.create_slice_concat_node(
            bmm2, axis=3, split_axis=1,
            split_block=self.cfg.lm_cfg.attn_cfg.num_attention_heads, split_repeat=1
        )
        assert get_expected_tensor_value(output.get_type().output).shape == output_shape

        mla_node = builder.finish_subnet("MLA_0")

        # Ensure that output type is float32
        if activation_type(quantizable) != ScalarType.float32:
            _ = builder.create_cast_node(mla_node, ScalarType.float32, backend=Backend.EV)
        net = builder.finish(self.model_name)
        return net

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """
        Get the custom tessellate params for model's inputs on the MLA.
        """
        tessellate_params = {}

        # Input order: [input, cached_keys, (cached_keys_scale), (attn_mask), cached_values,
        # (cached_values_scale)]

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
        if (self.cfg.model_type == VlmArchType.VLM_PALIGEMMA and self.num_tokens > 1) or \
                (self._future_token_mask_size > 1 and self.num_tokens == 1) or \
                self._is_speculative_decoding or self._uses_group_future_token_mask:
            attn_mask_tessellate_params = TensorTessellateParameters(
                tile_shape=(0, 0, 0, 0),
                enable_mla=True,
                dram_layout=TensorDRAMLayout.HWC
            )
            tessellate_params[idx] = attn_mask_tessellate_params
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

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """
        Get the custom tessellate params for model's output on the MLA.
        """
        # Use default tessellate params.
        return {}
