import numpy as np
from dataclasses import dataclass

from afe.apis.defines import gen2_target, TensorDRAMLayout
from afe.backends.backends import Backend
from afe.ir.serializer import save_awesomenet
from afe.ir.defines import Status, get_expected_tensor_value
from afe.ir.node import AwesomeNode
from afe.ir.tensor_type import TensorType, ScalarType
from afe.ir.build_node import NodeHandle, NodeOrHandle

from sima_lmm.model.base import TensorTessellateParameters, LoraGenMode, LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import (
    SimaBuilder, build_conv_from_dense_with_lora, activation_type, activation_dtype,
    create_channel_slice
)
from sima_lmm.config.vlm_config import VlmArchType


bfloat16 = ScalarType.numpy_type(ScalarType.bfloat16)

@dataclass
class LanguagePreModel(LanguagePartBaseModel):
    """Base implementation for the pre cache model of the language model.

    Attributes:
        num_tokens: Number of tokens. Set to a value greater than 1 to consume multiple input tokens
            in one model.
        layer_idx: Transformer layer index.
        embeddings_scale: Scale factor used for de-quantization when embeddings quantization is
            enabled, None otherwise.
    """
    num_tokens: int
    layer_idx: int
    embeddings_scale: float | np.ndarray | None = None

    def __post_init__(self):
        assert self.num_tokens >= 1
        assert 0 <= self.layer_idx < self.cfg.lm_cfg.num_hidden_layers

    @property
    def enable_filter_sharing(self) -> bool:
        return self.cfg.pipeline_cfg.enable_filter_sharing

    @property
    def _layer_base_name(self) -> str:
        base = self.hf_model.language_model_param_base_name
        return base if self.is_draft else f"{base}.layers.{self.layer_idx}"
    
    @property
    def layer_type(self) -> str:
        return self.cfg.lm_cfg.layer_types[self.layer_idx]

    @property
    def _is_proportional_rope_layer(self) -> bool:
        return (
            self.layer_type == "full_attention"
            and self.cfg.lm_cfg.rope_cfg.rope_scaling.rope_type == "proportional"
        )

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
        base_name = self._layer_base_name
        self.create_onnx_builder()
        self._onnx_builder.create_input_node(
            "input", (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens)
        )
        if self.is_draft:
            self._onnx_builder.create_input_node(
                "hidden_states", (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens)
            )
        self._onnx_builder.create_input_node(
            "freq_real", (1, self.cfg.lm_cfg.rope_cfg.get_rope_dimension_count(self.layer_type) // 2, 1, self.num_tokens)
        )
        self._onnx_builder.create_input_node(
            "freq_imag", (1, self.cfg.lm_cfg.rope_cfg.get_rope_dimension_count(self.layer_type) // 2, 1, self.num_tokens)
        )
        output_nodes = self._build_onnx_nodes(base_name, self._onnx_builder.input_nodes)

        # RoPE embedded q_proj (1, Head_Dim, n_heads, n_tokens).
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[0]),
            (
                1,
                self._head_dim,
                self.cfg.lm_cfg.attn_cfg.num_attention_heads,
                self.num_tokens
            )
        )

        if not self.cfg.lm_cfg.is_kv_shared_layer(self.layer_idx):
            # RoPE embedded k_proj and v_proj (1, Head_Dim, n_kv, n_tokens).
            kv_cache_shape = (
                1,
                self._head_dim,
                self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
                self.num_tokens
            )
            self._onnx_builder.create_output_node(
                self._onnx_builder.get_node_output_name(output_nodes[1]), kv_cache_shape
            )
            self._onnx_builder.create_output_node(
                self._onnx_builder.get_node_output_name(output_nodes[2]), kv_cache_shape
            )

        self._onnx_builder.create_and_save_model()

        # Set to None to deallocate the memory.
        self._onnx_builder = None

    def _build_onnx_nodes(self, base_name: str, input_nodes: list[OnnxNode]) -> list[OnnxNode]:
        # LFM2 uses 'operator_norm' instead of 'input_layernorm'.
        norm_name = (
            f"{base_name}.operator_norm"
            if self.check_hf_param(f"{base_name}.operator_norm.weight")
            else f"{base_name}.input_layernorm"
        )
        rms_norm = self._build_rms_norm(norm_name, input_nodes[0])
        if self.is_draft:
            # EAGLE3 draft model also normalizes the target hidden_states.
            hidden_states_norm = self._build_rms_norm(
                f"{base_name}.hidden_norm", input_nodes[1]
            )
            attn_input = self._onnx_builder.build_op(
                base_name=f"{base_name}.concat",
                input_nodes=[rms_norm, hidden_states_norm],
                op_type="Concat",
                axis=1,
            )
            freq_start = 2
        else:
            attn_input = rms_norm
            freq_start = 1
        q_out = self._build_onnx_attn_query(f"{base_name}.self_attn", [attn_input, *input_nodes[freq_start:]])
        if self.cfg.lm_cfg.is_kv_shared_layer(self.layer_idx):
            return [q_out]
        k_out = self._build_onnx_attn_key(f"{base_name}.self_attn", [attn_input, *input_nodes[freq_start:]])
        v_out = self._build_onnx_attn_value(f"{base_name}.self_attn", attn_input)
        return [q_out, k_out, v_out]

    def _build_onnx_rotary_emb(self, base_name: str, input_nodes: list[OnnxNode]) -> OnnxNode:
        layer_head_dim = self._head_dim
        layer_rope_dimension_count = self.cfg.lm_cfg.rope_cfg.get_rope_dimension_count(self.layer_type)
        imag_start = (
            layer_head_dim // 2
            if self._is_proportional_rope_layer
            else layer_rope_dimension_count // 2
        )
        imag_end = (
            layer_head_dim // 2 + layer_rope_dimension_count // 2
            if self._is_proportional_rope_layer
            else layer_rope_dimension_count
        )
        real_in = self._onnx_builder.build_op(
            f"{base_name}.real_in",
            [
                input_nodes[0],
                np.array([0], dtype=np.int64),
                np.array([layer_rope_dimension_count // 2], dtype=np.int64),
                np.array([1], dtype=np.int64)
            ],
            "Slice"
        )
        imag_in = self._onnx_builder.build_op(
            f"{base_name}.imag_in",
            [
                input_nodes[0],
                np.array([imag_start], dtype=np.int64),
                np.array([imag_end], dtype=np.int64),
                np.array([1], dtype=np.int64)
            ],
            "Slice"
        )

        mul_rr = self._onnx_builder.build_op(
            f"{base_name}.mul_rr", [real_in, input_nodes[1]], "Mul"
        )
        mul_ii = self._onnx_builder.build_op(
            f"{base_name}.mul_ii", [imag_in, input_nodes[2]], "Mul"
        )
        real_out = self._onnx_builder.build_op(f"{base_name}.real_out", [mul_rr, mul_ii], "Sub")

        mul_ri = self._onnx_builder.build_op(
            f"{base_name}.mul_ri", [real_in, input_nodes[2]], "Mul"
        )
        mul_ir = self._onnx_builder.build_op(
            f"{base_name}.mul_ir", [imag_in, input_nodes[1]], "Mul"
        )
        imag_out = self._onnx_builder.build_op(f"{base_name}.imag_out", [mul_ri, mul_ir], "Add")
        rotary_out = self._onnx_builder.build_op(
            f"{base_name}.concat", [real_out, imag_out], "Concat", axis=1
        )
        if self._is_proportional_rope_layer:
            mid1 = self._onnx_builder.build_op(
                f"{base_name}.mid1",
                [
                    input_nodes[0],
                    np.array([layer_rope_dimension_count // 2], dtype=np.int64),
                    np.array([layer_head_dim // 2], dtype=np.int64),
                    np.array([1], dtype=np.int64)
                ],
                "Slice"
            )
            mid2 = self._onnx_builder.build_op(
                f"{base_name}.mid2",
                [
                    input_nodes[0],
                    np.array([layer_head_dim // 2 + layer_rope_dimension_count // 2], dtype=np.int64),
                    np.array([layer_head_dim], dtype=np.int64),
                    np.array([1], dtype=np.int64)
                ],
                "Slice"
            )
            return self._onnx_builder.build_op(
                f"{base_name}.concat_proportional",
                [real_out, mid1, imag_out, mid2],
                "Concat",
                axis=1,
            )

        if layer_rope_dimension_count == layer_head_dim:
            return rotary_out

        tail = self._onnx_builder.build_op(
            f"{base_name}.tail",
            [
                input_nodes[0],
                np.array([layer_rope_dimension_count], dtype=np.int64),
                np.array([layer_head_dim], dtype=np.int64),
                np.array([1], dtype=np.int64)
            ],
            "Slice"
        )
        return self._onnx_builder.build_op(
            f"{base_name}.concat_full", [rotary_out, tail], "Concat", axis=1
        )

    def _build_onnx_attn_query(self, base_name: str, input_nodes: list[OnnxNode]) -> OnnxNode:
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "q_proj")
        q_proj = self._onnx_builder.build_conv_from_dense_with_lora(
            f"{base_name}.q_proj", input_nodes[0], lora_rank,
            q_size=self._q_size,
            kv_size=self._kv_size
        )
        reshape = self._onnx_builder.build_split_and_concat(
            f"{base_name}.q_proj.reshape", q_proj, self.cfg.lm_cfg.attn_cfg.num_attention_heads,
            split_axis=1, concat_axis=2
        )

        q_norm_name = None
        for suffix in ("q_layernorm", "q_norm"):
            if self.check_hf_param(f"{base_name}.{suffix}.weight"):
                q_norm_name = f"{base_name}.{suffix}"
                break

        if q_norm_name:
            reshape = self._build_rms_norm(q_norm_name, reshape)

        rotary_emb = self._build_onnx_rotary_emb(
            f"{base_name}.q_proj.rotary", [reshape, input_nodes[1], input_nodes[2]]
        )
        if self.cfg.model_type != VlmArchType.VLM_GEMMA4:
            rotary_emb = self._onnx_builder.build_op(
                f"{base_name}.q_proj.scaled_rotary_emb",
                [rotary_emb, self._head_dim**-0.5],
                "Mul"
            )
        return rotary_emb

    def _build_onnx_attn_key(self, base_name: str, input_nodes: list[OnnxNode]) -> OnnxNode:
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "k_proj")
        k_proj = self._onnx_builder.build_conv_from_dense_with_lora(
            f"{base_name}.k_proj", input_nodes[0], lora_rank,
            q_size=self._q_size,
            kv_size=self._kv_size
        )
        reshape1 = self._onnx_builder.build_split_and_concat(
            f"{base_name}.k_proj.reshape1", k_proj, self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
            split_axis=1, concat_axis=2
        )

        k_norm_name = None
        for suffix in ("k_layernorm", "k_norm"):
            if self.check_hf_param(f"{base_name}.{suffix}.weight"):
                k_norm_name = f"{base_name}.{suffix}"
                break

        if k_norm_name:
            reshape1 = self._build_rms_norm(k_norm_name, reshape1)

        rotary_emb = self._build_onnx_rotary_emb(
            f"{base_name}.k_proj", [reshape1, input_nodes[1], input_nodes[2]]
        )
        return rotary_emb

    def _build_onnx_attn_value(self, base_name: str, input_node: OnnxNode) -> OnnxNode:
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "v_proj")
        v_proj = self._onnx_builder.build_conv_from_dense_with_lora(
            f"{base_name}.v_proj", input_node, lora_rank,
            q_size=self._q_size,
            kv_size=self._kv_size
        )

        if self.cfg.model_type != VlmArchType.VLM_GEMMA4:
            return self._onnx_builder.build_split_and_concat(
                f"{base_name}.v_proj", v_proj, self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
                split_axis=1, concat_axis=2,
            )

        split = self._onnx_builder.build_split_and_concat(
            f"{base_name}.v_proj", v_proj,  self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
            split_axis=1, concat_axis=2
        )
        split = self._onnx_builder.build_rms_norm(
            f"{base_name}.v_norm", split, float(self.cfg.lm_cfg.rms_norm_eps),
            weightless=True, num_channels=self._head_dim,
        )
        return split

    def gen_model_sdk_files_directly(
        self,
        layer_cfg: LayerConfiguration,
        log_level: int, quantizable: bool
    ):
        base_name = self._layer_base_name
        merged_lora = layer_cfg.get("lora", LoraGenMode.LORA_DISABLED) == LoraGenMode.LORA_MERGED
        g = self._build_sima_nodes(base_name, quantizable, merged_lora)
        save_awesomenet(g, self.model_name + (".fp32" if quantizable else ""), str(self.sima_model_sdk_path))

    def _build_sima_nodes(self, base_name: str, quantizable: bool, merged_lora: bool = False):
        input_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        freq_shape = (
            1,
            1,
            self.num_tokens,
            self.cfg.lm_cfg.rope_cfg.get_rope_dimension_count(self.layer_type) // 2
        )
        builder = SimaBuilder(Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target)

        if self.cfg.pipeline_cfg.quantize_embeddings and self.layer_idx == 0:
            input_dtype = ScalarType.int8
        else:
            input_dtype = activation_type(quantizable)
        model_input_input = builder.create_placeholder_node(
            "input", TensorType(input_dtype, input_shape)
        )
        # EAGLE3 draft model has an extra hidden_states input
        if self.is_draft:
            model_input_hidden_states = builder.create_placeholder_node(
                "hidden_states", TensorType(activation_type(quantizable), input_shape)
            )
        model_input_freq_real = builder.create_placeholder_node(
            "freq_real", TensorType(activation_type(quantizable), freq_shape)
        )
        model_input_freq_imag = builder.create_placeholder_node(
            "freq_imag", TensorType(activation_type(quantizable), freq_shape)
        )

        # MLA subgraph inputs are the same as the model inputs, except the node names are different
        subnet_inputs = [model_input_input, model_input_freq_real, model_input_freq_imag]
        if self.is_draft:
            subnet_inputs.insert(1, model_input_hidden_states)
        builder.begin_subnet(subnet_inputs)
        mla_input_input = builder.create_placeholder_node(
            "MLA_0/input", TensorType(input_dtype, input_shape)
        )
        # EAGLE3 draft model has an extra hidden_states input
        if self.is_draft:
            mla_input_hidden_states = builder.create_placeholder_node(
                "MLA_0/hidden_states", TensorType(activation_type(quantizable), input_shape)
            )
        mla_input_freq_real = builder.create_placeholder_node(
            "MLA_0/freq_real", TensorType(activation_type(quantizable), freq_shape)
        )
        mla_input_freq_imag = builder.create_placeholder_node(
            "MLA_0/freq_imag", TensorType(activation_type(quantizable), freq_shape)
        )

        # De-quantize embeddings table if needed.
        if self.cfg.pipeline_cfg.quantize_embeddings and self.layer_idx == 0:
            assert self.embeddings_scale is not None
            rms_norm_in = builder.create_dequantization_node(
                "MLA_0/input",
                input_shape,
                1 / self.embeddings_scale,
                output_dtype=activation_dtype(quantizable)
            )
        else:
            rms_norm_in = mla_input_input

        norm_name = (
            f"{base_name}.operator_norm"
            if self.check_hf_param(f"{base_name}.operator_norm.weight")
            else f"{base_name}.input_layernorm"
        )
        rms_norm = self._build_sima_rms_norm(
            builder, norm_name, rms_norm_in
        )
        # EAGLE3 draft model additionally normalizes the hidden_states and concatenates.
        if self.is_draft:
            hidden_states_norm = self._build_sima_rms_norm(
                builder, f"{base_name}.hidden_norm", mla_input_hidden_states
            )
            attn_input = builder.create_concat_node([rms_norm, hidden_states_norm], 3)
        else:
            attn_input = rms_norm
        mla_q_out = self._build_sima_attn_query(
            builder, f"{base_name}.self_attn", attn_input, mla_input_freq_real, mla_input_freq_imag,
            quantizable, merged_lora
        )
        output_nodes = [mla_q_out]
        if not self.cfg.lm_cfg.is_kv_shared_layer(self.layer_idx):
            output_nodes.extend([
                self._build_sima_attn_key(
                    builder, f"{base_name}.self_attn", attn_input, mla_input_freq_real,
                    mla_input_freq_imag, merged_lora
                ),
                self._build_sima_attn_value(
                    builder, f"{base_name}.self_attn", attn_input, merged_lora
                )
            ])
        if len(output_nodes) > 1:
            _ = builder.create_tuple_node(output_nodes)

        mla_node = builder.finish_subnet("MLA_0")

        if activation_type(quantizable) != ScalarType.float32:
            if len(output_nodes) == 1:
                builder.create_cast_node(mla_node, ScalarType.float32)
            else:
                tuple_items = builder.create_tuple_get_item_nodes(mla_node)
                builder.create_tuple_node([builder.create_cast_node(x, ScalarType.float32) for x in tuple_items])

        net = builder.finish(self.model_name)
        return net

    def _build_sima_rotary_emb(
            self, builder: SimaBuilder,
            data: NodeOrHandle, freq_real: NodeOrHandle, freq_imag: NodeOrHandle
    ) -> AwesomeNode:
        """
        Create nodes that compute rotary embedding.
        """
        layer_head_dim = self._head_dim
        layer_rope_dimension_count = self.cfg.lm_cfg.rope_cfg.get_rope_dimension_count(self.layer_type)
        half_rope_dim = layer_rope_dimension_count // 2
        imag_start = layer_head_dim // 2 if self._is_proportional_rope_layer else half_rope_dim
        imag_end = imag_start + half_rope_dim

        real_in = create_channel_slice(builder, data, 0, half_rope_dim)
        imag_in = create_channel_slice(builder, data, imag_start, imag_end)
        real_out = builder.create_subtract_node(
            builder.create_mul_node(real_in, freq_real),
            builder.create_mul_node(imag_in, freq_imag)
        )
        imag_out = builder.create_add_node(
            builder.create_mul_node(real_in, freq_imag),
            builder.create_mul_node(imag_in, freq_real)
        )
        if self._is_proportional_rope_layer:
            mid1 = create_channel_slice(builder, data, half_rope_dim, layer_head_dim // 2)
            mid2 = create_channel_slice(builder, data, imag_end, layer_head_dim)
            return builder.create_concat_node([real_out, mid1, imag_out, mid2], 3)

        rotary_out = builder.create_concat_node([real_out, imag_out], 3)
        if layer_rope_dimension_count == layer_head_dim:
            return rotary_out

        tail = create_channel_slice(builder, data, layer_rope_dimension_count, layer_head_dim)
        return builder.create_concat_node([rotary_out, tail], 3)

    def _build_sima_attn_query(
            self, builder: SimaBuilder, base_name: str,
            rms_norm: NodeOrHandle, freq_real: NodeOrHandle, freq_imag: NodeOrHandle,
            quantizable: bool, merged_lora: bool = False
    ) -> AwesomeNode:
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "q_proj")
        q_proj = build_conv_from_dense_with_lora(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.q_proj", rms_norm,
            lora_rank, merged_lora=merged_lora,
            q_size=self._q_size,
            kv_size=self._kv_size
         )

        if self.cfg.lm_cfg.attn_cfg.num_attention_heads > 1:
            reshape1 = builder.create_slice_concat_node(
                q_proj, axis=1, split_axis=3,
                split_block=self.cfg.lm_cfg.attn_cfg.num_attention_heads, split_repeat=1
            )
        else:
            reshape1 = q_proj

        q_norm_name = None
        for suffix in ("q_layernorm", "q_norm"):
            if self.check_hf_param(f"{base_name}.{suffix}.weight"):
                q_norm_name = f"{base_name}.{suffix}"
                break

        if q_norm_name:
            reshape1 = self._build_sima_rms_norm(builder, q_norm_name, reshape1)

        rotary_emb = self._build_sima_rotary_emb(
            builder, reshape1, freq_real, freq_imag
        )

        if self.cfg.model_type == VlmArchType.VLM_GEMMA4:
            return rotary_emb

        return builder.create_mul_node(
            rotary_emb,
            builder.create_constant_node(
                np.array([self._head_dim**-0.5], dtype=activation_dtype(quantizable))
            )
        )

    def _build_sima_attn_key(
            self, builder: SimaBuilder, base_name: str,
            rms_norm: NodeOrHandle, freq_real: NodeOrHandle, freq_imag: NodeOrHandle,
            merged_lora: bool = False
    ) -> AwesomeNode:
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "k_proj")
        k_proj = build_conv_from_dense_with_lora(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.k_proj", rms_norm,
            lora_rank, merged_lora=merged_lora,
            q_size=self._q_size,
            kv_size=self._kv_size
        )

        reshape1 = builder.create_slice_concat_node(
            k_proj, axis=1, split_axis=3,
            split_block=self.cfg.lm_cfg.attn_cfg.num_key_value_heads, split_repeat=1
        )

        k_norm_name = None
        for suffix in ("k_layernorm", "k_norm"):
            if self.check_hf_param(f"{base_name}.{suffix}.weight"):
                k_norm_name = f"{base_name}.{suffix}"
                break

        if k_norm_name:
            reshape1 = self._build_sima_rms_norm(builder, k_norm_name, reshape1)

        rotary_emb = self._build_sima_rotary_emb(
            builder, reshape1, freq_real, freq_imag
        )
        return rotary_emb


    def _build_sima_attn_value(
            self, builder: SimaBuilder, base_name: str, input_node: NodeOrHandle, merged_lora: bool = False
    ) -> AwesomeNode:
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "v_proj")
        v_proj = build_conv_from_dense_with_lora(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.v_proj", input_node,
            lora_rank, merged_lora=merged_lora,
            q_size=self._q_size,
            kv_size=self._kv_size
        )
        if self.cfg.model_type != VlmArchType.VLM_GEMMA4:
            # Strided cache stores KV heads explicitly instead of flattened into kv_size.
            return builder.create_slice_concat_node(
                v_proj, axis=1, split_axis=3,
                split_block=self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
                split_repeat=1
            )

        # Gemma4 applies value RMS norm per KV head before writing V to cache.
        split = builder.create_slice_concat_node(
            v_proj,
            axis=1,
            split_axis=3,
            split_block=self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
            split_repeat=1,
        )
        split = self._build_sima_rms_norm(
            builder,
            f"{base_name}.v_norm",
            split,
            weightless=True,
            num_channels=self._head_dim,
        )
        return split

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters] :
        """
        Get the DRAM layouts to use for this model's inputs on the MLA.
        """
        return {}

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters] :
        """
        Get the DRAM layouts to use for this model's inputs on the MLA.
        """
        # Define DRAM shape to enable strided KV cache access for kv cache outputs.
        dram_shape = (
            1,
            self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
            self.cfg.pipeline_cfg.max_num_tokens,
            self._head_dim
        )

        k_cache_tessellate_params = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC16,
            persistent_mem_name="cached_keys",
            dram_shape=dram_shape
        )

        v_cache_tessellate_params = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC16,
            persistent_mem_name="cached_values",
            dram_shape=dram_shape
        )

        # 1st and 2nd output are kv cache outputs.
        tessellate_params =  {1: k_cache_tessellate_params, 2: v_cache_tessellate_params}

        return tessellate_params
