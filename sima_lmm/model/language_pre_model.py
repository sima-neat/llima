import numpy as np
from dataclasses import dataclass

from afe.apis.defines import TensorDRAMLayout
from afe.ir.node import AwesomeNode
from afe.ir.tensor_type import TensorType, ScalarType
from afe.ir.build_node import NodeOrHandle

from sima_lmm.model.base import TensorTessellateParameters, LoraGenMode, LayerConfiguration
from sima_lmm.model.model_graph import ModelGraph, save_model_graph, activation_dtype
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.config.vlm_config import VlmArchType


_bfloat16 = ScalarType.numpy_type(ScalarType.bfloat16)

@dataclass
class LanguagePreModel(LanguagePartBaseModel):
    """Base implementation for the pre cache model of the language model.

    Attributes:
        num_tokens: Number of tokens. Set to a value greater than 1 to consume multiple input tokens
            in one model.
        layer_idx: Transformer layer index.
    """
    num_tokens: int
    layer_idx: int

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

        if self.cfg.lm_cfg.attn_cfg.attn_output_gate:
            self._onnx_builder.create_output_node(
                self._onnx_builder.get_node_output_name(output_nodes[-1]),
                (1, self._q_size, 1, self.num_tokens)
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
        q_result = self._build_onnx_attn_query(f"{base_name}.self_attn", [attn_input, *input_nodes[freq_start:]])
        gate_out = None
        if self.cfg.lm_cfg.attn_cfg.attn_output_gate:
            q_out, gate_out = q_result
        else:
            q_out = q_result

        output_nodes = [q_out]
        if self.cfg.lm_cfg.is_kv_shared_layer(self.layer_idx):
            if gate_out is not None:
                output_nodes.append(gate_out)
            return output_nodes
        k_out = self._build_onnx_attn_key(f"{base_name}.self_attn", [attn_input, *input_nodes[freq_start:]])
        v_out = self._build_onnx_attn_value(f"{base_name}.self_attn", attn_input)
        output_nodes.extend([k_out, v_out])
        if gate_out is not None:
            output_nodes.append(gate_out)
        return output_nodes

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

    def _build_onnx_attn_query(self, base_name: str, input_nodes: list[OnnxNode]):
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "q_proj")
        q_proj = self._onnx_builder.build_conv_from_dense_with_lora(
            f"{base_name}.q_proj", input_nodes[0], lora_rank,
            q_size=self._q_size,
            kv_size=self._kv_size
        )
        gate_out = None
        if self.cfg.lm_cfg.attn_cfg.attn_output_gate:
            q_fused = self._onnx_builder.build_split_and_concat(
                f"{base_name}.q_proj.reshape_q_gate",
                q_proj,
                self.cfg.lm_cfg.attn_cfg.num_attention_heads,
                split_axis=1,
                concat_axis=2,
            )
            reshape = self._onnx_builder.build_op(
                f"{base_name}.q_proj.q",
                [
                    q_fused,
                    np.array([0], dtype=np.int64),
                    np.array([self.cfg.lm_cfg.attn_cfg.head_dim], dtype=np.int64),
                    np.array([1], dtype=np.int64),
                ],
                "Slice",
            )
            gate_out = self._onnx_builder.build_op(
                f"{base_name}.q_proj.gate",
                [
                    q_fused,
                    np.array([self.cfg.lm_cfg.attn_cfg.head_dim], dtype=np.int64),
                    np.array([2 * self.cfg.lm_cfg.attn_cfg.head_dim], dtype=np.int64),
                    np.array([1], dtype=np.int64),
                ],
                "Slice",
            )
            gate_out = self._onnx_builder.build_split_and_concat(
                f"{base_name}.q_proj.gate.reshape",
                gate_out,
                self.cfg.lm_cfg.attn_cfg.num_attention_heads,
                split_axis=2,
                concat_axis=1,
            )
        else:
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
        if gate_out is not None:
            return rotary_emb, gate_out
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
        save_model_graph(self, g, quantizable)

    def _build_sima_nodes(self, base_name: str, quantizable: bool, merged_lora: bool = False):
        input_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        scale_shape = (1, 1, self.num_tokens, 1)
        freq_shape = (
            1,
            1,
            self.num_tokens,
            self.cfg.lm_cfg.rope_cfg.get_rope_dimension_count(self.layer_type) // 2,
        )
        input_specs = {"input": input_shape}
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            input_specs["input"] = TensorType(ScalarType.int8, input_shape)
            input_specs["input_scale"] = scale_shape
        if self.is_draft:
            input_specs["hidden_states"] = input_shape
        input_specs.update(freq_real=freq_shape, freq_imag=freq_shape)
        graph = ModelGraph(self, input_specs, quantizable)
        inputs = graph.inputs
        mla_input_input = inputs["input"]
        mla_input_freq_real = inputs["freq_real"]
        mla_input_freq_imag = inputs["freq_imag"]

        # Dequantize the selected embedding rows before the first layer consumes them.
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            rms_norm_in = graph.dequant(mla_input_input, inputs["input_scale"])
        else:
            rms_norm_in = mla_input_input

        norm_name = (
            f"{base_name}.operator_norm"
            if self.check_hf_param(f"{base_name}.operator_norm.weight")
            else f"{base_name}.input_layernorm"
        )
        rms_norm = self._build_sima_rms_norm(graph, norm_name, rms_norm_in)
        # EAGLE3 draft model additionally normalizes the hidden_states and concatenates.
        if self.is_draft:
            hidden_states_norm = self._build_sima_rms_norm(
                graph, f"{base_name}.hidden_norm", inputs["hidden_states"]
            )
            attn_input = graph.concat([rms_norm, hidden_states_norm], 3)
        else:
            attn_input = rms_norm
        mla_q_result = self._build_sima_attn_query(
            graph,
            f"{base_name}.self_attn",
            attn_input,
            mla_input_freq_real,
            mla_input_freq_imag,
            quantizable,
            merged_lora,
        )
        gate_out = None
        if self.cfg.lm_cfg.attn_cfg.attn_output_gate:
            mla_q_out, gate_out = mla_q_result
        else:
            mla_q_out = mla_q_result
        output_nodes = [mla_q_out]
        if not self.cfg.lm_cfg.is_kv_shared_layer(self.layer_idx):
            mla_k_out = self._build_sima_attn_key(
                graph,
                f"{base_name}.self_attn",
                attn_input,
                mla_input_freq_real,
                mla_input_freq_imag,
                merged_lora,
            )
            mla_v_out = self._build_sima_attn_value(
                graph, f"{base_name}.self_attn", attn_input, merged_lora
            )

            if self.cfg.pipeline_cfg.quantize_kv_cache:
                k_quant, k_scale = graph.quant(mla_k_out)
                v_quant, v_scale = graph.quant(mla_v_out)
                output_nodes.extend([k_quant, k_scale, v_quant, v_scale])
            else:
                output_nodes.extend([mla_k_out, mla_v_out])

        if gate_out is not None:
            output_nodes.append(gate_out)
        return graph.finish(output_nodes)

    def _build_sima_rotary_emb(
        self, graph: ModelGraph, data: NodeOrHandle, freq_real: NodeOrHandle, freq_imag: NodeOrHandle
    ):
        """
        Create nodes that compute rotary embedding.
        """
        return graph.rope(
            data,
            freq_real,
            freq_imag,
            self.cfg.lm_cfg.rope_cfg.get_rope_dimension_count(self.layer_type),
            proportional=self._is_proportional_rope_layer,
        )

    def _build_sima_attn_query(
        self,
        graph: ModelGraph,
        base_name: str,
        rms_norm: NodeOrHandle,
        freq_real: NodeOrHandle,
        freq_imag: NodeOrHandle,
        quantizable: bool,
        merged_lora: bool = False,
    ) -> AwesomeNode:
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "q_proj")
        q_proj = graph.linear(
            f"{base_name}.q_proj",
            rms_norm,
            merged_lora=merged_lora,
            q_size=self._q_size,
            kv_size=self._kv_size,
            lora_rank=lora_rank,
        )

        gate_out = None
        if self.cfg.lm_cfg.attn_cfg.attn_output_gate:
            q_fused = graph.split_heads(q_proj, self.cfg.lm_cfg.attn_cfg.num_attention_heads)
            head_dim = self.cfg.lm_cfg.attn_cfg.head_dim
            gate_out = graph.slice(q_fused, [head_dim], [2 * head_dim], [1], [3])
            q_proj = graph.slice(q_fused, [0], [head_dim], [1], [3])
            gate_out = graph.merge_heads(gate_out)
            reshape1 = q_proj
        elif self.cfg.lm_cfg.attn_cfg.num_attention_heads > 1:
            reshape1 = graph.split_heads(q_proj, self.cfg.lm_cfg.attn_cfg.num_attention_heads)
        else:
            reshape1 = q_proj

        q_norm_name = None
        for suffix in ("q_layernorm", "q_norm"):
            if self.check_hf_param(f"{base_name}.{suffix}.weight"):
                q_norm_name = f"{base_name}.{suffix}"
                break

        if q_norm_name:
            reshape1 = self._build_sima_rms_norm(graph, q_norm_name, reshape1)

        rotary_emb = self._build_sima_rotary_emb(graph, reshape1, freq_real, freq_imag)

        if self.cfg.model_type != VlmArchType.VLM_GEMMA4:
            rotary_emb = graph.mul(
                rotary_emb,
                graph.constant(np.array([self._head_dim**-0.5], dtype=activation_dtype(quantizable))),
            )
        if gate_out is not None:
            return rotary_emb, gate_out
        return rotary_emb

    def _build_sima_attn_key(
        self,
        graph: ModelGraph,
        base_name: str,
        rms_norm: NodeOrHandle,
        freq_real: NodeOrHandle,
        freq_imag: NodeOrHandle,
        merged_lora: bool = False,
    ) -> AwesomeNode:
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "k_proj")
        k_proj = graph.linear(
            f"{base_name}.k_proj",
            rms_norm,
            merged_lora=merged_lora,
            q_size=self._q_size,
            kv_size=self._kv_size,
            lora_rank=lora_rank,
        )

        reshape1 = graph.split_heads(k_proj, self.cfg.lm_cfg.attn_cfg.num_key_value_heads)

        k_norm_name = None
        for suffix in ("k_layernorm", "k_norm"):
            if self.check_hf_param(f"{base_name}.{suffix}.weight"):
                k_norm_name = f"{base_name}.{suffix}"
                break

        if k_norm_name:
            reshape1 = self._build_sima_rms_norm(graph, k_norm_name, reshape1)

        rotary_emb = self._build_sima_rotary_emb(graph, reshape1, freq_real, freq_imag)
        return rotary_emb


    def _build_sima_attn_value(
        self, graph: ModelGraph, base_name: str, input_node: NodeOrHandle, merged_lora: bool = False
    ) -> AwesomeNode:
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "v_proj")
        v_proj = graph.linear(
            f"{base_name}.v_proj",
            input_node,
            merged_lora=merged_lora,
            q_size=self._q_size,
            kv_size=self._kv_size,
            lora_rank=lora_rank,
        )
        if self.cfg.model_type != VlmArchType.VLM_GEMMA4:
            # Strided cache stores KV heads explicitly instead of flattened into kv_size.
            return graph.split_heads(v_proj, self.cfg.lm_cfg.attn_cfg.num_key_value_heads)

        # Gemma4 applies value RMS norm per KV head before writing V to cache.
        split = graph.split_heads(v_proj, self.cfg.lm_cfg.attn_cfg.num_key_value_heads)
        split = self._build_sima_rms_norm(
            graph,
            f"{base_name}.v_norm",
            split,
            weightless=True,
            num_channels=self._head_dim,
        )
        return split


    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters] :
        """
        Get the DRAM layouts to use for this model's outputs on the MLA.
        """
        # Define DRAM shape to enable strided KV cache access for kv cache outputs.
        dram_shape = (
            1,
            self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
            self.cfg.pipeline_cfg.max_num_tokens,
            self._head_dim
        )

        k_cache_params = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC16,
            dram_shape=dram_shape
        )

        v_cache_params = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC16,
            dram_shape=dram_shape
        )

        if self.cfg.pipeline_cfg.quantize_kv_cache:
            scale_dram_shape = (
                1,
                self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
                self.cfg.pipeline_cfg.max_num_tokens,
                1
            )
            k_scale_params = TensorTessellateParameters(
                tile_shape=(0, 0, 0, 0),
                enable_mla=True,
                dram_layout=TensorDRAMLayout.HWC16,
                dram_shape=scale_dram_shape
            )
            v_scale_params = TensorTessellateParameters(
                tile_shape=(0, 0, 0, 0),
                enable_mla=True,
                dram_layout=TensorDRAMLayout.HWC16,
                dram_shape=scale_dram_shape
            )
            # Output order: [q, k_quant, k_scale, v_quant, v_scale]
            tessellate_params = {
                1: k_cache_params,
                2: k_scale_params,   # k_scale
                3: v_cache_params,
                4: v_scale_params,   # v_scale
            }
        else:
            # Output order: [q, k, v]
            tessellate_params = {1: k_cache_params, 2: v_cache_params}

        return tessellate_params
