import numpy as np
from dataclasses import dataclass

from afe.ir.tensor_type import TensorType, ScalarType
from afe.ir.build_node import NodeOrHandle

from sima_lmm.model.base import LoraGenMode, LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePostBaseModel
from sima_lmm.model.model_graph import ModelGraph, save_model_graph
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import SimaBuilder, activation_type, activation_dtype
from sima_lmm.config.vlm_config import VlmArchType


@dataclass
class LanguagePostModel(LanguagePostBaseModel):
    """Implementation for the post cache model of transformer-based language models."""

    def __post_init__(self):
        assert self.num_tokens >= 1
        assert 0 <= self.layer_idx < self.cfg.lm_cfg.num_hidden_layers

    @property
    def layer_type(self) -> str:
        return self.cfg.lm_cfg.layer_types[self.layer_idx]

    @property
    def enable_filter_sharing(self) -> bool:
        return self.cfg.pipeline_cfg.enable_filter_sharing

    @property
    def uses_quantized_input_embeddings(self) -> bool:
        # EAGLE3 draft post consumes the BF16 FC-fused hidden state, not an embedding row.
        return super().uses_quantized_input_embeddings and not self.is_draft

    @property
    def _layer_base_name(self) -> str:
        base = self.hf_model.language_model_param_base_name
        return base if self.is_draft else f"{base}.layers.{self.layer_idx}"

    def gen_onnx_files(self):
        base_name = self._layer_base_name
        self.create_onnx_builder()
        self._onnx_builder.create_input_node(
            "input", (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens)
        )
        self._onnx_builder.create_input_node(
            "self_attn", (1, self.cfg.lm_cfg.attn_cfg.get_q_size(self.layer_type), 1, self.num_tokens)
        )
        if self.cfg.model_type == VlmArchType.VLM_GEMMA4:
            self._onnx_builder.create_input_node(
                "per_layer_input",
                (1, self.cfg.lm_cfg.hidden_size_per_layer_input, 1, self.num_tokens),
            )

        if self.cfg.lm_cfg.attn_cfg.attn_output_gate:
            self._onnx_builder.create_input_node(
                "gate", (1, self.cfg.lm_cfg.attn_cfg.get_q_size(self.layer_type), 1, self.num_tokens)
            )

        llm_injection_layers = range(len(getattr(self.cfg.vm_cfg, "deepstack_visual_indexes", [])))
        if self.cfg.vm_cfg and self.layer_idx in llm_injection_layers and self.num_tokens > 1:
            self._onnx_builder.create_input_node(
                "deepstack_features", (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens)
            )
        output_nodes = self._build_onnx_nodes(base_name, self._onnx_builder.input_nodes)
        output_name = self._onnx_builder.get_node_output_name(output_nodes[0])
        if self.layer_idx < self.cfg.lm_cfg.num_hidden_layers - 1:
            self._onnx_builder.create_output_node(
                output_name, (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens)
            )
        else:
            self._create_final_layer_output_nodes(output_nodes)

        self._onnx_builder.create_and_save_model()

        # Set to None to deallocate the memory.
        self._onnx_builder = None


    def _build_onnx_nodes(self, base_name: str, input_nodes: list[OnnxNode]) -> list[OnnxNode]:
        # LFM2 uses self_attn.out_proj instead of o_proj.
        attn_out_name = ("out_proj" if self.check_hf_param(f"{base_name}.self_attn.out_proj.weight") else "o_proj")
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, attn_out_name)

        input_idx = 2
        per_layer_input = None
        if self.cfg.model_type == VlmArchType.VLM_GEMMA4:
            per_layer_input = input_nodes[input_idx]
            input_idx += 1

        attn_in = input_nodes[1]
        if self.cfg.lm_cfg.attn_cfg.attn_output_gate:
            gate = input_nodes[input_idx]
            sig = self._onnx_builder.build_op(
                f"{base_name}.self_attn.gate_sigmoid", [gate], "Sigmoid"
            )
            attn_in = self._onnx_builder.build_op(
                f"{base_name}.self_attn.gate_mul", [input_nodes[1], sig], "Mul"
            )

        o_proj = self._onnx_builder.build_conv_from_dense_with_lora(
            f"{base_name}.self_attn.{attn_out_name}", attn_in, lora_rank
        )

        has_ffn_norms = self.has_ffn_layernorms(base_name)
        if has_ffn_norms:
            rms_norm1 = self._build_rms_norm(f"{base_name}.post_attention_layernorm", o_proj)
            add1 = self._onnx_builder.build_op(
                f"{base_name}.add1", [input_nodes[0], rms_norm1], "Add"
            )
            rms_norm2 = self._build_rms_norm(f"{base_name}.pre_feedforward_layernorm", add1)
        else:
            norm_name = "ffn_norm" if self.check_hf_param(f"{base_name}.ffn_norm.weight") else "post_attention_layernorm"
            add1 = self._onnx_builder.build_op(
                f"{base_name}.add1", [input_nodes[0], o_proj], "Add"
            )
            rms_norm2 = self._build_rms_norm(f"{base_name}.{norm_name}", add1)

        # LFM2 uses feed_forward.{w1,w3,w2}; fall back to mlp.{gate,up,down}.
        mlp_base = (
            f"{base_name}.feed_forward"
            if all(
                self.check_hf_param(f"{base_name}.feed_forward.{w}.weight")
                for w in ("w1", "w2", "w3")
            )
            else f"{base_name}.mlp"
        )

        if has_ffn_norms:
            mlp = self._build_onnx_mlp(mlp_base, [rms_norm2])
            rms_norm = self._build_rms_norm(f"{base_name}.post_feedforward_layernorm", mlp)
            add2 = self._onnx_builder.build_op(f"{base_name}.add2", [add1, rms_norm], "Add")
        else:
            add2 = self._build_onnx_mlp(
                mlp_base, [rms_norm2, add1], with_residual_add=True
            )

        final_output = add2
        if self.cfg.model_type == VlmArchType.VLM_GEMMA4:
            final_output = self._build_onnx_per_layer_input_branch(
                base_name, final_output, per_layer_input
            )
        llm_injection_layers = range(len(getattr(self.cfg.vm_cfg, "deepstack_visual_indexes", [])))
        if self.layer_idx in llm_injection_layers and self.num_tokens > 1:
            deepstack_features_input = input_nodes[-1]
            final_output = self._onnx_builder.build_op(
                f"{base_name}.deepstack_add",
                [final_output, deepstack_features_input],
                "Add"
            )
        if self.layer_idx < self.cfg.lm_cfg.num_hidden_layers - 1:
            return [final_output]

        # Include the operations after the last transformer layer into last post cache model.
        return self._build_onnx_post_transformer(base_name, final_output)

    def _build_onnx_per_layer_input_branch(
        self, base_name: str, hidden_states: OnnxNode, per_layer_input: OnnxNode
    ) -> OnnxNode:
        residual = hidden_states
        gate = self._onnx_builder.build_conv_from_dense_with_lora(
            f"{base_name}.per_layer_input_gate", hidden_states, None
        )
        act = self._onnx_builder.build_activation(
            f"{base_name}.per_layer_input_act", gate, self.cfg.lm_cfg.mlp_cfg.act
        )
        mul = self._onnx_builder.build_op(
            f"{base_name}.per_layer_input_mul", [act, per_layer_input], "Mul"
        )
        proj = self._onnx_builder.build_conv_from_dense_with_lora(
            f"{base_name}.per_layer_projection", mul, None
        )
        norm = self._build_rms_norm(f"{base_name}.post_per_layer_input_norm", proj)
        add = self._onnx_builder.build_op(
            f"{base_name}.per_layer_input_add", [residual, norm], "Add"
        )
        layer_scalar = self._onnx_builder.create_initializer(
            f"{base_name}.layer_scalar",
            self.get_hf_param(f"{base_name}.layer_scalar").astype(np.float32).reshape(1, 1, 1, 1),
        )
        return self._onnx_builder.build_op(
            f"{base_name}.layer_scalar_mul", [add, layer_scalar], "Mul"
        )

    def _build_sima_per_layer_input_branch(
        self,
        builder: SimaBuilder,
        base_name: str,
        hidden_states: NodeOrHandle,
        per_layer_input: NodeOrHandle,
        quantizable: bool,
        merged_lora: bool = False,
    ) -> NodeOrHandle:
        graph = ModelGraph.from_builder(self, builder)
        residual = hidden_states
        gate = graph.linear(
            f"{base_name}.per_layer_input_gate", hidden_states, merged_lora=merged_lora, lora_rank=None
        )
        act = graph.activation(gate, self.cfg.lm_cfg.mlp_cfg.act)
        mul = builder.create_mul_node(act, per_layer_input)
        proj = graph.linear(
            f"{base_name}.per_layer_projection", mul, merged_lora=merged_lora, lora_rank=None
        )
        norm = self._build_sima_rms_norm(builder, f"{base_name}.post_per_layer_input_norm", proj)
        add = builder.create_add_node(residual, norm)
        layer_scalar = graph.constant(
            self.get_hf_param(f"{base_name}.layer_scalar")
            .astype(activation_dtype(quantizable))
            .reshape(1)
        )
        return builder.create_mul_node(add, layer_scalar)

    def gen_model_sdk_files_directly(
        self,
        layer_cfg: LayerConfiguration,
        log_level: int,
        quantizable: bool,
    ):
        base_name = self._layer_base_name
        merged_lora = layer_cfg.get("lora", LoraGenMode.LORA_DISABLED) == LoraGenMode.LORA_MERGED
        g = self._build_sima_nodes(base_name, quantizable, merged_lora)
        save_model_graph(self, g, quantizable)

    def has_ffn_layernorms(self, base_name):
        pre_ln = f"{base_name}.pre_feedforward_layernorm.weight"
        post_ln = f"{base_name}.post_feedforward_layernorm.weight"
        return self.check_hf_param(pre_ln) and self.check_hf_param(post_ln)

    def _build_sima_nodes(self, base_name: str, quantizable: bool, merged_lora: bool = False):
        input_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        scale_shape = (1, 1, self.num_tokens, 1)
        self_attn_shape = (
            1,
            1,
            self.num_tokens,
            self.cfg.lm_cfg.attn_cfg.get_q_size(self.layer_type),
        )
        per_layer_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size_per_layer_input)

        input_specs = {"input": input_shape}
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            input_specs["input"] = TensorType(ScalarType.int8, input_shape)
            input_specs["input_scale"] = scale_shape
        # Check if this layer needs deepstack injection.
        llm_injection_layers = (
            range(len(getattr(self.cfg.vm_cfg, "deepstack_visual_indexes", [])))
            if self.cfg.vm_cfg
            else []
        )
        needs_deepstack = self.layer_idx in llm_injection_layers and self.num_tokens > 1
        input_specs["self_attn"] = self_attn_shape
        if self.cfg.model_type == VlmArchType.VLM_GEMMA4:
            input_specs["per_layer_input"] = per_layer_shape
        if self.cfg.lm_cfg.attn_cfg.attn_output_gate:
            input_specs["gate"] = self_attn_shape
        if needs_deepstack:
            input_specs["deepstack_features"] = input_shape
        graph = ModelGraph(self, input_specs, quantizable)
        builder = graph.raw
        mla_input_input = graph.inputs["input"]
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            mla_input_scale = graph.inputs["input_scale"]
        mla_input_self_attn = graph.inputs["self_attn"]
        mla_input_per_layer = None
        if self.cfg.model_type == VlmArchType.VLM_GEMMA4:
            mla_input_per_layer = graph.inputs["per_layer_input"]
        mla_input_gate = None
        if self.cfg.lm_cfg.attn_cfg.attn_output_gate:
            mla_input_gate = graph.inputs["gate"]
        mla_input_deepstack = None
        if needs_deepstack:
            mla_input_deepstack = graph.inputs["deepstack_features"]
        attn_out_name = (
            "out_proj" if self.check_hf_param(f"{base_name}.self_attn.out_proj.weight") else "o_proj"
        )
        attn_out_full_name = f"{base_name}.self_attn.{attn_out_name}"

        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, attn_out_name)

        attn_in = mla_input_self_attn
        if mla_input_gate is not None:
            sig = builder.create_sigmoid_node(mla_input_gate)
            attn_in = builder.create_mul_node(mla_input_self_attn, sig)
        o_proj = graph.linear(attn_out_full_name, attn_in, merged_lora=merged_lora, lora_rank=lora_rank)

        # Dequantize the selected embedding rows before the residual path consumes them.
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            rms_norm_in = graph.dequantize(mla_input_input, mla_input_scale)
        else:
            rms_norm_in = mla_input_input

        has_ffn_norms = self.has_ffn_layernorms(base_name)
        if has_ffn_norms:
            rms_norm1 = self._build_sima_rms_norm(
                builder, f"{base_name}.post_attention_layernorm", o_proj
            )
            add1 = builder.create_add_node(rms_norm_in, rms_norm1)
            rms_norm2 = self._build_sima_rms_norm(
                builder, f"{base_name}.pre_feedforward_layernorm", add1
            )
        else:
            if self.check_hf_param(f"{base_name}.ffn_norm.weight"):
                rms_norm_name = "ffn_norm"
            elif self.hf_model.is_gguf:
                rms_norm_name = "pre_feedforward_layernorm"
            else:
                rms_norm_name = "post_attention_layernorm"

            add1 = builder.create_add_node(rms_norm_in, o_proj)
            rms_norm2 = self._build_sima_rms_norm(builder, f"{base_name}.{rms_norm_name}", add1)

        # LFM2 uses feed_forward.{w1,w3,w2}; fall back to mlp.{gate,up,down}.
        mlp_base = (
            f"{base_name}.feed_forward"
            if all(
                self.check_hf_param(f"{base_name}.feed_forward.{w}.weight") for w in ("w1", "w2", "w3")
            )
            else f"{base_name}.mlp"
        )

        if has_ffn_norms:
            mlp = self._build_sima_mlp(builder, mlp_base, [rms_norm2], quantizable, merged_lora)
            mlp = self._build_sima_rms_norm(builder, f"{base_name}.post_feedforward_layernorm", mlp)
            add2 = builder.create_add_node(add1, mlp)
        else:
            add2 = self._build_sima_mlp(
                builder, mlp_base, [rms_norm2, add1], quantizable, merged_lora, with_residual_add=True
            )

        # Add deepstack features if needed
        final_output = add2
        if self.cfg.model_type == VlmArchType.VLM_GEMMA4:
            final_output = self._build_sima_per_layer_input_branch(
                builder,
                base_name,
                final_output,
                mla_input_per_layer,
                quantizable,
                merged_lora,
            )
        if needs_deepstack and mla_input_deepstack is not None:
            final_output = builder.create_add_node(final_output, mla_input_deepstack)

        if self.layer_idx == self.cfg.lm_cfg.num_hidden_layers - 1:
            outputs = self._build_post_transformer(builder, final_output, quantizable)
        else:
            outputs = [final_output]

        return graph.finish(outputs)
