import numpy as np
from dataclasses import dataclass

from afe.apis.defines import gen2_target, TensorDRAMLayout
from afe.ir.serializer import save_awesomenet
from afe.ir.defines import Status, TensorValue, TupleValue, get_expected_tensor_value
from afe.ir.tensor_type import TensorType, ScalarType
from afe.ir.build_node import NodeHandle, NodeOrHandle

from sima_lmm.model.base import TensorTessellateParameters, LoraGenMode, LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePostBaseModel
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import (
    SimaBuilder, build_conv_from_dense_with_lora,
    build_activation, activation_type, activation_dtype
)
from sima_lmm.config.vlm_config import LlmArchType, VlmArchType
from sima_lmm.utils import ceil_div


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
    def split_mlp(self) -> bool:
        return self.cfg.pipeline_cfg.split_mlp

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

        o_proj = self._onnx_builder.build_conv_from_dense_with_lora(
            f"{base_name}.self_attn.{attn_out_name}", input_nodes[1], lora_rank
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
                base_name, final_output, input_nodes[2]
            )
        llm_injection_layers = range(len(getattr(self.cfg.vm_cfg, "deepstack_visual_indexes", [])))
        if self.layer_idx in llm_injection_layers and self.num_tokens > 1:
            deepstack_features_input = input_nodes[2]
            final_output = self._onnx_builder.build_op(
                f"{base_name}.deepstack_add",
                [add2, deepstack_features_input],
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
        residual = hidden_states
        gate = build_conv_from_dense_with_lora(
            builder,
            self.get_hf_param,
            self.check_hf_param,
            f"{base_name}.per_layer_input_gate",
            hidden_states,
            None,
            merged_lora=merged_lora,
        )
        act = build_activation(builder, gate, self.cfg.lm_cfg.mlp_cfg.act, quantizable)
        mul = builder.create_mul_node(act, per_layer_input)
        proj = build_conv_from_dense_with_lora(
            builder,
            self.get_hf_param,
            self.check_hf_param,
            f"{base_name}.per_layer_projection",
            mul,
            None,
            merged_lora=merged_lora,
        )
        norm = self._build_sima_rms_norm(builder, f"{base_name}.post_per_layer_input_norm", proj)
        add = builder.create_add_node(residual, norm)
        layer_scalar = builder.create_constant_node(
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
        save_awesomenet(g, self.model_name + (".fp32" if quantizable else ""), str(self.sima_model_sdk_path))

    def has_ffn_layernorms(self, base_name):
        pre_ln = f"{base_name}.pre_feedforward_layernorm.weight"
        post_ln = f"{base_name}.post_feedforward_layernorm.weight"
        return self.check_hf_param(pre_ln) and self.check_hf_param(post_ln)

    def _build_sima_nodes(self, base_name: str, quantizable: bool, merged_lora: bool = False):
        input_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        scale_shape = (1, 1, self.num_tokens, 1)
        self_attn_shape = (
            1, 1, self.num_tokens, self.cfg.lm_cfg.attn_cfg.get_q_size(self.layer_type),
        )
        per_layer_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size_per_layer_input)

        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            input_dtype = ScalarType.int8
        else:
            input_dtype = activation_type(quantizable)

        builder = SimaBuilder(Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target)

        # Check if this layer needs deepstack injection
        llm_injection_layers = range(len(getattr(self.cfg.vm_cfg, "deepstack_visual_indexes", []))) if self.cfg.vm_cfg else []
        needs_deepstack = self.layer_idx in llm_injection_layers and self.num_tokens > 1

        model_input_input = builder.create_placeholder_node(
            "input", TensorType(input_dtype, input_shape)
        )
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            model_input_scale = builder.create_placeholder_node(
                "input_scale", TensorType(activation_type(quantizable), scale_shape)
            )
        model_input_self_attn = builder.create_placeholder_node(
            "self_attn", TensorType(activation_type(quantizable), self_attn_shape)
        )
        subnet_inputs = [model_input_input]
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            subnet_inputs.append(model_input_scale)
        subnet_inputs.append(model_input_self_attn)
        if self.cfg.model_type == VlmArchType.VLM_GEMMA4:
            model_input_per_layer = builder.create_placeholder_node(
                "per_layer_input", TensorType(activation_type(quantizable), per_layer_shape)
            )
            subnet_inputs.append(model_input_per_layer)
        if needs_deepstack:
            model_input_deepstack = builder.create_placeholder_node(
                "deepstack_features", TensorType(activation_type(quantizable), input_shape)
            )
            subnet_inputs.append(model_input_deepstack)
        # MLA subgraph inputs are the same as the model inputs, except the node names are different
        builder.begin_subnet(subnet_inputs)
        mla_input_input = builder.create_placeholder_node(
            "MLA_0/input", TensorType(input_dtype, input_shape)
        )
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            mla_input_scale = builder.create_placeholder_node(
                "MLA_0/input_scale", TensorType(activation_type(quantizable), scale_shape)
            )
        mla_input_self_attn = builder.create_placeholder_node(
            "MLA_0/self_attn", TensorType(activation_type(quantizable), self_attn_shape)
        )
        if self.cfg.model_type == VlmArchType.VLM_GEMMA4:
            mla_input_per_layer = builder.create_placeholder_node(
                "MLA_0/per_layer_input",
                TensorType(activation_type(quantizable), per_layer_shape),
            )
        mla_input_deepstack = None
        if needs_deepstack:
            mla_input_deepstack = builder.create_placeholder_node(
                "MLA_0/deepstack_features", TensorType(activation_type(quantizable), input_shape)
            )
        attn_out_name = ("out_proj" if self.check_hf_param(f"{base_name}.self_attn.out_proj.weight") else "o_proj")
        attn_out_full_name = f"{base_name}.self_attn.{attn_out_name}"

        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, attn_out_name)
        o_proj = build_conv_from_dense_with_lora(
            builder, self.get_hf_param, self.check_hf_param, attn_out_full_name,
            mla_input_self_attn, lora_rank, merged_lora=merged_lora
        )

        # Dequantize the selected embedding rows before the residual path consumes them.
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            rms_norm_in = builder.create_dynamic_dequant_node(
                mla_input_input, mla_input_scale
            )
        else:
            rms_norm_in = mla_input_input

        has_ffn_norms = self.has_ffn_layernorms(base_name)
        if has_ffn_norms:
            rms_norm1 = self._build_sima_rms_norm(builder, f"{base_name}.post_attention_layernorm", o_proj)
            add1 = builder.create_add_node(rms_norm_in, rms_norm1)
            rms_norm2 = self._build_sima_rms_norm(builder, f"{base_name}.pre_feedforward_layernorm", add1)
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
                self.check_hf_param(f"{base_name}.feed_forward.{w}.weight")
                for w in ("w1", "w2", "w3")
            )
            else f"{base_name}.mlp"
        )

        if has_ffn_norms:
            mlp = self._build_sima_mlp(builder, mlp_base, [rms_norm2], quantizable, merged_lora)
            mlp = self._build_sima_rms_norm(builder, f"{base_name}.post_feedforward_layernorm", mlp)
            add2 = builder.create_add_node(add1, mlp)
        else:
            add2 = self._build_sima_mlp(
                builder, mlp_base, [rms_norm2, add1], quantizable, merged_lora,
                with_residual_add=True
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
            if len(outputs) > 1:
                _ = builder.create_tuple_node(outputs)

        mla_node = builder.finish_subnet("MLA_0")

        self._cast_bf16_outputs_to_fp32(builder, mla_node)

        net = builder.finish(self.model_name)
        return net

    def get_mla_input_tessellate_params(self) ->  dict[int, TensorTessellateParameters]:
        """
        Get the custom tessellate params for model's inputs on the MLA.
        """
        # Use default tessellate params.
        return {}


    def get_mla_output_tessellate_params(self) ->  dict[int, TensorTessellateParameters]:
        """
        Get the custom tessellate params for model's output on the MLA.
        """
        # Use default tessellate params.
        return {}
