from dataclasses import dataclass

import numpy as np

from sima_lmm.model.base import LoraGenMode, LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePostBaseModel
from sima_lmm.model.model_graph import ModelGraph, Node
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

    def _build_per_layer_input_branch(
        self,
        graph: ModelGraph,
        base_name: str,
        hidden_states: Node,
        per_layer_input: Node,
        merged_lora: bool = False,
    ) -> Node:
        residual = hidden_states
        gate = graph.linear(
            f"{base_name}.per_layer_input_gate", hidden_states, merged_lora=merged_lora, lora_rank=None
        )
        act = graph.activation(gate, self.cfg.lm_cfg.mlp_cfg.act)
        mul = graph.mul(act, per_layer_input)
        proj = graph.linear(
            f"{base_name}.per_layer_projection", mul, merged_lora=merged_lora, lora_rank=None
        )
        norm = graph.rms_norm(f"{base_name}.post_per_layer_input_norm", proj)
        add = graph.add(residual, norm)
        layer_scalar = graph.constant(
            self.get_hf_param(f"{base_name}.layer_scalar")
            .reshape(1)
        )
        return graph.mul(add, layer_scalar)

    def generate_graph(
        self,
        layer_cfg: LayerConfiguration,
        quantizable: bool,
    ):
        base_name = self._layer_base_name
        merged_lora = layer_cfg.get("lora", LoraGenMode.LORA_DISABLED) == LoraGenMode.LORA_MERGED
        input_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        if self.cfg.lm_cfg.moe_cfg is not None and self.expert_idx >= 0:
            graph = ModelGraph(self, {
                "norm_hidden": input_shape,
                "router": (1, 1, self.num_tokens, self.cfg.lm_cfg.moe_cfg.num_experts),
            }, quantizable)
            mlp_base = f"{base_name}.mlp.experts.{self.expert_idx}"
            if not self.check_hf_param(f"{mlp_base}.gate_proj.weight"):
                mlp_base = f"{base_name}.mlp"
            expert = self._build_mlp(graph, mlp_base, [graph.inputs["norm_hidden"]], merged_lora)
            weight = graph.slice(
                graph.inputs["router"], start=self.expert_idx, stop=self.expert_idx + 1, axis=-1
            )
            graph.save([graph.mul(expert, weight)])
            return
        scale_shape = (1, 1, self.num_tokens, 1)
        self_attn_shape = (
            1,
            1,
            self.num_tokens,
            self.cfg.lm_cfg.attn_cfg.get_q_size(self.layer_type),
        )
        per_layer_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size_per_layer_input)

        input_specs = {"input": input_shape}
        input_dtypes = {}
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            input_dtypes["input"] = np.int8
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
        graph = ModelGraph(self, input_specs, quantizable, input_dtypes=input_dtypes)
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
            sig = graph.sigmoid(mla_input_gate)
            attn_in = graph.mul(mla_input_self_attn, sig)
        o_proj = graph.linear(attn_out_full_name, attn_in, merged_lora=merged_lora, lora_rank=lora_rank)

        # Dequantize the selected embedding rows before the residual path consumes them.
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            rms_norm_in = graph.dequant(mla_input_input, mla_input_scale)
        else:
            rms_norm_in = mla_input_input

        has_ffn_norms = self.has_ffn_layernorms(base_name)
        if has_ffn_norms:
            rms_norm1 = graph.rms_norm(f"{base_name}.post_attention_layernorm", o_proj)
            add1 = graph.add(rms_norm_in, rms_norm1)
            rms_norm2 = graph.rms_norm(f"{base_name}.pre_feedforward_layernorm", add1)
        else:
            if self.check_hf_param(f"{base_name}.ffn_norm.weight"):
                rms_norm_name = "ffn_norm"
            elif self.hf_model.is_gguf:
                rms_norm_name = "pre_feedforward_layernorm"
            else:
                rms_norm_name = "post_attention_layernorm"

            add1 = graph.add(rms_norm_in, o_proj)
            rms_norm2 = graph.rms_norm(f"{base_name}.{rms_norm_name}", add1)

        # LFM2 uses feed_forward.{w1,w3,w2}; fall back to mlp.{gate,up,down}.
        mlp_base = (
            f"{base_name}.feed_forward"
            if all(
                self.check_hf_param(f"{base_name}.feed_forward.{w}.weight") for w in ("w1", "w2", "w3")
            )
            else f"{base_name}.mlp"
        )

        if has_ffn_norms:
            mlp = self._build_mlp(graph, mlp_base, [rms_norm2], merged_lora)
            mlp = graph.rms_norm(f"{base_name}.post_feedforward_layernorm", mlp)
            add2 = graph.add(add1, mlp)
        else:
            add2 = self._build_mlp(
                graph, mlp_base, [rms_norm2, add1], merged_lora, with_residual_add=True
            )

        # Add deepstack features if needed
        final_output = add2
        if self.cfg.model_type == VlmArchType.VLM_GEMMA4:
            final_output = self._build_per_layer_input_branch(
                graph,
                base_name,
                final_output,
                mla_input_per_layer,
                merged_lora,
            )
        if needs_deepstack and mla_input_deepstack is not None:
            final_output = graph.add(final_output, mla_input_deepstack)

        if self.layer_idx == self.cfg.lm_cfg.num_hidden_layers - 1:
            outputs = self._build_post_transformer(graph, final_output)
        else:
            outputs = [final_output]

        graph.save(outputs)

    def has_ffn_layernorms(self, base_name):
        pre_ln = f"{base_name}.pre_feedforward_layernorm.weight"
        post_ln = f"{base_name}.post_feedforward_layernorm.weight"
        return self.check_hf_param(pre_ln) and self.check_hf_param(post_ln)
