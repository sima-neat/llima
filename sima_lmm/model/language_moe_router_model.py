from dataclasses import dataclass

import numpy as np

from sima_lmm.config.vlm_config import LlmArchType
from sima_lmm.model.base import LayerConfiguration, LoraGenMode
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.model.model_graph import ModelGraph


@dataclass
class LanguageMoeRouterModel(LanguagePartBaseModel):
    """Project attention, add the residual, normalize, and select expert weights."""

    num_tokens: int
    layer_idx: int

    def __post_init__(self):
        assert self.num_tokens >= 1
        assert 0 <= self.layer_idx < self.cfg.lm_cfg.num_hidden_layers
        assert self.cfg.lm_cfg.moe_cfg is not None

    @property
    def layer_type(self) -> str:
        return self.cfg.lm_cfg.layer_types[self.layer_idx]

    def generate_graph(self, layer_cfg: LayerConfiguration, quantizable: bool):
        if getattr(self.cfg.lm_cfg.moe_cfg, "streamed_experts", False):
            self._generate_streamed_graph(layer_cfg, quantizable)
            return
        base = f"{self.hf_model.language_model_param_base_name}.layers.{self.layer_idx}"
        merged_lora = layer_cfg.get("lora", LoraGenMode.LORA_DISABLED) == LoraGenMode.LORA_MERGED
        shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        inputs = {"input": shape}
        quantized_embeddings = self.uses_quantized_input_embeddings and self.layer_idx == 0
        if quantized_embeddings:
            inputs["input_scale"] = (1, 1, self.num_tokens, 1)
        inputs["self_attn"] = (
            1, 1, self.num_tokens, self.cfg.lm_cfg.attn_cfg.get_q_size(self.layer_type)
        )
        graph = ModelGraph(
            self, inputs, quantizable,
            input_dtypes={"input": np.int8} if quantized_embeddings else None,
        )
        hidden = graph.inputs["input"]
        if quantized_embeddings:
            hidden = graph.dequant(hidden, graph.inputs["input_scale"])
        out_name = "out_proj" if self.check_hf_param(f"{base}.self_attn.out_proj.weight") else "o_proj"
        lora_rank = (
            self.cfg.lm_cfg.get_lora_rank(f"{base}.self_attn", out_name)
            if self.cfg.lm_cfg.lora_cfg is not None else None
        )
        attention = graph.linear(
            f"{base}.self_attn.{out_name}", graph.inputs["self_attn"],
            lora_rank=lora_rank, merged_lora=merged_lora,
        )
        residual = graph.add(hidden, attention)
        norm_hidden = graph.rms_norm(f"{base}.post_attention_layernorm", residual)
        gate_name = "gate" if self.check_hf_param(f"{base}.mlp.gate.weight") else "router"
        gate_rank = (
            self.cfg.lm_cfg.get_lora_rank(f"{base}.mlp", gate_name)
            if self.cfg.lm_cfg.lora_cfg is not None else None
        )
        logits = graph.linear(
            f"{base}.mlp.{gate_name}", norm_hidden,
            lora_rank=gate_rank, merged_lora=merged_lora,
        )
        moe = self.cfg.lm_cfg.moe_cfg
        if self.cfg.lm_cfg.arch == LlmArchType.OLMOE:
            values, indices = graph.topk(graph.softmax(logits), moe.num_experts_per_tok)
            if moe.norm_topk_prob:
                values = graph.mul(values, graph.reciprocal(graph.sum_channels(values)))
        else:
            values, indices = graph.topk(logits, moe.num_experts_per_tok)
            values = graph.softmax(values)
        graph.save([values, indices, residual, norm_hidden])

    def _generate_streamed_graph(self, layer_cfg: LayerConfiguration, quantizable: bool):
        """Router for streamed experts (Qwen3-Next, Qwen3.5-MoE).

        Attention layers project and gate attention, then add the residual. Linear-attention
        layers receive the residual sum from their linear unit. The router then normalizes,
        selects experts (softmax over all experts then top-k; with norm_topk_prob the
        renormalized top-k weights are computed as a softmax over the top-k logits)
        and seeds the expert accumulator with the residual and the gated shared expert.

        Outputs: values and indices (1, 1, N, top_k), acc (1, 1, N, hidden), norm_hidden
        (1, 1, N, hidden), then for single-token routers one (1, 1, 1, 1) weight per selected
        slot, which the runtime binds directly as the expert template's weight input.
        With MixtureOfExpertsConfig.host_topk, single-token routers instead output the expert
        logits (1, 1, 1, num_experts), acc and norm_hidden; the runtime selects the experts.
        """
        if layer_cfg.get("lora", LoraGenMode.LORA_DISABLED) != LoraGenMode.LORA_DISABLED:
            raise NotImplementedError("LoRA is not supported with streamed experts")
        base = f"{self.hf_model.language_model_param_base_name}.layers.{self.layer_idx}"
        moe = self.cfg.lm_cfg.moe_cfg
        is_attention = self.layer_type in ("full_attention", "sliding_attention")
        shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        inputs = {"input": shape}
        quantized_embeddings = (
            is_attention and self.uses_quantized_input_embeddings and self.layer_idx == 0
        )
        if quantized_embeddings:
            inputs["input_scale"] = (1, 1, self.num_tokens, 1)
        if is_attention:
            attn_shape = (
                1, 1, self.num_tokens, self.cfg.lm_cfg.attn_cfg.get_q_size(self.layer_type)
            )
            inputs["self_attn"] = attn_shape
            if self.cfg.lm_cfg.attn_cfg.attn_output_gate:
                inputs["gate"] = attn_shape
        graph = ModelGraph(
            self, inputs, quantizable,
            input_dtypes={"input": np.int8} if quantized_embeddings else None,
        )
        residual = graph.inputs["input"]
        if quantized_embeddings:
            residual = graph.dequant(residual, graph.inputs["input_scale"])
        if is_attention:
            attn = graph.inputs["self_attn"]
            if self.cfg.lm_cfg.attn_cfg.attn_output_gate:
                attn = graph.mul(attn, graph.sigmoid(graph.inputs["gate"]))
            residual = graph.add(residual, graph.linear(f"{base}.self_attn.o_proj", attn))
        norm_hidden = graph.rms_norm(f"{base}.post_attention_layernorm", residual)

        logits = graph.linear(f"{base}.mlp.gate", norm_hidden)
        host_topk = moe.host_topk and self.num_tokens == 1
        if host_topk:
            values = indices = None
        elif moe.norm_topk_prob:
            # Softmax over all experts, top-k, then renormalize equals a softmax over the
            # top-k logits: p_i / sum_topk(p_j) = exp(l_i) / sum_topk(exp(l_j)). This also avoids
            # a reciprocal, which the MLA compiler does not lower.
            values, indices = graph.topk(logits, moe.num_experts_per_tok)
            values = graph.softmax(values)
        else:
            values, indices = graph.topk(graph.softmax(logits), moe.num_experts_per_tok)

        acc = residual
        if moe.shared_expert_intermediate_size:
            shared = graph.mlp(
                f"{base}.mlp.shared_expert", norm_hidden, self.cfg.lm_cfg.mlp_cfg.act,
                projections=("gate_proj", "up_proj", "down_proj"),
            )
            shared_gate = graph.sigmoid(graph.linear(f"{base}.mlp.shared_expert_gate", norm_hidden))
            acc = graph.add(acc, graph.mul(shared, shared_gate))

        if host_topk:
            graph.save([logits, acc, norm_hidden])
            return
        outputs = [values, indices, acc, norm_hidden]
        if self.num_tokens == 1:
            outputs.extend(
                graph.slice(values, start=j, stop=j + 1, axis=3)
                for j in range(moe.num_experts_per_tok)
            )
        graph.save(outputs)
