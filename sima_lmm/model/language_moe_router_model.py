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
