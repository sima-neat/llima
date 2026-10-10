from dataclasses import dataclass

from sima_lmm.model.base import LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePostBaseModel
from sima_lmm.model.model_graph import ModelGraph


@dataclass
class LanguageMoeWeightedSumModel(LanguagePostBaseModel):
    """Sum weighted experts, then add the residual and optional final output head."""

    def __post_init__(self):
        assert self.num_tokens >= 1
        assert 0 <= self.layer_idx < self.cfg.lm_cfg.num_hidden_layers
        assert self.cfg.lm_cfg.moe_cfg is not None

    def generate_graph(self, layer_cfg: LayerConfiguration, quantizable: bool):
        shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        moe = self.cfg.lm_cfg.moe_cfg
        num_experts = moe.num_experts_per_tok if self.num_tokens == 1 else moe.num_experts
        inputs = {f"expert_{e}": shape for e in range(num_experts)}
        inputs["residual"] = shape
        graph = ModelGraph(self, inputs, quantizable)
        combined = graph.inputs["expert_0"]
        for e in range(1, num_experts):
            combined = graph.add(combined, graph.inputs[f"expert_{e}"])
        combined = graph.add(combined, graph.inputs["residual"])
        outputs = (
            self._build_post_transformer(graph, combined)
            if self.layer_idx == self.cfg.lm_cfg.num_hidden_layers - 1 else [combined]
        )
        graph.save(outputs)
