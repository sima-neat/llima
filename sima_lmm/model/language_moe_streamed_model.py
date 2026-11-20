"""Units for streamed Mixture-of-Experts models (``MixtureOfExpertsConfig.streamed_experts``).

A streamed MoE layer runs a router that seeds an accumulator, then one job per selected
expert on a shared expert template:

    acc_0 = residual + sigmoid(shared_expert_gate(x)) * shared_expert(x)
    acc_{j+1} = acc_j + w_j * expert_{e_j}(x)

where ``x`` is the post-attention-normalized hidden state. The template's projection weights
are relocatable, so the runtime binds the weights of expert ``(layer, e_j)`` into the template
instead of compiling one model per expert. The final head applies the output norm and
``lm_head`` to the last layer's accumulator.
"""

from dataclasses import dataclass

from sima_lmm.model.base import LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePostBaseModel
from sima_lmm.model.model_graph import ModelGraph

# The template is compiled from this layer/expert's weights. The runtime replaces them through
# the reloc sections, so any expert works; every expert has the same shapes.
TEMPLATE_LAYER_IDX = 0
TEMPLATE_EXPERT_IDX = 0

# Projection names in gate/up/down order, as in the HF expert MLP.
EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def expert_param_base(base_name: str, layer_idx: int, expert_idx: int) -> str:
    """HF parameter prefix of one routed expert."""
    return f"{base_name}.layers.{layer_idx}.mlp.experts.{expert_idx}"


@dataclass
class LanguageMoeExpertTemplateModel(LanguagePostBaseModel):
    """Routed expert with relocatable weights that accumulates its weighted output.

    Inputs are NHWC: norm_hidden and acc (1, 1, num_tokens, hidden), weight
    (1, 1, num_tokens, 1). The output is acc + weight * expert(norm_hidden).
    """

    def __post_init__(self):
        assert self.num_tokens >= 1
        assert self.cfg.lm_cfg.moe_cfg is not None and self.cfg.lm_cfg.moe_cfg.streamed_experts

    def generate_graph(self, layer_cfg: LayerConfiguration, quantizable: bool):
        hidden_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        graph = ModelGraph(self, {
            "norm_hidden": hidden_shape,
            "weight": (1, 1, self.num_tokens, 1),
            "acc": hidden_shape,
        }, quantizable)
        base = expert_param_base(
            self.hf_model.language_model_param_base_name, TEMPLATE_LAYER_IDX, TEMPLATE_EXPERT_IDX
        )
        gate, up, down = (
            lambda node, proj=proj: graph.linear(
                f"{base}.{proj}", node, relocatable=True, reloc_block_quantize=True
            )
            for proj in EXPERT_PROJECTIONS
        )
        x = graph.inputs["norm_hidden"]
        hidden = graph.mul(graph.activation(gate(x), self.cfg.lm_cfg.mlp_cfg.act), up(x))
        weighted = graph.mul(down(hidden), graph.inputs["weight"])
        graph.save([graph.add(graph.inputs["acc"], weighted)])


@dataclass
class LanguageMoeFusedExpertTemplateModel(LanguagePostBaseModel):
    """All top_k routed experts of one decode token in a single job.

    Slot j holds relocatable weights (compiled from expert j of TEMPLATE_LAYER_IDX); the runtime
    binds each slot to any cached expert record. Inputs are NHWC: norm_hidden and acc
    (1, 1, num_tokens, hidden), then weight_j (1, 1, num_tokens, 1) for each slot. The output is
    acc + sum_j weight_j * expert_j(norm_hidden). One job per layer replaces top_k chained jobs,
    removing their fixed per-job MLA and host costs.
    """

    def __post_init__(self):
        assert self.num_tokens == 1
        assert self.cfg.lm_cfg.moe_cfg is not None and self.cfg.lm_cfg.moe_cfg.streamed_experts

    @property
    def num_slots(self) -> int:
        return self.cfg.lm_cfg.moe_cfg.num_experts_per_tok

    def generate_graph(self, layer_cfg: LayerConfiguration, quantizable: bool):
        hidden_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        inputs = {"norm_hidden": hidden_shape, "acc": hidden_shape}
        inputs.update({f"weight_{j}": (1, 1, self.num_tokens, 1) for j in range(self.num_slots)})
        graph = ModelGraph(self, inputs, quantizable)
        x = graph.inputs["norm_hidden"]
        total = graph.inputs["acc"]
        for j in range(self.num_slots):
            base = expert_param_base(
                self.hf_model.language_model_param_base_name, TEMPLATE_LAYER_IDX, j
            )
            gate_name, up_name, down_name = (f"{base}.{proj}" for proj in EXPERT_PROJECTIONS)
            reloc = {"relocatable": True, "reloc_block_quantize": True}
            gate = graph.linear(gate_name, x, **reloc)
            up = graph.linear(up_name, x, **reloc)
            hidden = graph.mul(graph.activation(gate, self.cfg.lm_cfg.mlp_cfg.act), up)
            out = graph.linear(down_name, hidden, **reloc)
            total = graph.add(total, graph.mul(out, graph.inputs[f"weight_{j}"]))
        graph.save([total])


@dataclass
class LanguageFinalHeadModel(LanguagePostBaseModel):
    """Output norm and lm_head (or argmax) applied to the last layer's MoE accumulator."""

    def __post_init__(self):
        assert self.num_tokens >= 1
        assert self.layer_idx == self.cfg.lm_cfg.num_hidden_layers - 1

    def generate_graph(self, layer_cfg: LayerConfiguration, quantizable: bool):
        graph = ModelGraph(
            self, {"acc": (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)}, quantizable
        )
        graph.save(self._build_post_transformer(graph, graph.inputs["acc"]))
