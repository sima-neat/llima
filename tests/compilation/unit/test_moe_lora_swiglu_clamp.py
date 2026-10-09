"""GPT-OSS's SwiGLU clamp must follow a branch-mode LoRA add."""
from types import SimpleNamespace

import pytest

import sima_lmm.model.model_graph as model_graph
from sima_lmm.model.model_graph import ModelGraph


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]

RANK = 8
LIMIT = 7.0
BASE = {"gate_proj": 10.0, "up_proj": -10.0, "down_proj": 0.0}
DELTA = {"gate_proj": -5.0, "up_proj": 4.0, "down_proj": 0.0}


def _clip(value, a_min, a_max):
    return min(max(value, a_min), a_max)


class _Graph:
    """Evaluate scalar projections while recording fused and separate clamps."""

    def __init__(self):
        self.convs = []
        self.clips = []
        self.projected = {}
        self.model = SimpleNamespace(
            cfg=SimpleNamespace(lm_cfg=SimpleNamespace(get_effective_intermediate_size=lambda _: 8)),
            layer_idx=0,
        )

    def linear(self, name, data, lora_rank=None, merged_lora=False, **kwargs):
        proj = name.rsplit(".", 1)[-1]
        value = BASE[proj]
        activation = kwargs.get("activation")
        if activation is not None:
            value = _clip(value, activation.a_min, activation.a_max)
        if lora_rank and not merged_lora:
            value += DELTA[proj]
        self.convs.append(SimpleNamespace(activation=activation))
        self.projected[proj] = value
        return value

    def clip(self, data, a_min, a_max):
        self.clips.append((data, a_min, a_max))
        return _clip(data, a_min, a_max)

    def constant(self, value):
        return value[0]

    def sigmoid(self, data):
        self.gate = data / 1.702
        return 0.5

    def add(self, lhs, rhs):
        self.up = lhs
        return lhs + rhs

    def mul(self, lhs, rhs):
        return lhs * rhs


def _build(monkeypatch, lora_rank, merged_lora):
    monkeypatch.setattr(model_graph, "tensor_type", lambda _: SimpleNamespace(shape=(1, 1, 1, 8), scalar=model_graph.ScalarType.float32))
    graph = _Graph()
    ModelGraph.mlp(
        graph, "model.layers.0.mlp.experts.1", 0.0, "silu",
        projections=("gate_proj", "up_proj", "down_proj"),
        lora_ranks={name: lora_rank for name in BASE},
        merged_lora=merged_lora, expert_idx=1, swiglu_limit=LIMIT,
    )
    return graph


def test_branch_lora_clamps_after_the_adapter_add(monkeypatch):
    graph = _build(monkeypatch, lora_rank=RANK, merged_lora=False)
    assert graph.gate == _clip(BASE["gate_proj"] + DELTA["gate_proj"], -1e30, LIMIT)
    assert graph.up == _clip(BASE["up_proj"] + DELTA["up_proj"], -LIMIT, LIMIT)
    assert graph.convs[0].activation is None
    assert graph.convs[1].activation is None
    assert len(graph.clips) == 2
    assert graph.clips[0][2] == LIMIT
    assert graph.clips[1][1] == -LIMIT


def test_clamp_stays_fused_without_an_adapter(monkeypatch):
    graph = _build(monkeypatch, lora_rank=None, merged_lora=False)
    assert graph.convs[0].activation.a_max == LIMIT
    assert graph.convs[1].activation.a_min == -LIMIT
    assert not graph.clips
    assert graph.gate == LIMIT


def test_clamp_stays_fused_for_merged_lora(monkeypatch):
    graph = _build(monkeypatch, lora_rank=RANK, merged_lora=True)
    assert graph.convs[0].activation.a_max == LIMIT
    assert graph.convs[1].activation.a_min == -LIMIT
    assert not graph.clips
