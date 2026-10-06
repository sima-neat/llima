"""The gpt-oss SwiGLU clamp must follow a branch-mode LoRA add.

The ONNX path clips the combined projection, so the SDK path must not fold the
clamp into the convolution when an adapter branch is appended after it.
"""
from types import SimpleNamespace

import afe.ir.defines as afe_defines
import pytest

import sima_lmm.model.sima_builder as sima_builder
from sima_lmm.config.vlm_config import LlmArchType
from sima_lmm.model.language_part_base import LanguagePartBaseModel


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]

RANK = 8
LIMIT = 7.0
GATE_CLIP = "gate_clip"
UP_CLIP = "up_clip"

# Base projection and adapter contribution per projection, chosen so the base
# lands outside the clamp and base + delta lands inside it.
BASE = {"gate_proj": 10.0, "up_proj": -10.0, "down_proj": 0.0}
DELTA = {"gate_proj": -5.0, "up_proj": 4.0, "down_proj": 0.0}


def _clip(value, a_min, a_max):
    return min(max(value, a_min), a_max)


class _Builder:
    """Records and applies the clip nodes the MLP builder appends."""

    def __init__(self):
        self.clips = []

    def create_clip_node(self, data, a_min, a_max):
        self.clips.append((data, a_min, a_max))
        return _clip(data, a_min, a_max)

    def create_add_node(self, a, b):
        return a + b


def _build(monkeypatch, lora_rank, merged_lora):
    """Build one SwiGLU MLP through the real code path, with arithmetic stubs.

    Returns (conv calls, clip calls, gate value, up value).
    """
    convs = []
    gated = {}

    def fake_build_conv(builder, get_param, check_param, base_name, ifm, rank=None,
                        merged_lora=False, **kwargs):
        proj_name = base_name.rsplit(".", 1)[-1]
        value = BASE[proj_name]
        activation = kwargs.get("activation")
        # A fused activation clamps the convolution itself...
        if activation is not None:
            value = _clip(value, activation.a_min, activation.a_max)
        # ...and the adapter branch is added afterwards.
        if rank and not merged_lora:
            value += DELTA[proj_name]
        convs.append(SimpleNamespace(name=base_name, rank=rank, activation=activation))
        return value

    def fake_swiglu_clip(ifm_type, out_channels, swiglu_limit):
        return (
            SimpleNamespace(a_min=-1e30, a_max=swiglu_limit, tag=GATE_CLIP),
            SimpleNamespace(a_min=-swiglu_limit, a_max=swiglu_limit, tag=UP_CLIP),
        )

    def fake_build_swiglu(builder, gate, up):
        # Receives the gate/up values after any separate clip node.
        gated["gate"], gated["up"] = gate, up
        return 0.0

    monkeypatch.setattr(sima_builder, "build_conv_from_dense_with_lora", fake_build_conv)
    monkeypatch.setattr(sima_builder, "swiglu_clip", fake_swiglu_clip)
    monkeypatch.setattr(sima_builder, "build_swiglu", fake_build_swiglu)
    monkeypatch.setattr(
        afe_defines, "get_expected_tensor_value",
        lambda _node: SimpleNamespace(shape=(1, 1, 1, 8)),
    )

    lm_cfg = SimpleNamespace(
        lora_cfg=SimpleNamespace(r=RANK) if lora_rank else None,
        arch=LlmArchType.GPT_OSS,
        mlp_cfg=SimpleNamespace(swiglu_limit=LIMIT),
        get_effective_intermediate_size=lambda _idx: 8,
        get_lora_rank=lambda _base, _module: lora_rank,
    )
    model = SimpleNamespace(
        cfg=SimpleNamespace(lm_cfg=lm_cfg),
        layer_idx=0,
        expert_idx=1,
        get_hf_param=lambda _name: None,
        # True keeps the split-expert path, so LORA_MERGED is not rejected.
        check_hf_param=lambda _name: True,
    )
    builder = _Builder()
    ifm = SimpleNamespace(get_type=lambda: SimpleNamespace(output=None))

    LanguagePartBaseModel._build_sima_swiglu_mlp(
        model, builder, "model.layers.0.mlp.experts.1", [ifm], LIMIT, merged_lora, False
    )
    return convs, builder.clips, gated["gate"], gated["up"]


def test_branch_lora_clamps_after_the_adapter_add(monkeypatch):
    convs, clips, gate, up = _build(monkeypatch, lora_rank=RANK, merged_lora=False)

    # The ONNX ordering is clip(base + delta).
    assert gate == _clip(BASE["gate_proj"] + DELTA["gate_proj"], -1e30, LIMIT)
    assert up == _clip(BASE["up_proj"] + DELTA["up_proj"], -LIMIT, LIMIT)

    assert convs[0].activation is None
    assert convs[1].activation is None
    assert len(clips) == 2
    assert clips[0][2] == LIMIT
    assert clips[1][1] == -LIMIT


def test_clamp_stays_fused_without_an_adapter(monkeypatch):
    convs, clips, gate, _ = _build(monkeypatch, lora_rank=None, merged_lora=False)

    assert convs[0].activation.tag == GATE_CLIP
    assert convs[1].activation.tag == UP_CLIP
    assert not clips
    assert gate == _clip(BASE["gate_proj"], -1e30, LIMIT)


def test_clamp_stays_fused_for_merged_lora(monkeypatch):
    # A merged adapter is already inside the weights, so nothing is added after.
    convs, clips, _, _ = _build(monkeypatch, lora_rank=RANK, merged_lora=True)

    assert convs[0].activation.tag == GATE_CLIP
    assert convs[1].activation.tag == UP_CLIP
    assert not clips
