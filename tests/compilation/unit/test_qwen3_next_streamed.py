"""Qwen3-Next streamed-MoE contracts: weight remapping, arch parsing, layer IDs, record layout."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet

from sima_lmm.config.vlm_config import ModelFormat, VlmArchType, VlmConfig, get_model_arch_gen
from sima_lmm.hf.fused_expert_params import fused_expert_source, load_fused_expert_param
from sima_lmm.hf.qwen3_next_params import Qwen3NextLinearDims, split_ba, split_qkvz
from sima_lmm.host.gen_expert_blobs import SECTION_ALIGNMENT, layout_from_map, record_size

pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


def _small_config(**overrides) -> Qwen3NextConfig:
    values = dict(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
        linear_value_head_dim=8, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, vocab_size=128, full_attention_interval=4,
    )
    values.update(overrides)
    return Qwen3NextConfig(**values)


def test_qkvz_and_ba_split_match_hf_ordering():
    config = _small_config()
    net = Qwen3NextGatedDeltaNet(config, layer_idx=0)
    dims = Qwen3NextLinearDims.from_config(config.to_dict())
    rng = np.random.default_rng(0)
    qkvz = rng.standard_normal(net.in_proj_qkvz.weight.shape).astype(np.float32)
    ba = rng.standard_normal(net.in_proj_ba.weight.shape).astype(np.float32)
    x = rng.standard_normal((1, 3, config.hidden_size)).astype(np.float32)

    # HF projects, then reorders per key-head group.
    projected_qkvz = torch.from_numpy(x @ qkvz.T)
    projected_ba = torch.from_numpy(x @ ba.T)
    q, k, v, z, b, a = net.fix_query_key_value_ordering(projected_qkvz, projected_ba)
    expected_qkv = torch.cat([t.reshape(1, 3, -1) for t in (q, k, v)], dim=-1).numpy()

    parts = {**split_qkvz(qkvz, dims), **split_ba(ba, dims)}
    np.testing.assert_allclose(x @ parts["qkv"].T, expected_qkv, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(x @ parts["z"].T, z.reshape(1, 3, -1).numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(x @ parts["b"].T, b.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(x @ parts["a"].T, a.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "text_type, arch, gen",
    [("qwen3_next", "qwen", "3_next"), ("qwen3", "qwen", "3"), ("qwen3_5_text", "qwen", "3_5"),
     ("qwen3_5_moe_text", "qwen", "3_5_moe"), ("gpt_oss", "gpt_oss", None)],
)
def test_arch_parsing(text_type, arch, gen):
    _, lm_arch, parsed_gen = get_model_arch_gen(False, text_type, text_type, None)
    assert lm_arch == arch and parsed_gen == gen


def test_arch_parsing_rejects_unknown_variants():
    with pytest.raises(NotImplementedError):
        get_model_arch_gen(False, "qwen3_omni_moe", "qwen3_omni_moe", None)


def test_streamed_layer_ids():
    cfg_dict = _small_config().to_dict()
    cfg_dict.update(architectures=["Qwen3NextForCausalLM"], dtype="bfloat16")
    vlm = VlmConfig.from_hf_config(ModelFormat.FORMAT_HF, Path("tiny"), cfg_dict)
    vlm.config_pipeline(None, None, 1024, 128, 128)
    lm = vlm.lm_cfg
    assert vlm.model_type == VlmArchType.LLM_QWEN3_NEXT
    assert lm.moe_cfg.streamed_experts and lm.moe_cfg.norm_topk_prob
    assert lm.attn_cfg.attn_output_gate and lm.rms_norm_unit_offset
    parts = [layer_id.part for layer_id in vlm.get_layer_ids()]
    assert "single_expert" not in parts and "single_weightedsum" not in parts
    assert parts.count("single_router") == 4 and parts.count("group_router") == 3
    assert parts.count("single_linear") == 3 and parts.count("single_pre") == 1
    for part in ("group_expert_template", "single_expert_template", "single_expert_fused_template",
                 "single_final_head"):
        assert parts.count(part) == 1


def test_record_layout_sorted_and_aligned():
    base = "model.layers.0.mlp.experts.0"
    rows = 16  # MLA row size on the compile platform is 16 bytes or more; use multiples

    def entry(section, size):
        return [{"section": section, "dram_start": 0, "dram_size": size, "channel_range": [0, 1, 0, 1]}]

    weight_map = {
        f"{base}.up_proj.weight": entry("reloc_c", 300 * rows),
        f"scale.{base}.up_proj.weight": entry("reloc_d", rows),
        f"{base}.gate_proj.weight": entry("reloc_a", 300 * rows),
        f"scale.{base}.gate_proj.weight": entry("reloc_b", rows),
    }
    specs = layout_from_map(weight_map, base)
    assert [s.name for s in specs] == ["reloc_a", "reloc_b", "reloc_c", "reloc_d"]
    assert all(s.offset % SECTION_ALIGNMENT == 0 for s in specs)
    assert record_size(specs) % SECTION_ALIGNMENT == 0
    assert [s.is_scale for s in specs] == [False, True, False, True]


def test_record_layout_rejects_foreign_weights():
    weight_map = {"model.layers.0.self_attn.o_proj.weight": [
        {"section": "reloc_a", "dram_start": 0, "dram_size": 256, "channel_range": [0, 1, 0, 1]}
    ]}
    with pytest.raises(ValueError):
        layout_from_map(weight_map, "model.layers.0.mlp.experts.0")


def test_fused_expert_slices_match_hf_experts():
    """Per-expert gate/up/down views of fused Qwen3.5-MoE tensors reproduce Qwen3_5MoeExperts."""
    from transformers import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

    rng = np.random.default_rng(0)
    cfg = Qwen3_5MoeTextConfig(hidden_size=32, moe_intermediate_size=16, num_experts=4)
    experts = Qwen3_5MoeExperts(cfg)
    with torch.no_grad():
        for p in (experts.gate_up_proj, experts.down_proj):
            p.copy_(torch.from_numpy(rng.standard_normal(p.shape).astype(np.float32)))
    prefix = "model.language_model.layers.0.mlp.experts"
    fused = {f"{prefix}.gate_up_proj": experts.gate_up_proj.detach().numpy(),
             f"{prefix}.down_proj": experts.down_proj.detach().numpy()}
    assert fused_expert_source(f"{prefix}.2.up_proj.weight") == (f"{prefix}.gate_up_proj", 2, "up_proj")
    assert fused_expert_source(f"{prefix}.gate_up_proj") is None

    x = rng.standard_normal((3, 32)).astype(np.float32)
    expert = 2
    want = experts(torch.from_numpy(x), torch.full((3, 1), expert), torch.ones(3, 1)).detach().numpy()
    w = {proj: load_fused_expert_param(f"{prefix}.{expert}.{proj}.weight", lambda n, e: fused[n][e])
         for proj in ("gate_proj", "up_proj", "down_proj")}
    gate, up = x @ w["gate_proj"].T, x @ w["up_proj"].T
    got = (gate / (1 + np.exp(-gate)) * up) @ w["down_proj"].T
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_qwen3_5_moe_streamed_config():
    from transformers import Qwen3_5MoeTextConfig

    cfg = Qwen3_5MoeTextConfig(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=8,
        num_experts_per_tok=2, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, vocab_size=128,
    ).to_dict()
    cfg.pop("norm_topk_prob", None)  # absent from the released checkpoints' configs
    cfg.update(architectures=["Qwen3_5MoeForCausalLM"], dtype="bfloat16")
    vlm = VlmConfig.from_hf_config(ModelFormat.FORMAT_HF, Path("tiny"), cfg)
    vlm.config_pipeline(None, None, 1024, 128, 128)
    lm = vlm.lm_cfg
    assert vlm.model_type == VlmArchType.LLM_QWEN3_5_MOE
    assert lm.moe_cfg.streamed_experts and lm.moe_cfg.norm_topk_prob
    assert lm.rms_norm_unit_offset
    parts = [layer_id.part for layer_id in vlm.get_layer_ids()]
    assert parts.count("single_router") == 4 and parts.count("single_expert_template") == 1
