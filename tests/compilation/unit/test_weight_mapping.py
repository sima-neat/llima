import pytest

from sima_lmm.gguf.gguf_conversion import DEFAULT_HF_GGUF_WEIGHT_MAP


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


def _convert_with_default_gguf_map(name: str) -> str:
    for hf_name, gguf_name in DEFAULT_HF_GGUF_WEIGHT_MAP.items():
        if hf_name in name:
            name = name.replace(hf_name, gguf_name)
    return name


@pytest.mark.parametrize(
    ("hf_name", "gguf_name"),
    [
        ("model.layers.0.self_attn.q_proj.bias", "blk.0.attn_q.bias"),
        ("model.layers.0.self_attn.k_proj.bias", "blk.0.attn_k.bias"),
        ("model.layers.0.self_attn.v_proj.bias", "blk.0.attn_v.bias"),
    ],
)
def test_attention_bias_weight_map(hf_name: str, gguf_name: str):
    assert _convert_with_default_gguf_map(hf_name) == gguf_name
