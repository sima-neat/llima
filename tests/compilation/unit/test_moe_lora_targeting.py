import pytest

from sima_lmm.config.vlm_config import LanguageModelConfig, LoraConfig


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]

RANK = 8


def _language_config(layers_to_transform):
    """A config carrying only the LoRA settings the target lookups read."""
    cfg = LanguageModelConfig.__new__(LanguageModelConfig)
    cfg.lora_cfg = LoraConfig()
    cfg.lora_cfg.set_config({
        "r": RANK,
        "lora_alpha": 2 * RANK,
        "layers_to_transform": layers_to_transform,
        "target_modules": ["gate_proj", "up_proj", "down_proj"],
    })
    return cfg


@pytest.mark.parametrize(
    ("base_name", "module_name", "expected"),
    [
        # Selected layers are adapted for every expert.
        ("model.layers.0.mlp.experts.0", "gate_proj", RANK),
        ("model.layers.0.mlp.experts.1", "gate_proj", RANK),
        ("model.layers.0.mlp.experts.31", "down_proj", RANK),
        ("model.layers.2.mlp.experts.7", "up_proj", RANK),
        # Unselected layers are not, whatever the expert index.
        ("model.layers.1.mlp.experts.0", "gate_proj", None),
        ("model.layers.1.mlp.experts.31", "gate_proj", None),
        ("model.layers.3.mlp.experts.2", "down_proj", None),
        # Dense MLPs are unaffected.
        ("model.layers.0.mlp", "gate_proj", RANK),
        ("model.layers.1.mlp", "gate_proj", None),
        # A module outside target_modules is never adapted.
        ("model.layers.0.mlp.experts.1", "q_proj", None),
    ],
)
def test_layers_to_transform_selects_layers_not_experts(base_name, module_name, expected):
    """The expert index must not be read as a second transformer layer index."""
    cfg = _language_config([0, 2])
    assert cfg.get_lora_rank(base_name, module_name) == expected


@pytest.mark.parametrize(
    "base_name",
    ["model.layers.0.mlp.experts.1", "model.layers.9.mlp.experts.30"],
)
def test_experts_are_adapted_when_no_layer_filter_is_set(base_name):
    cfg = _language_config(None)
    assert cfg.get_lora_rank(base_name, "gate_proj") == RANK
