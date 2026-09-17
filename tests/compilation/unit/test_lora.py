from pathlib import Path

import pytest

from sima_lmm.config.layer_id import LayerID
from sima_lmm.host.compile_lora_adapter import _layer_id_from_weight_map
from sima_lmm.host.configuration_helper import (
    default_configuration_lora,
    read_configuration_file_lora,
)
from sima_lmm.model import FileGenPrecision, LoraGenMode


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


def test_lora_configuration_uses_all_compiled_language_model_parts(
    tmp_path: Path,
):
    cases = {
        "model_n128_pre_layer2_reloc.json": LayerID("group_pre", 2),
        "model_n1_post_layer2_reloc.json": LayerID("single_post", 2),
        "model_n128_layer2_conv_reloc.json": LayerID("group_conv", 2),
        "model_n1_layer2_conv_reloc.json": LayerID("single_conv", 2),
        "model_n128_layer2_linear_reloc.json": LayerID("group_linear", 2),
        "model_n1_layer2_linear_reloc.json": LayerID("single_linear", 2),
    }

    layer_ids = [
        _layer_id_from_weight_map(Path("maps") / filename)
        for filename in cases
    ]
    assert layer_ids == list(cases.values())

    configuration = default_configuration_lora(3, layer_ids=layer_ids)
    assert configuration["precision"] == {
        layer_id: FileGenPrecision.BF16 for layer_id in cases.values()
    }
    assert configuration["lora"] == {
        layer_id: LoraGenMode.LORA_BRANCH for layer_id in cases.values()
    }

    configuration_file = tmp_path / "lora_config.py"
    configuration_file.write_text(
        "def get_layer_configuration(_model, _layer):\n"
        "    return {'precision': 'BF16', 'lora': 'LORA_BRANCH'}\n"
    )
    configured = read_configuration_file_lora(
        3, configuration_file, layer_ids=layer_ids
    )
    assert configured == configuration

    with pytest.raises(ValueError, match="LoRA relocation map"):
        _layer_id_from_weight_map("model_n128_cache_token2_reloc.json")
