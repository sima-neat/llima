from pathlib import Path

import pytest

from sima_lmm.config.vlm_config import GgufFileType
from sima_lmm.gguf.gguf_conversion import GgufModel
from tests.compilation.cases import GGUF_PARSER_CASES, GgufFileCase
from tests.compilation.helpers.paths import require_readable_path


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_source]

EXPECTED_FILE_TYPES = {
    "Q8_0": GgufFileType.Q8_0,
    "Q4_0": GgufFileType.Q4_0,
}


@pytest.mark.parametrize("case", GGUF_PARSER_CASES, ids=lambda case: case.id)
def test_parser_detects_quantization_type(
    model_inputs_path: Path, case: GgufFileCase
):
    model_path = require_readable_path(
        model_inputs_path / case.relative_path,
        f"{case.quantization} GGUF model",
    )

    model = GgufModel(model_path)

    assert model.model_config["data_type"] == EXPECTED_FILE_TYPES[case.quantization]
