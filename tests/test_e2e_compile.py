#########################################################
# Copyright (C) 2025 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################
"""
End-to-end compilation test: randomly selects one model architecture per run,
compiles layer 0 single models (pre + cache + post for LLMs, conv for LFM2)
all the way to machine code via llima-compile.
"""
import random
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.conftest import require_readable_path

_COMPILE_CONFIGS = [
    "models--meta-llama--Llama-3.2-3B-Instruct",
    "models--google--gemma-3-4b-it",
    "models--mistralai--Mistral-7B-Instruct-v0.3",
    "models--microsoft--Phi-3.5-mini-instruct",
    "models--Qwen--Qwen2.5-3B-Instruct",
    "models--Qwen--Qwen3-4B",
    "models--LiquidAI--LFM2-1.2B",
]

# Compile only single (non-group) models at layer index 0.
# For transformer LLMs this gives pre[0] + cache[0] + post[0].
# For LFM2 this gives conv[0].
_CONFIG_PY = """\
def get_layer_configuration(model_properties, layer):
    if layer["is_group"] or layer["index"] != 0:
        return {"compile": False}
    return {"precision": "A_BF16_W_INT4"}
"""


@pytest.mark.premerge
@pytest.mark.serial
def test_e2e_compile(hf_models_path: Path) -> None:
    model_folder = random.choice(_COMPILE_CONFIGS)
    print(f"\nRandomly selected model: {model_folder}")
    model_path = require_readable_path(hf_models_path / model_folder)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_path = tmpdir / "config.py"
        config_path.write_text(_CONFIG_PY)

        base_cmd = [
            "llima-compile",
            "-c", str(config_path),
            "-j", "4",
            "-o", str(tmpdir / "output"),
            str(model_path),
        ]

        for stage, flag in [
            ("onnx",     "--onnx"),
            ("quantize", "--quantize"),
            ("compile",  "--compile"),
        ]:
            result = subprocess.run(
                base_cmd + [flag],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"llima-compile {stage} failed for {model_folder}:\n{result.stderr}"
            )
