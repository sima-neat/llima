#########################################################
# Copyright (C) 2026 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################
"""
Code for setting up target + draft model pairs for speculative decoding tests.
"""
import gc

from pathlib import Path

from sima_lmm.model import VisionLanguageModel
from tests.conftest import require_readable_path


def load_speculative_test_draft_model(
    target_model_name: str,
    draft_model_name: str,
    generated_file_path: Path,
    max_num_tokens: int,
    models_path: Path,
) -> VisionLanguageModel:
    """
    Load the EAGLE3 draft model for a speculative decoding test, releasing the
    target model before returning. The target is needed only at draft
    construction time (to seed rope_cfg and, when the draft lacks tokenizer
    files, vlm_helper); holding it for the rest of the test costs memory and
    has caused CI OOMs.

    Both models must be in HuggingFace safetensors format and located under
    models_path.

    Args:
        target_model_name: Directory name of the target model.
        draft_model_name: Directory name of the EAGLE3 draft model.
        generated_file_path: Path where generated files will be created.
        max_num_tokens: Maximum number of tokens the model supports.
        models_path: Root directory containing model subdirectories.
    Returns:
        The constructed draft VisionLanguageModel.
    """
    target_model_path = require_readable_path(models_path / target_model_name, "target model")
    draft_model_path = require_readable_path(models_path / draft_model_name, "draft model")

    base_model = VisionLanguageModel.from_hf_cache(
        hf_cache_path=target_model_path,
        model_name=target_model_path.name,
        onnx_path=generated_file_path / target_model_name / "onnx_files",
        sima_path=generated_file_path / target_model_name / "sima_files",
        max_num_tokens=max_num_tokens,
        system_prompt=None,
    )
    base_model.configure_speculative_decoding(is_draft=False)

    draft_model = VisionLanguageModel.from_hf_cache(
        hf_cache_path=draft_model_path,
        model_name=draft_model_path.name,
        onnx_path=generated_file_path / draft_model_name / "onnx_files",
        sima_path=generated_file_path / draft_model_name / "sima_files",
        target_model=base_model,
        max_num_tokens=max_num_tokens,
        system_prompt=None
    )

    # Drop the target now that the draft has copied what it needs (rope_cfg,
    # vlm_helper fallback). Force a collection so its weights/onnx buffers are
    # freed before the test runs — keeping both loaded causes CI OOMs.
    del base_model
    gc.collect()

    return draft_model
