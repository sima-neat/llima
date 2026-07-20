"""
Code for setting up models that is shared by various tests.
"""
from pathlib import Path

from sima_lmm.model import VisionLanguageModel
from tests.conftest import require_readable_path

def load_hf_test_model(
    model_name: str,
    generated_file_path: Path,
    max_num_tokens: int,
    models_path: Path,
) -> VisionLanguageModel:
    """
    Load a model for a test.

    A model must be in the HuggingFace safetensors format.
    This function searches for a matching directory name in the
    hardcoded test path, and it loads the model from that directory.

    Args:
        model_name: The directory name where the model is found.
            This is also used as the model's name.
        generated_file_path: Path where generated files will be created.
        max_num_tokens: Maximum number of tokens the model supports.
        models_path: Root directory containing model subdirectories.
    Returns:
        Object representing the model.
    """
    model_path = require_readable_path(models_path / model_name)
    return VisionLanguageModel.from_hf_cache(
        hf_cache_path=model_path,
        model_name=model_path.name,
        onnx_path=generated_file_path / "onnx_files",
        sima_path=generated_file_path / "sima_files",
        max_num_tokens=max_num_tokens,
        system_prompt=None,
    )
