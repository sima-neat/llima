import gc
from pathlib import Path

from sima_lmm.model import VisionLanguageModel
from tests.compilation.helpers.paths import require_readable_path


def load_hf_model(
    model_folder: str,
    output_path: Path,
    model_inputs_path: Path,
    *,
    max_num_tokens: int = 1024,
    image_resolution: tuple[int, int] | None = None,
) -> VisionLanguageModel:
    model_path = require_readable_path(
        model_inputs_path / model_folder,
        f"source model {model_folder}",
    )
    return VisionLanguageModel.from_hf_cache(
        hf_cache_path=model_path,
        model_name=model_path.name,
        onnx_path=output_path / "onnx",
        sima_path=output_path / "sima",
        max_num_tokens=max_num_tokens,
        system_prompt=None,
        image_resolution=(
            list(image_resolution) if image_resolution is not None else None
        ),
    )


def load_speculative_draft_model(
    target_model_folder: str,
    draft_model_folder: str,
    output_path: Path,
    model_inputs_path: Path,
    *,
    max_num_tokens: int = 1024,
) -> VisionLanguageModel:
    target_path = require_readable_path(
        model_inputs_path / target_model_folder,
        f"speculative target model {target_model_folder}",
    )
    draft_path = require_readable_path(
        model_inputs_path / draft_model_folder,
        f"speculative draft model {draft_model_folder}",
    )

    target_model = VisionLanguageModel.from_hf_cache(
        hf_cache_path=target_path,
        model_name=target_path.name,
        onnx_path=output_path / "target" / "onnx",
        sima_path=output_path / "target" / "sima",
        max_num_tokens=max_num_tokens,
        system_prompt=None,
    )
    target_model.configure_speculative_decoding(is_draft=False)

    draft_model = VisionLanguageModel.from_hf_cache(
        hf_cache_path=draft_path,
        model_name=draft_path.name,
        onnx_path=output_path / "draft" / "onnx",
        sima_path=output_path / "draft" / "sima",
        target_model=target_model,
        max_num_tokens=max_num_tokens,
        system_prompt=None,
    )

    del target_model
    gc.collect()
    return draft_model
