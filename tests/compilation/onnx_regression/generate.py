"""Generate the ONNX regression matrix for one installed LLiMa revision."""

import argparse
import gc
import importlib.metadata
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import sima_lmm

from sima_lmm.config.layer_id import LayerID
from sima_lmm.model import FileGenMode, FileGenPrecision, VisionLanguageModel
from sima_lmm.model.language_cache_model import LanguageCacheModel
from sima_lmm.model.language_conv_model import LanguageConvModel
from sima_lmm.model.language_draft_fc_model import LanguageDraftFCModel
from sima_lmm.model.language_per_layer_model import LanguagePerLayerModel
from sima_lmm.model.language_post_model import LanguagePostModel
from sima_lmm.model.language_pre_model import LanguagePreModel
from sima_lmm.model.vision_model import VisionModel
from tests.compilation.cases import ONNX_REGRESSION_CASES, OnnxRegressionCase
from tests.compilation.helpers.model_factory import (
    load_hf_model,
    load_speculative_draft_model,
)


LAYER_IDX = 0
TOKEN_IDX = 2
NUM_TOKENS = 1


def _standard_model(
    case: OnnxRegressionCase, vlm_model: VisionLanguageModel
) -> tuple[object, FileGenPrecision]:
    cfg = vlm_model.cfg
    if case.component == "pre":
        model = LanguagePreModel(
            cfg,
            f"{vlm_model.model_name}_language_n{NUM_TOKENS}_pre_layer{LAYER_IDX}",
            onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path,
            hf_model=vlm_model.hf_model,
            num_tokens=NUM_TOKENS,
            layer_idx=LAYER_IDX,
        )
    elif case.component == "cache":
        model = LanguageCacheModel(
            cfg,
            f"{vlm_model.model_name}_language_n{NUM_TOKENS}_cache_token{LAYER_IDX}",
            onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path,
            hf_model=vlm_model.hf_model,
            num_tokens=NUM_TOKENS,
            token_idx=LAYER_IDX,
            logit_softcapping=cfg.lm_cfg.attn_logit_softcapping,
        )
    elif case.component == "post":
        model = LanguagePostModel(
            cfg,
            f"{vlm_model.model_name}_language_n{NUM_TOKENS}_post_layer{LAYER_IDX}",
            onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path,
            hf_model=vlm_model.hf_model,
            num_tokens=NUM_TOKENS,
            layer_idx=LAYER_IDX,
            final_softcapping=None,
        )
    elif case.component == "per_layer":
        model = LanguagePerLayerModel(
            cfg,
            f"{vlm_model.model_name}_language_n{NUM_TOKENS}_per_layer",
            onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path,
            hf_model=vlm_model.hf_model,
            num_tokens=NUM_TOKENS,
        )
    elif case.component == "conv":
        model = LanguageConvModel(
            cfg,
            f"{vlm_model.model_name}_language_n{NUM_TOKENS}_layer{LAYER_IDX}_conv",
            onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path,
            hf_model=vlm_model.hf_model,
            num_tokens=NUM_TOKENS,
            layer_idx=LAYER_IDX,
            final_softcapping=cfg.lm_cfg.final_logit_softcapping,
        )
    elif case.component == "vision":
        vision_model = VisionModel(
            cfg,
            vlm_model.vision_model_name,
            onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path,
            hf_model=vlm_model.hf_model,
        )
        model = vision_model._get_part_model(LAYER_IDX)
    else:
        raise ValueError(f"Unsupported standard ONNX component: {case.component}")

    return model, FileGenPrecision.BF16


def _speculative_model(
    case: OnnxRegressionCase, draft_model: VisionLanguageModel
) -> tuple[object, dict]:
    cfg = draft_model.cfg
    num_tokens = cfg.lm_cfg.speculative_decoding_cfg.speculative_budget
    if case.component == "pre":
        model = LanguagePreModel(
            cfg,
            f"{draft_model.model_name}_language_n{num_tokens}_pre_layer{LAYER_IDX}",
            onnx_path=draft_model.onnx_path,
            sima_path=draft_model.sima_path,
            hf_model=draft_model.hf_model,
            num_tokens=num_tokens,
            layer_idx=LAYER_IDX,
        )
        layer_id = LayerID("single_pre", LAYER_IDX)
    elif case.component == "cache":
        model = LanguageCacheModel(
            cfg,
            f"{draft_model.model_name}_language_n{num_tokens}_cache_token{TOKEN_IDX}",
            onnx_path=draft_model.onnx_path,
            sima_path=draft_model.sima_path,
            hf_model=draft_model.hf_model,
            num_tokens=num_tokens,
            token_idx=TOKEN_IDX,
            logit_softcapping=cfg.lm_cfg.attn_logit_softcapping,
        )
        layer_id = LayerID("single_cache", TOKEN_IDX)
    elif case.component == "post":
        model = LanguagePostModel(
            cfg,
            f"{draft_model.model_name}_language_n{num_tokens}_post_layer{LAYER_IDX}",
            onnx_path=draft_model.onnx_path,
            sima_path=draft_model.sima_path,
            hf_model=draft_model.hf_model,
            num_tokens=num_tokens,
            layer_idx=LAYER_IDX,
            final_softcapping=cfg.lm_cfg.final_logit_softcapping,
        )
        layer_id = LayerID("single_post", LAYER_IDX)
    elif case.component == "draft_fc":
        model = LanguageDraftFCModel(
            cfg,
            f"{draft_model.model_name}_language_n{num_tokens}_draft_fc",
            onnx_path=draft_model.onnx_path,
            sima_path=draft_model.sima_path,
            hf_model=draft_model.hf_model,
            num_tokens=num_tokens,
        )
        layer_id = LayerID("single_draft_fc", 0)
    else:
        raise ValueError(f"Unsupported speculative ONNX component: {case.component}")

    return model, {"precision": {layer_id: FileGenPrecision.BF16}}


def _record_model(
    manifest: dict,
    case: OnnxRegressionCase,
    model,
    output_dir: Path,
) -> None:
    onnx_path = Path(model.onnx_file_name).resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX was not generated for {case.id}: {onnx_path}")
    manifest["cases"][case.id] = {
        "component": case.component,
        "mode": case.mode,
        "status": "available",
        "onnx_path": str(onnx_path.relative_to(output_dir.resolve())),
    }


def _record_unavailable(
    manifest: dict,
    cases: list[OnnxRegressionCase],
    error: Exception,
) -> None:
    reason = f"{type(error).__name__}: {error}"
    for case in cases:
        manifest["cases"][case.id] = {
            "component": case.component,
            "mode": case.mode,
            "status": "unavailable",
            "reason": reason,
        }
        print(
            f"Informative baseline ONNX unavailable for {case.id}: {reason}",
            flush=True,
        )


def _generate_standard_cases(
    cases: list[OnnxRegressionCase],
    model_inputs_path: Path,
    output_dir: Path,
    manifest: dict,
    allow_informative_unavailable: bool,
) -> None:
    grouped: dict[tuple, list[OnnxRegressionCase]] = defaultdict(list)
    for case in cases:
        grouped[(case.model_folder, case.image_resolution)].append(case)

    for group_index, ((model_folder, image_resolution), group_cases) in enumerate(
        grouped.items()
    ):
        model_output = output_dir / f"standard-{group_index}"
        try:
            vlm_model = load_hf_model(
                model_folder,
                model_output,
                model_inputs_path,
                image_resolution=image_resolution,
            )
        except Exception as error:
            if not allow_informative_unavailable or any(
                case.mode != "informative" for case in group_cases
            ):
                raise
            _record_unavailable(manifest, group_cases, error)
            continue

        for case in group_cases:
            try:
                model, precision = _standard_model(case, vlm_model)
                model.gen_files(
                    FileGenMode.SOURCE_TO_ONNX,
                    layer_cfg={"precision": precision},
                    log_level=logging.WARNING,
                    resume=False,
                )
                _record_model(manifest, case, model, output_dir)
            except Exception as error:
                if (
                    not allow_informative_unavailable
                    or case.mode != "informative"
                ):
                    raise
                _record_unavailable(manifest, [case], error)
        del vlm_model
        gc.collect()


def _generate_speculative_cases(
    cases: list[OnnxRegressionCase],
    model_inputs_path: Path,
    output_dir: Path,
    manifest: dict,
    allow_informative_unavailable: bool,
) -> None:
    grouped: dict[tuple[str, str], list[OnnxRegressionCase]] = defaultdict(list)
    for case in cases:
        assert case.target_model_folder is not None
        grouped[(case.target_model_folder, case.model_folder)].append(case)

    for group_index, ((target_folder, draft_folder), group_cases) in enumerate(
        grouped.items()
    ):
        model_output = output_dir / f"speculative-{group_index}"
        try:
            draft_model = load_speculative_draft_model(
                target_folder,
                draft_folder,
                model_output,
                model_inputs_path,
            )
        except Exception as error:
            if not allow_informative_unavailable or any(
                case.mode != "informative" for case in group_cases
            ):
                raise
            _record_unavailable(manifest, group_cases, error)
            continue

        for case in group_cases:
            try:
                model, gen_config = _speculative_model(case, draft_model)
                draft_model.gen_files(
                    FileGenMode.SOURCE_TO_ONNX,
                    gen_config=gen_config,
                    num_processes=1,
                    log_level=logging.WARNING,
                    resume=False,
                )
                _record_model(manifest, case, model, output_dir)
            except Exception as error:
                if (
                    not allow_informative_unavailable
                    or case.mode != "informative"
                ):
                    raise
                _record_unavailable(manifest, [case], error)
        del draft_model
        gc.collect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-inputs-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--expected-package-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--allow-informative-unavailable", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    module_path = Path(sima_lmm.__file__).resolve()
    expected_package_root = args.expected_package_root.resolve()
    if expected_package_root not in module_path.parents:
        raise RuntimeError(
            f"sima_lmm imported from {module_path}, outside {expected_package_root}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    enabled_cases = [
        case for case in ONNX_REGRESSION_CASES if case.mode != "disabled"
    ]
    manifest = {
        "revision": args.revision,
        "package_version": importlib.metadata.version("sima-lmm"),
        "package_path": str(module_path),
        "allow_informative_unavailable": args.allow_informative_unavailable,
        "cases": {},
    }
    print(
        f"Generating ONNX for revision {manifest['revision']} with "
        f"sima-lmm {manifest['package_version']} from {manifest['package_path']}",
        flush=True,
    )

    standard_cases = [
        case for case in enabled_cases if case.target_model_folder is None
    ]
    speculative_cases = [
        case for case in enabled_cases if case.target_model_folder is not None
    ]
    _generate_standard_cases(
        standard_cases,
        args.model_inputs_path,
        args.output_dir,
        manifest,
        args.allow_informative_unavailable,
    )
    _generate_speculative_cases(
        speculative_cases,
        args.model_inputs_path,
        args.output_dir,
        manifest,
        args.allow_informative_unavailable,
    )

    expected_ids = {case.id for case in enabled_cases}
    if set(manifest["cases"]) != expected_ids:
        raise RuntimeError("Generated ONNX manifest does not match enabled case matrix")
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2) + os.linesep,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
