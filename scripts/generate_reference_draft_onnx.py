#!/usr/bin/env python3
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
Generate golden ONNX files for the EAGLE3 speculative decoding draft model.

Run this script once to create reference ONNX files, and again whenever
the ONNX export logic is intentionally changed. Output files are stored
on the datacenter and accessed by the CI test suite.

Usage:
    python scripts/generate_reference_draft_onnx.py --hf-models-path /path/to/huggingface
    python scripts/generate_reference_draft_onnx.py --hf-models-path /path/to/huggingface --output /custom/output/path
"""
import argparse
import logging
import os
import tempfile
from pathlib import Path

from sima_lmm.config.layer_id import LayerID
from sima_lmm.model import FileGenMode, FileGenPrecision, VisionLanguageModel
DEFAULT_OUTPUT_PATH = Path("reference_draft_onnx")

LAYER_IDX = 0
TOKEN_IDX = 2
MAX_NUM_TOKENS = 1024

# (target_model_folder, draft_model_folder, model_types)
# model_types: list of "pre", "post", "fc", "cache"
_DRAFT_ONNX_CONFIGS = [
    ("models--meta-llama--Llama-3.1-8B-Instruct", "models--lmsys--SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge", ["pre", "post", "fc", "cache"],),
]


def generate_pre(draft_model: VisionLanguageModel) -> None:
    gen_config = {"precision": {LayerID("single_pre", LAYER_IDX): FileGenPrecision.BF16}}
    draft_model.gen_files(FileGenMode.SOURCE_TO_ONNX, gen_config=gen_config, num_processes=1,
                          log_level=logging.WARNING, resume=False)


def generate_post(draft_model: VisionLanguageModel) -> None:
    gen_config = {"precision": {LayerID("single_post", LAYER_IDX): FileGenPrecision.BF16}}
    draft_model.gen_files(FileGenMode.SOURCE_TO_ONNX, gen_config=gen_config, num_processes=1,
                          log_level=logging.WARNING, resume=False)


def generate_fc(draft_model: VisionLanguageModel) -> None:
    gen_config = {"precision": {LayerID("single_draft_fc", 0): FileGenPrecision.BF16}}
    draft_model.gen_files(FileGenMode.SOURCE_TO_ONNX, gen_config=gen_config, num_processes=1,
                          log_level=logging.WARNING, resume=False)


def generate_cache(draft_model: VisionLanguageModel) -> None:
    gen_config = {"precision": {LayerID("single_cache", TOKEN_IDX): FileGenPrecision.BF16}}
    draft_model.gen_files(FileGenMode.SOURCE_TO_ONNX, gen_config=gen_config, num_processes=1,
                          log_level=logging.WARNING, resume=False)


_GENERATORS = {
    "pre":   generate_pre,
    "post":  generate_post,
    "fc":    generate_fc,
    "cache": generate_cache,
}


def generate_golden_draft_onnx(
    target_model_folder: str,
    draft_model_folder: str,
    model_types: list[str],
    hf_models_path: Path,
    output_path: Path,
) -> None:
    target_model_path = hf_models_path / target_model_folder
    draft_model_path  = hf_models_path / draft_model_folder

    if not target_model_path.exists():
        raise FileNotFoundError(f"Target model not found: {target_model_path}")
    if not draft_model_path.exists():
        raise FileNotFoundError(f"Draft model not found: {draft_model_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        base_model = VisionLanguageModel.from_hf_cache(
            hf_cache_path=target_model_path,
            model_name=target_model_path.name,
            onnx_path=Path(tmpdir) / "target" / "onnx",
            sima_path=Path(tmpdir) / "target" / "sima",
            max_num_tokens=MAX_NUM_TOKENS,
            system_prompt=None,
        )
        base_model.configure_speculative_decoding(is_draft=False)

        draft_model = VisionLanguageModel.from_hf_cache(
            hf_cache_path=draft_model_path,
            model_name=draft_model_folder,
            onnx_path=output_path,
            sima_path=Path(tmpdir) / "draft" / "sima",
            max_num_tokens=MAX_NUM_TOKENS,
            system_prompt=None,
            target_model=base_model
        )

        for model_type in model_types:
            _GENERATORS[model_type](draft_model)
            print(f"  OK  {draft_model_folder} [{model_type}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden ONNX files for the draft model")
    parser.add_argument(
        "--hf-models-path", type=Path, default=os.environ.get("LLIMA_HF_MODELS_PATH"),
        help="Root directory containing HuggingFace model cache folders. "
             "Can also be set with LLIMA_HF_MODELS_PATH."
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help=f"Output directory (default: {DEFAULT_OUTPUT_PATH})"
    )
    args = parser.parse_args()
    if args.hf_models_path is None:
        parser.error("--hf-models-path is required unless LLIMA_HF_MODELS_PATH is set")

    print(f"Writing golden draft ONNX files to: {args.output}\n")

    failed = []
    for target_folder, draft_folder, model_types in _DRAFT_ONNX_CONFIGS:
        print(f"Processing {target_folder} + {draft_folder}...")
        try:
            generate_golden_draft_onnx(target_folder, draft_folder, model_types, args.hf_models_path, args.output)
        except Exception as e:
            print(f"  FAIL {draft_folder}: {e}")
            failed.append(draft_folder)

    print(f"\n{len(_DRAFT_ONNX_CONFIGS) - len(failed)}/{len(_DRAFT_ONNX_CONFIGS)} configs generated successfully.")
    if failed:
        print("Failed:")
        for m in failed:
            print(f"  - {m}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
