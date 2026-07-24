#!/usr/bin/env python3
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
Generate golden ONNX files for all supported model architectures.

Run this script once to create reference ONNX files, and again whenever
the ONNX export logic is intentionally changed. Output files are stored
on the datacenter and accessed by the CI test suite.

Usage:
    python scripts/generate_reference_onnx.py --hf-models-path /path/to/huggingface
    python scripts/generate_reference_onnx.py --hf-models-path /path/to/huggingface --output /custom/output/path
"""
import argparse
import logging
import os
import tempfile
from pathlib import Path

from sima_lmm.config.layer_id import LayerID
from sima_lmm.model import FileGenMode, FileGenPrecision, VisionLanguageModel
DEFAULT_OUTPUT_PATH = Path("reference_onnx")

LAYER_IDX = 0
NUM_TOKENS = 1
MAX_NUM_TOKENS = 1024

# (hf_model_folder, model_types, image_resolution)
# model_types: list of "pre", "cache", "post", "per_layer", "conv", "vision"
_ONNX_CONFIGS = [
    # Transformer LLMs — pre, cache, post
    ("models--meta-llama--Llama-3.2-1B-Instruct",   ["pre", "cache", "post"], None),
    ("models--google--gemma-3-1b-it",               ["pre", "cache", "post"], None),
    ("models--google--gemma-4-E2B-it",              ["pre", "cache", "post", "per_layer", "vision"], [240, 240]),
    ("models--mistralai--Mistral-7B-Instruct-v0.3", ["pre", "cache", "post"], None),
    ("models--microsoft--Phi-3.5-mini-instruct",    ["pre", "cache", "post"], None),
    ("models--Qwen--Qwen2.5-0.5B-Instruct",         ["pre", "cache", "post"], None),
    ("models--Qwen--Qwen3-0.6B",                    ["pre", "cache", "post"], None),
    # LFM2 — conv only
    ("models--LiquidAI--LFM2-350M",                 ["conv"],                 None),
    # VLMs — vision encoder only
    ("models--stribomon--gemma3-siglip448",          ["vision"],               None),
    ("models--Qwen--Qwen2.5-VL-3B-Instruct",        ["vision"],               [224, 224]),
    ("models--Qwen--Qwen3-VL-2B-Instruct",          ["vision"],               [224, 224]),
    ("models--LiquidAI--LFM2-VL-450M",              ["vision"],               None),
]


def generate_pre(vlm_model: VisionLanguageModel) -> None:
    gen_config = {"precision": {LayerID("single_pre", LAYER_IDX): FileGenPrecision.BF16}}
    vlm_model.gen_files(FileGenMode.SOURCE_TO_ONNX, gen_config=gen_config, num_processes=1,
                        log_level=logging.WARNING, resume=False)


def generate_cache(vlm_model: VisionLanguageModel) -> None:
    gen_config = {"precision": {LayerID("single_cache", LAYER_IDX): FileGenPrecision.BF16}}
    vlm_model.gen_files(FileGenMode.SOURCE_TO_ONNX, gen_config=gen_config, num_processes=1,
                        log_level=logging.WARNING, resume=False)


def generate_post(vlm_model: VisionLanguageModel) -> None:
    gen_config = {"precision": {LayerID("single_post", LAYER_IDX): FileGenPrecision.BF16}}
    vlm_model.gen_files(FileGenMode.SOURCE_TO_ONNX, gen_config=gen_config, num_processes=1,
                        log_level=logging.WARNING, resume=False)


def generate_per_layer(vlm_model: VisionLanguageModel) -> None:
    gen_config = {"precision": {LayerID("single_per_layer", 0): FileGenPrecision.BF16}}
    vlm_model.gen_files(FileGenMode.SOURCE_TO_ONNX, gen_config=gen_config, num_processes=1,
                        log_level=logging.WARNING, resume=False)


def generate_conv(vlm_model: VisionLanguageModel) -> None:
    gen_config = {"precision": {LayerID("single_conv", LAYER_IDX): FileGenPrecision.BF16}}
    vlm_model.gen_files(FileGenMode.SOURCE_TO_ONNX, gen_config=gen_config, num_processes=1,
                        log_level=logging.WARNING, resume=False)


def generate_vision(vlm_model: VisionLanguageModel) -> None:
    gen_config = {"precision": {LayerID("vision", LAYER_IDX): FileGenPrecision.BF16}}
    vlm_model.gen_files(FileGenMode.SOURCE_TO_ONNX, gen_config=gen_config, num_processes=1,
                        log_level=logging.WARNING, resume=False)


_GENERATORS = {
    "pre":    generate_pre,
    "cache":  generate_cache,
    "post":   generate_post,
    "per_layer": generate_per_layer,
    "conv":   generate_conv,
    "vision": generate_vision,
}


def generate_golden_onnx(
    model_folder: str, model_types: list[str], output_path: Path,
    hf_models_path: Path,
    image_resolution: list[int] | None = None,
) -> None:
    model_path = hf_models_path / model_folder
    if not model_path.exists():
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        vlm_model = VisionLanguageModel.from_hf_cache(
            hf_cache_path=model_path,
            model_name=model_path.name,
            onnx_path=output_path,
            sima_path=Path(tmpdir),
            max_num_tokens=MAX_NUM_TOKENS,
            system_prompt=None,
            image_resolution=image_resolution,
        )
        for model_type in model_types:
            _GENERATORS[model_type](vlm_model)
            print(f"  OK  {model_folder} [{model_type}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden ONNX files")
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

    print(f"Writing golden ONNX files to: {args.output}\n")

    failed = []
    for model_folder, model_types, image_resolution in _ONNX_CONFIGS:
        print(f"Processing {model_folder}...")
        try:
            generate_golden_onnx(model_folder, model_types, args.output, args.hf_models_path, image_resolution)
        except Exception as e:
            print(f"  FAIL {model_folder}: {e}")
            failed.append(model_folder)

    print(f"\n{len(_ONNX_CONFIGS) - len(failed)}/{len(_ONNX_CONFIGS)} models generated successfully.")
    if failed:
        print("Failed:")
        for m in failed:
            print(f"  - {m}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
