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
Generate golden VlmConfig JSON files for all supported models.

Run this script once to create reference configs, and again whenever
a new config field is intentionally added. The output files should be
committed to the repo under tests/compilation/configuration/references/.

Usage:
    python scripts/generate_reference_configs.py --hf-models-path /path/to/huggingface
    python scripts/generate_reference_configs.py --hf-models-path /path/to/huggingface --output /custom/output/path
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from transformers import AutoConfig

from sima_lmm.config.vlm_config import VlmConfig, ModelFormat

DEFAULT_OUTPUT_PATH = (
    Path(__file__).parent.parent
    / "tests"
    / "compilation"
    / "configuration"
    / "references"
)

_EXAMPLE_CONFIGS = [
    # (hf_model_folder,                                    golden_config,                          image_resolution)
    # Llama
    ("models--meta-llama--Llama-2-7b-chat-hf",            "llama2_vlm_config.json",               None),
    ("models--meta-llama--Llama-3.1-8B-Instruct",         "llama3.1_vlm_config.json",             None),
    ("models--meta-llama--Llama-3.2-1B-Instruct",         "llama3.2_vlm_config.json",             None),
    # Gemma
    ("models--google--gemma-2-2b-it",                     "gemma2_vlm_config.json",               None),
    ("models--google--gemma-3-1b-it",                     "gemma3_vlm_config.json",               None),
    ("models--google--gemma-4-E2B-it",                    "gemma4_e2b_it_vlm_config.json",        [480, 480]),
    # Mistral
    ("models--mistralai--Mistral-7B-Instruct-v0.3",       "mistral_vlm_config.json",              None),
    # Phi
    ("models--microsoft--Phi-3.5-mini-instruct",          "phi3.5_vlm_config.json",               None),
    ("models--microsoft--Phi-4-mini-instruct",            "phi4_vlm_config.json",                 None),
    # Qwen
    ("models--Qwen--Qwen2.5-0.5B-Instruct",               "qwen2.5_vlm_config.json",              None),
    ("models--Qwen--Qwen3-0.6B",                          "qwen3_vlm_config.json",                None),
    # LiquidAI
    ("models--LiquidAI--LFM2-350M",                       "lfm2_vlm_config.json",                 None),
    # VLMs
    ("models--stribomon--gemma3-siglip448",                "gemma3_siglip448_vlm_config.json",     None),
    ("models--Qwen--Qwen2.5-VL-3B-Instruct",              "qwen2.5_vl_vlm_config.json",           [448, 448]),
    ("models--Qwen--Qwen3-VL-2B-Instruct",                "qwen3_vl_vlm_config.json",             [448, 448]),
    ("models--LiquidAI--LFM2-VL-450M",                    "lfm2_vl_vlm_config.json",              None),
    ("models--LiquidAI--LFM2.5-VL-450M",                  "lfm2.5_vl_450m_vlm_config.json",       None),
]


def generate_golden_config(
    model_folder_name: str, golden_filename: str, hf_models_path: Path, output_path: Path,
    image_resolution: list[int] | None = None
) -> None:
    model_path = hf_models_path / model_folder_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    hf_config = AutoConfig.from_pretrained(model_path)
    model_config = hf_config.to_dict()

    vlm_cfg = VlmConfig.from_hf_config(ModelFormat.FORMAT_HF, model_path, model_config, image_resolution)

    vlm_dict = asdict(vlm_cfg)
    output_file = output_path / golden_filename
    output_file.write_text(json.dumps(vlm_dict, indent=4))
    print(f"  OK  {golden_filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden VlmConfig JSON files")
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

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Writing golden configs to: {args.output}\n")

    failed = []
    for model_folder_name, golden_filename, image_resolution in _EXAMPLE_CONFIGS:
        try:
            generate_golden_config(
                model_folder_name, golden_filename, args.hf_models_path, args.output, image_resolution
            )
        except Exception as e:
            print(f"  FAIL {model_folder_name}: {e}")
            failed.append(model_folder_name)

    print(f"\n{len(_EXAMPLE_CONFIGS) - len(failed)}/{len(_EXAMPLE_CONFIGS)} configs generated successfully.")
    if failed:
        print("Failed:")
        for m in failed:
            print(f"  - {m}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
