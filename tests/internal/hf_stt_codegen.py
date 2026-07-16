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
import argparse
import logging
from pathlib import Path

from afe.apis.error_handling_variables import enable_verbose_error_messages
from sima_utils.logging.sima_logger import sima_log_info
from sima_lmm.model import FileGenMode, FileGenPrecision, WhisperModel


enable_verbose_error_messages()


def stt_gen_files(
    model_path: Path,
    gen_modes: list[FileGenMode],
    precision: FileGenPrecision | dict[str, FileGenPrecision] = FileGenPrecision.BF16,
    num_processes: int = 1,
    part: str | None = None,
    part_idx: int | None = None,
    use_future_token_mask: bool = False,
    resume: bool = False,
):
    """Generate files for a local cached HuggingFace model.
    """
    model = WhisperModel.from_hf_cache(
        hf_cache_path=model_path,
        model_name=model_path.name,
        onnx_path=Path(f"{model_path.name}/onnx_files"),
        sima_path=Path(f"{model_path.name}/sima_files"),
        use_future_token_mask=use_future_token_mask,
    )

    log_level = logging.DEBUG
    kwargs = {
        "precision": precision,
        "num_processes": num_processes,
        "log_level": log_level,
        "resume": resume,
    }
    for gen_mode in gen_modes:
        if part != "unique":
            model.gen_files(gen_mode, part=part, part_idx=part_idx, **kwargs)
            continue

        model.gen_files(gen_mode, part="encoder", **kwargs)

        layer_idx = 0
        for name in ("init", "single_pre", "single_post"):
            model.gen_files(gen_mode, part=name, part_idx=layer_idx, **kwargs)

        model.gen_files(
            gen_mode, part="init", part_idx=model.cfg.decoder_layers - 1, **kwargs
        )
        model.gen_files(
            gen_mode, part="single_post", part_idx=model.cfg.decoder_layers - 1, **kwargs
        )

        token_idx = 0
        model.gen_files(gen_mode, part="single_cache", part_idx=token_idx, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper generate file arguments")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--devkit", action="store_true")
    part_choices = [
        "all", "unique", "encoder", "init", "single_pre", "single_post", "single_cache"
    ]
    parser.add_argument("--part", type=str, choices=part_choices, default="all")
    parser.add_argument("--part_idx", type=int)
    precision_choices = ["bf16", "a_bf16_w_int8", "a_bf16_w_int4"]
    parser.add_argument(
        "--precision", type=str, choices=precision_choices, default="a_bf16_w_int8"
    )
    parser.add_argument(
        "--encoder_precision", type=str, choices=precision_choices, default=None
    )
    parser.add_argument(
        "--decoder_precision", type=str, choices=precision_choices, default=None
    )
    parser.add_argument(
        "--decoder_init_precision", type=str, choices=precision_choices, default=None
    )
    parser.add_argument("--decoder_use_future_token_mask", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()
    sima_log_info("Arguments: %s", args)

    gen_modes = list()
    if args.all or args.onnx:
        gen_modes.append(FileGenMode.SOURCE_TO_ONNX)
    if args.all or args.quantize:
        gen_modes.append(FileGenMode.ONNX_TO_QUANT)
    if args.all or args.compile:
        gen_modes.append(FileGenMode.MODEL_SDK_COMPILE)
    if args.all or args.devkit:
        gen_modes.append(FileGenMode.DEVKIT)
    if not gen_modes:
        gen_modes = [
            FileGenMode.SOURCE_TO_ONNX, FileGenMode.ONNX_TO_QUANT, FileGenMode.MODEL_SDK_COMPILE,
            FileGenMode.DEVKIT
        ]
    precision = {"all": FileGenPrecision(args.precision)}
    if args.encoder_precision:
        precision["encoder"] = FileGenPrecision(args.encoder_precision)
    if args.decoder_precision:
        precision["decoder"] = FileGenPrecision(args.decoder_precision)
    if args.decoder_init_precision:
        precision["init"] = FileGenPrecision(args.decoder_init_precision)
    stt_gen_files(
        args.model_path, gen_modes, precision, args.num_processes, args.part, args.part_idx,
        args.decoder_use_future_token_mask, args.resume
    )
