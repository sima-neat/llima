#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2025 SiMa.ai, All Rights Reserved.       ||
#**************************************************************************
# NOTICE:  All information contained herein is, and remains the property of
# SiMa.ai. The intellectual and technical concepts contained herein are
# proprietary to SiMa and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
#**************************************************************************

import logging

from pathlib import Path

from afe.apis.error_handling_variables import enable_verbose_error_messages
from sima_lmm.model import FileGenMode, FileGenPrecision, WhisperModel


DEFAULT_MODEL_PATH = Path("/project/mlasw/share/huggingface/models--openai--whisper-small")


def gen_files(
    model_path: Path, num_processes: int, resume: bool, part: str | None, enable_log_probe: bool
):
    enable_verbose_error_messages()

    model = WhisperModel.from_hf_cache(
        hf_cache_path=model_path,
        model_name=model_path.name,
        onnx_path=Path(f"{model_path.name}/onnx_files"),
        sima_path=Path(f"{model_path.name}/sima_files"),
        use_future_token_mask=True,
        enable_filter_sharing=True,
        enable_log_probe=enable_log_probe,
    )

    precision = {
        "all": {"precision": FileGenPrecision.A_BF16_W_INT8},
    }
    model.gen_files(
        FileGenMode.ALL,
        precision=precision,
        log_level=logging.INFO,
        num_processes=num_processes,
        part=part,
        resume=resume,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Whisper generate file arguments")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--part",
        choices=[
            "all", "encoder", "language_detect", "init", "single_pre", "single_post",
            "single_cache", "init_log_probe", "single_post_log_probe",
        ],
        default="all",
        help="Whisper model part to compile.",
    )
    parser.add_argument("--enable_log_probe", action="store_true", default=False)
    args = parser.parse_args()
    print("Arguments:", args, flush=True)
    gen_files(args.model_path, args.num_processes, args.resume, args.part, args.enable_log_probe)
