import logging

from pathlib import Path

from afe.apis.error_handling_variables import enable_verbose_error_messages
from sima_lmm.model import FileGenMode, FileGenPrecision, WhisperModel


DEFAULT_MODEL_PATH = Path("/project/mlasw/share/huggingface/models--openai--whisper-small")


def gen_files(
    model_path: Path,
    output_path: Path,
    num_processes: int,
    resume: bool,
    part: str | None,
    enable_log_probe: bool,
):
    enable_verbose_error_messages()

    model = WhisperModel.from_hf_cache(
        hf_cache_path=model_path,
        model_name=model_path.name,
        onnx_path=output_path / "onnx_files",
        sima_path=output_path / "sima_files",
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
    parser.add_argument(
        "-o",
        "--output",
        "--output_path",
        dest="output_path",
        type=Path,
        default=None,
        help="Directory to write model output into (default: ./<model_name>)",
    )
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--part",
        choices=[
            "all", "encoder", "language_detect", "init", "single_pre", "single_post",
            "single_cache",
        ],
        default="all",
        help="Whisper model part to compile.",
    )
    parser.add_argument("--enable_log_probe", action="store_true", default=False)
    args = parser.parse_args()
    args.output_path = args.output_path or Path(".") / args.model_path.name
    print("Arguments:", args, flush=True)
    gen_files(
        args.model_path,
        args.output_path,
        args.num_processes,
        args.resume,
        args.part,
        args.enable_log_probe,
    )
