import argparse
import logging
from pathlib import Path

from sima_utils.logging.sima_logger import ScopedLogLevel
from sima_lmm.model import EvalMode, WhisperModel


def stt_evaluate(model_path: Path, eval_mode: EvalMode):
    # Create the model.
    model = WhisperModel.from_hf_cache(
        hf_cache_path=model_path,
        model_name=model_path.name,
        onnx_path=Path(f"{model_path.name}/onnx_files"),
        sima_path=Path(f"{model_path.name}/sima_files"),
    )

    audio_file = None
    while True:
        input_str = input(">>> ")
        if input_str == "quit":
            break
        if input_str == "set audio":
            audio_file = None
            continue
        if input_str.startswith("set audio "):
            new_audio_file = Path(input_str.split(" ")[-1])
            if not new_audio_file.is_file():
                print(f"Audio file not found: {new_audio_file}")
                continue
            audio_file = new_audio_file

            print("Transcription: ", end="", flush=True)
            output_text = model.evaluate(eval_mode, audio_file)
            print(output_text, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM demo arguments")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--mode", type=str, choices=["hf", "onnx", "sdk"], default="onnx")
    args = parser.parse_args()

    with ScopedLogLevel(logging.DEBUG):
        stt_evaluate(args.model_path, EvalMode(args.mode))
