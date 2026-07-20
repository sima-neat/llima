import argparse
import logging
from pathlib import Path

from sima_utils.logging.sima_logger import ScopedLogLevel
from sima_lmm.model import EvalMode, VisionLanguageModel
from sima_lmm.preproc.vlm_helper import Chat


def llm_evaluate(
    model_path: Path,
    eval_mode: EvalMode,
    max_num_tokens: int = 1024,
    system_prompt: str | None = None,
    language_group_size: int | None = None,
    language_group_offsets: list[int] | None = None,
    language_future_token_mask_size: int = 1,
    enable_chat_history: bool = False,
    quantize_kv_cache: bool = False,
):
    # Create the model.
    vlm_model = VisionLanguageModel.from_hf_cache(
        hf_cache_path=model_path,
        model_name=model_path.name,
        onnx_path=Path(f"{model_path.name}/onnx_files"),
        sima_path=Path(f"{model_path.name}/sima_files"),
        max_num_tokens=max_num_tokens,
        system_prompt=system_prompt,
        override_language_group_size=language_group_size,
        override_language_group_offsets=language_group_offsets,
        override_language_future_token_mask_size=language_future_token_mask_size,
        quantize_kv_cache=quantize_kv_cache,
    )

    chat = Chat(vlm_model.vlm_helper)
    while True:
        input_str = input(">>> ")
        if input_str == "quit":
            break
        if input_str.startswith("add image "):
            new_image_file = Path(input_str.split(" ")[-1])
            if not vlm_model.cfg.is_supported_multimodal:
                print("Inference with image is not supported.")
            elif not new_image_file.is_file():
                print(f"Image file not found: {new_image_file}.")
            else:
                chat.add_image(new_image_file)
            continue
        if input_str.startswith("clear image"):
            chat.clear_images()
            continue
        if input_str.startswith("set system "):
            system_prompt = input_str[11:]
            chat.set_system_prompt(system_prompt)
            print("Set system message and cleared chat history.")
            continue
        if input_str.startswith("clear system"):
            chat.clear_history()
            print("Cleared system message and chat history.")
            continue
        if input_str.startswith("clear history"):
            chat.clear_history()
            print("Cleared chat history")
            continue
        if input_str.startswith("print history"):
            chat.print_history()
            continue

        query = input_str
        print("Query:", query)
        chat.add_query(query)

        print("Assistant: ", end="", flush=True)
        response = vlm_model.evaluate(eval_mode, chat)
        print(response, flush=True)

        if enable_chat_history:
            chat.add_response(response.strip())
        else:
            chat.clear_messages()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM demo arguments")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--mode", type=str, choices=["hf", "onnx", "sdk"], default="onnx")
    parser.add_argument("--max_num_tokens", type=int, default=1024)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--system_prompt_file", type=Path, default=None)
    parser.add_argument("--language_group_size", type=int, default=None)
    parser.add_argument("--language_group_offsets", type=str, default=None)
    parser.add_argument("--language_future_token_mask_size", type=int, default=1)
    parser.add_argument(
        "--chat_history", type=bool, default=False, action=argparse.BooleanOptionalAction,
        dest="enable_chat_history", help="Enable chat history."
    )
    parser.add_argument(
        "--quantize_kv_cache", type=bool, default=False, action=argparse.BooleanOptionalAction,
        help="Enable KV cache quantization."
    )
    args = parser.parse_args()

    if args.system_prompt:
        system_prompt = args.system_prompt
    elif args.system_prompt_file:
        system_prompt = open(args.system_prompt_file, "r").read()
    else:
        system_prompt = None
    if args.language_group_offsets is not None:
        language_group_offsets = list(map(int, args.language_group_offsets.split(",")))
    else:
        language_group_offsets = None

    with ScopedLogLevel(logging.DEBUG):
        llm_evaluate(
            args.model_path, EvalMode(args.mode), args.max_num_tokens, system_prompt,
            args.language_group_size, language_group_offsets, args.language_future_token_mask_size,
            args.enable_chat_history, args.quantize_kv_cache
        )
