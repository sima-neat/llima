# Probe whether host environment is set up by importing one afe symbol
import sys
try:
    from afe.apis.error_handling_variables import enable_verbose_error_messages
except ImportError:
    print("Missing sima-frontend library.  "
          "This program requires Palette tools to be installed.", file=sys.stderr
    )
    sys.exit(-1)
import argparse
import logging
from pathlib import Path
import psutil

from sima_lmm.config.layer_id import LayerID
from sima_lmm.gguf.gguf_conversion import GgufModel
from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel
from sima_lmm.host.configuration_helper import (
    _abort, default_configuration, read_configuration_file
)
from sima_lmm.host.compile_lora_adapter import compile_lora_adapter
from sima_lmm.model import FileGenMode, FileGenPrecision, VisionLanguageModel

def _print_precisions(precision: dict[LayerID, FileGenPrecision], for_quantize: bool):
    """
    Print diagnostic message about layers and precisions that will be processed.

    Args:
        precision: Layers to be compiled and their precision.
        for_quantize: Whether quantization will be performed.  If False,
            then precision is irrelevant, so only layer IDs will be displayed.
    """
    messages = []
    for layer_id, layer_p in precision.items():
        layer_text = f"{layer_id.part}_{layer_id.part_idx}"
        if for_quantize:
            # Show the precision after each layer.  Add spaces for column alignment.
            layer_text = f"  {layer_text:17} {layer_p}\n"
        else:
            layer_text = f"  {layer_text}\n"
        messages.append(layer_text)

    action_text = "quantized" if for_quantize else "compiled"
    message = f"Layers that will be {action_text}:\n" + "".join(messages)
    print(message, flush=True)


def gen_files(
    num_processes: int, resume: bool, model_path: Path, lora_path: Path | None,
    output_path: Path, file_gen_mode: FileGenMode, configuration_path: Path | None,
    system_prompt: str | None, chat_template: str | None, max_num_tokens: int,
    language_group_size: int, language_group_offsets: list[int] | None,
    future_token_mask_size: int, use_strided_kv_cache: bool, enable_filter_sharing: bool,
    quantize_embeddings: bool, split_mlp: bool, return_logits: bool, log_level: int,
    image_resolution: list[int] | None, draft_model_path: Path | None, draft_output_path: Path | None
):
    enable_verbose_error_messages()
    models = list()

    base_model = VisionLanguageModel.from_hf_cache(
        hf_cache_path=model_path,
        model_name=model_path.name,
        onnx_path=Path(output_path / "onnx_files"),
        sima_path=Path(output_path / "sima_files"),
        max_num_tokens=max_num_tokens,
        system_prompt=system_prompt,
        chat_template=chat_template,
        override_language_group_size=language_group_size,
        override_language_group_offsets=language_group_offsets,
        override_language_future_token_mask_size=future_token_mask_size,
        return_logits=return_logits,
        use_strided_kv_cache=use_strided_kv_cache,
        enable_filter_sharing=enable_filter_sharing,
        quantize_embeddings=quantize_embeddings,
        split_mlp=split_mlp,
        image_resolution=image_resolution
    )
    models.append(base_model)

    # Check if LoRA adapter is needed.
    if lora_path is not None:
        base_model.set_lora_adapter(lora_path)

    # Check if draft model is provided
    if draft_model_path is not None:
        base_model.configure_speculative_decoding(is_draft=False)
        draft_model = VisionLanguageModel.from_hf_cache(
            hf_cache_path=draft_model_path,
            model_name=draft_model_path.name,
            onnx_path=Path(draft_output_path / "onnx_files"),
            sima_path=Path(draft_output_path / "sima_files"),
            max_num_tokens=max_num_tokens,
            system_prompt=system_prompt,
            chat_template=chat_template,
            override_language_group_size=language_group_size,
            override_language_group_offsets=language_group_offsets,
            override_language_future_token_mask_size=future_token_mask_size,
            return_logits=return_logits,
            use_strided_kv_cache=use_strided_kv_cache,
            enable_filter_sharing=enable_filter_sharing,
            quantize_embeddings=quantize_embeddings,
            split_mlp=split_mlp,
            image_resolution=image_resolution,
            target_model=base_model
        )
        models.append(draft_model)

    for model in models:
        if configuration_path is None:
            gen_config = default_configuration(model)
        else:
            gen_config = read_configuration_file(model, configuration_path)

        if file_gen_mode == FileGenMode.ALL:
            # Use different compiler stages for HF and for GGUF
            if isinstance(model.hf_model, LocalHuggingFaceModel):
                # Check if this is an llm-compressor quantized model (AWQ/GPTQ)
                if model.hf_model.is_compressed_tensors_model():
                    # Pre-quantized models use MODEL_SDK_DIRECT like GGUF
                    modes = [
                        FileGenMode.DEVKIT, FileGenMode.SOURCE_TO_QUANT,
                        FileGenMode.MODEL_SDK_COMPILE
                    ]
                elif lora_path is not None or quantize_embeddings:
                    # Use SOURCE_TO_FP mode, which keeps track of LoRA weights
                    modes = [
                        FileGenMode.DEVKIT, FileGenMode.SOURCE_TO_FP,
                        FileGenMode.FP_TO_QUANT, FileGenMode.MODEL_SDK_COMPILE
                    ]
                else:
                    # Use ONNX mode, no need to track LoRA weights
                    modes = [
                        FileGenMode.DEVKIT, FileGenMode.SOURCE_TO_ONNX, FileGenMode.ONNX_TO_QUANT,
                        FileGenMode.MODEL_SDK_COMPILE
                    ]
            else:
                assert isinstance(model.hf_model, GgufModel)
                modes = [
                    FileGenMode.DEVKIT, FileGenMode.SOURCE_TO_QUANT,
                    FileGenMode.MODEL_SDK_COMPILE
                ]
        else:
            if isinstance(model.hf_model, GgufModel) and file_gen_mode == FileGenMode.SOURCE_TO_ONNX:
                _abort("ONNX generation mode not supported for GGUF models")
            modes = [file_gen_mode]

        _print_precisions(gen_config["precision"], FileGenMode.SOURCE_TO_QUANT in modes or FileGenMode.FP_TO_QUANT in modes)

        for mode in modes:
            model.gen_files(
                mode, gen_config=gen_config, log_level=log_level, num_processes=num_processes,
                resume=resume
            )
            print(f"Generated mode={mode.name} files for {model.model_name}", flush=True)

def _safe_resolve(p: Path) -> Path:
    """
    Resolve a path to an absolute path.
    """
    try:
        p = p.resolve(strict=False)
    except OSError:
        # This error is possible in Python 3.12 and earlier
        _abort(f"Too many levels of symbolic links in {p}")

    return p


def check_output_path_conflict(model_path: Path, output_path: Path):
    """
    Check if output_path is model_path or is a subdirectory thereof.
    If true, abort with a user error.
    """
    model_path = _safe_resolve(model_path)
    output_path = _safe_resolve(output_path)
    if output_path.is_relative_to(model_path):
        _abort(
            "Output path must not be input path or a subdirectory of it.\n"
            "Change directory or use -o to specify a different output path."
        )


# Association from command line argument names to FileGenMode enum values.
_FILE_GEN_MODE_OPTIONS: list[tuple[str, FileGenMode]] = [
    ("onnx", FileGenMode.SOURCE_TO_ONNX),
    ("source_to_fp", FileGenMode.SOURCE_TO_FP),
    ("fp_to_quant", FileGenMode.FP_TO_QUANT),
    ("quantize", FileGenMode.ONNX_TO_QUANT),
    ("model_sdk", FileGenMode.SOURCE_TO_QUANT),
    ("compile", FileGenMode.MODEL_SDK_COMPILE),
    ("devkit", FileGenMode.DEVKIT)
]

# Association from command line logging levels to numbers used by the logging API.
_LOGGING_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}


def main():
    parser = argparse.ArgumentParser(description="Vision and Language Model compiler")
    parser.add_argument(
        "model_path", type=Path,
        help="Path of GGUF model file or HuggingFace model directory"
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Directory to write output files into (default: file name of model_path)"
    )
    parser.add_argument("-j", "--jobs", type=int, help="Number of concurrent compiler processes")
    parser.add_argument(
        "--log_level", metavar="LEVEL", default="WARNING",
        help="Set the logging verbosity level (default: WARNING)"
    )

    group = parser.add_argument_group("Options to run only one compiler pass")
    egroup = group.add_mutually_exclusive_group()
    egroup.add_argument("--onnx", action="store_true", help="Compile to ONNX files")
    egroup.add_argument(
        "--quantize", action="store_true",
        help="Convert ONNX files to Model SDK files and quantize them"
    )
    egroup.add_argument(
        "--source_to_fp", action="store_true", help="Compile to floating-point Model SDK files"
    )
    egroup.add_argument(
        "--fp_to_quant", action="store_true", help="Quantize floating-point Model SDK files"
    )
    egroup.add_argument(
        "--model_sdk", action="store_true", help="Compile directly to Model SDK files"
    )
    egroup.add_argument(
        "--compile", action="store_true", help="Compile quantized Model SDK files to machine code"
    )
    egroup.add_argument(
        "--devkit", action="store_true",
        help="Produce additional files used for running model on devkit"
    )

    group = parser.add_argument_group("Model compilation parameters")
    group.add_argument(
        "--max_num_tokens", type=int, metavar="N", default=1024,
        help="Maximum number of input tokens that the model will support (default: 1024)"
    )
    group.add_argument(
        "--language_group_size", type=int, metavar="N", default=128,
        help="Compile models for a multiple of this many tokens.  "
             "Reduces compiled code size at the cost of runtime overhead for padding tokens.  "
             "(default: 128)"
    )
    group.add_argument(
        "--language_group_offsets", metavar="LIST",
        help="Comma-separated list of positive integers.  "
             "Grouped token processing will use groups starting at these indices.  "
             "(default: derived from language_group_size)"
    )
    group.add_argument(
        "--future_token_mask_size", type=int, metavar="N", default=128,
        help="Size of token mask.  "
             "Token masks reduce compiled code size at the cost of redundant computation by "
             "reusing models for generating multiple tokens.  "
             "(default: 128)"
    )
    group.add_argument(
        "--return_logits", action=argparse.BooleanOptionalAction, default=False,
        help="Return logits at the last layer output."
    )
    group.add_argument(
        "--use_strided_kv_cache", action=argparse.BooleanOptionalAction, default=False,
        help="Enables strided access to the KV cache stored in DRAM, affecting both read and "
             "write operations."
    )
    group.add_argument(
        "--enable_filter_sharing", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Enables filter sharing between group and single models to reduce the overall DRAM"
            " usage at a cost of higher TTFT and lower TPS. This is only effective when both group"
            " and single models are compiled with the same precision."
        )
    )
    group.add_argument(
        "--quantize_embeddings", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Enables embedding quantization to reduce memory consumption. This may result in a loss"
            " of accuracy."
        )
    )
    egroup = group.add_mutually_exclusive_group()
    egroup.add_argument(
        "--system_prompt", type=str, help="System prompt that is passed to the model"
    )
    egroup.add_argument(
        "--system_prompt_file", type=Path,
        help="Path of file containing system prompt that is passed to the model"
    )
    egroup.add_argument(
        "--chat_template", type=str,
        help="Chat template string to be used for conversation formatting"
    )
    egroup.add_argument(
        "--chat_template_file", type=Path,
        help="Path of the file containing the chat template string to be used "
        "for conversation formatting"
    )
    group.add_argument(
        "--input_height", type=int, metavar="N",
        help="The height of the input image (for Siglip2 based models)"
    )
    group.add_argument(
        "--input_width", type=int, metavar="N",
        help="The width of the input image (for Siglip2 based models)"
    )
    group.add_argument(
        "--draft_model_path", type=Path,
        help="Path of the EAGLE3 draft model for the base (target) model."
    )

    group = parser.add_argument_group("Options to compile LoRA")
    group.add_argument(
        "--lora_name", type=str, dest="lora_names", action="append",
        help="Names of the LoRA adapters."
    )
    group.add_argument(
        "--lora_path", type=Path, dest="lora_paths", action="append",
        help="Paths of LoRA adapter directories for the base model. "
             "If present, compile the model including LoRA adapter as parallel branches but with "
             "zero weights. The LoRA weights are applied or updated at run time."
    )
    group.add_argument(
        "--compile_lora", type=bool, action=argparse.BooleanOptionalAction, default=True,
        help="Set to compile LoRA weights."
    )

    group = parser.add_argument_group("Advanced options")
    group.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False,
        help="Only compile files that are missing"
    )
    group.add_argument(
        "-c", "--configuration_file",
        help="Configuration file with layer-specific compilation options"
    )

    args = parser.parse_args()

    num_processes = args.jobs
    if num_processes is None:
        num_processes = psutil.cpu_count(logical=False)

    if args.output is None:
        if args.draft_model_path is not None:
            base_output_path = Path(f"{args.model_path.name}-speculative-decoding")
        else:
            base_output_path = Path(args.model_path.name)
    else:
        base_output_path = Path(args.output)

    if args.draft_model_path is not None:
        output_path = base_output_path / args.model_path.name
        draft_output_path = base_output_path / args.draft_model_path.name
    else:
        output_path = base_output_path
        draft_output_path = None
     
    check_output_path_conflict(args.model_path, output_path)

    try:
        log_level = _LOGGING_LEVELS[args.log_level]
    except KeyError:
        _abort("Unrecognized logging level: " + args.log_level)

    if args.system_prompt:
        system_prompt = args.system_prompt
    elif args.system_prompt_file:
        try:
            system_prompt = open(args.system_prompt_file, "r").read()
        except IOError:
            _abort("Cannot read file: " + args.system_prompt_file)
    else:
        system_prompt = None

    if args.chat_template:
        chat_template = args.chat_template
    elif args.chat_template_file:
        try:
            chat_template = open(args.chat_template_file, "r").read()
        except IOError:
            _abort("Cannot read file: " + args.chat_template_file)
    else:
        chat_template = None

    if args.language_group_offsets is not None:
        try:
            language_group_offsets = list(map(int, args.language_group_offsets.split(",")))
        except:
            _abort(
                "Cannot parse argument value as a comma-separated list of integers: "
                + args.language_group_offsets
            )
    else:
        lgs = args.language_group_size
        language_group_offsets = list(range(0, args.max_num_tokens - lgs + 1, lgs))

    # Select a compile mode.  At most one of these flags can be used on the command line.
    for compile_mode_option, mode_flag in _FILE_GEN_MODE_OPTIONS:
        if getattr(args, compile_mode_option, None):
            # Use this mode
            break
    else:
        # No compile mode flag was used on the command line
        mode_flag = FileGenMode.ALL
    image_resolution = None
    if args.input_height is not None and args.input_width is not None:
        image_resolution = [args.input_height, args.input_width]
    elif args.input_height is not None or args.input_width is not None:
        _abort("Both --input_height and --input_width must be provided.")

    if mode_flag == FileGenMode.SOURCE_TO_ONNX and args.quantize_embeddings:
        _abort(
            "Embedding quantization is not supported for ONNX file generation mode."
        )

    lora_path_for_base_model = None
    if args.lora_names is not None and args.lora_paths is not None:
        if len(args.lora_names) != len(args.lora_paths):
            _abort("Number of --lora_name do not match the number of --lora_path")
        lora_path_for_base_model = args.lora_paths[0]
    elif args.lora_names is not None or args.lora_paths is not None:
        _abort("Number of --lora_name do not match the number of --lora_path")

    # Enable mlp splitting and LoRA is not used. The feature is not implemented for LoRA.
    split_mlp = lora_path_for_base_model is None

    gen_files(
        num_processes, args.resume, args.model_path, lora_path_for_base_model, output_path,
        mode_flag, args.configuration_file, system_prompt, chat_template, args.max_num_tokens,
        args.language_group_size, language_group_offsets, args.future_token_mask_size,
        args.use_strided_kv_cache, args.enable_filter_sharing, args.quantize_embeddings,
        split_mlp, args.return_logits, log_level, image_resolution, args.draft_model_path, draft_output_path
    )

    # Compile LoRA weights if requested.
    if args.compile_lora and args.lora_paths:
        for lora_name, lora_path in zip(args.lora_names, args.lora_paths):
            lora_output_path = output_path / "sima_files/npy_files" / lora_name
            lora_weight_map_path = output_path / "sima_files/mpk"
            compile_lora_adapter(
                args.model_path, lora_path, args.configuration_file, lora_output_path,
                lora_weight_map_path
            )


if __name__ == "__main__":  
    main()
