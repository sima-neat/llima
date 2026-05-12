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
import os
import sys
import time
import subprocess

from enum import Enum
from pathlib import Path

from sima_lmm.devkit.utils import CLI, WEB, ZMQServer, connect, disconnect
from sima_lmm.devkit import model_manager
from sima_utils.logging.sima_logger import (
    _get_logging_config as sima_get_logging_config,
    _initialize_logger as sima_initialize_logger,
    sima_log_exception,
    sima_log_info,
)


# Common codes for both CLI and WEB modes.
class DemoMode(str, Enum):
    CLI = "cli"
    WEB = "web"


def _init_logging(mode: DemoMode, log_level: str | None) -> None:
    config_file_path = os.path.join(os.path.dirname(__file__), "logging_config.yaml")
    os.environ["SIMA_LOGGING_CONFIG_YAML_FILE"] = os.getenv(
        "SIMA_LOGGING_CONFIG_YAML_FILE", config_file_path
    )
    logging_config = sima_get_logging_config()
    if mode == DemoMode.WEB:
        logging_config["root"]["handlers"].append("console")
    if log_level is not None:
        logging_config["handlers"]["console"]["level"] = log_level
        logging_config["handlers"]["file"]["level"] = log_level
    sima_initialize_logger(logging_config)


def _resolve_run_model_path(model: str) -> Path:
    resolved = model_manager.resolve_model_path(model)
    if resolved is None:
        print("Model not found.", flush=True)
        available = model_manager.list_models()
        if available:
            print("Available local models:", flush=True)
            for model_dir in available:
                print(f"  {model_dir}", flush=True)
        print(f"To download a model, run `llima pull {model}`.", flush=True)
        raise FileNotFoundError(model)
    return resolved


def _read_text_file(path: Path, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot open the {label} file: {path}")
    return path.read_text()


def _kill_existing_llima_session() -> None:
    result = subprocess.run(
        ["sudo", "pkill", "--older", "60", "-f", "[l]lima"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    ret_code = result.returncode
    if ret_code == 0:
        sima_log_info("Killed existing llima inference sessions.")
        time.sleep(5)
    elif ret_code in (2, 3):
        msg = f"pkill failed while killing llima sessions with return code {ret_code})."
        sima_log_exception(msg)
        raise RuntimeError("Failed to kill existing llima sessions.")


def run_model(args: argparse.Namespace) -> int:
    args.mode = DemoMode(args.mode)
    model_path = _resolve_run_model_path(args.model)
    if not (model_path / "devkit").is_dir() or not (model_path / "elf_files").is_dir():
        print(
            "Model directory missing required 'devkit' or 'elf_files' directories.",
            flush=True,
        )
        return 1
    if not (model_path / "devkit" / "vlm_config.json").is_file():
        print(
            "Model directory missing required 'devkit/vlm_config.json' file.",
            flush=True,
        )
        return 1

    # Update logging config.
    _init_logging(args.mode, args.log_level)

    # Kill any previous llima processes that may still be running
    _kill_existing_llima_session()

    # Connect to cpp api.
    connect(logging.INFO if args.log_level is None else args.log_level)

    # Configure system message.
    if args.system_prompt:
        system_prompt = args.system_prompt
    elif args.system_prompt_file:
        system_prompt = _read_text_file(args.system_prompt_file, "system prompt")
    else:
        system_prompt = None

    if args.chat_template:
        chat_template = args.chat_template
    elif args.chat_template_file:
        chat_template = _read_text_file(args.chat_template_file, "chat template")
    else:
        chat_template = None

    print("Setting up environments and loading models", flush=True)
    try:
        if args.mode == DemoMode.CLI:
            demo = CLI(
                model_path,
                args.stt_model_path,
                system_prompt,
                chat_template,
                args.parallel_load,
            )
        else:
            demo = WEB(
                model_path,
                args.stt_model_path,
                system_prompt,
                chat_template,
                args.parallel_load,
            )
    except Exception:
        msg = "Failed to create VLM or STT model"
        print(msg, flush=True)
        sima_log_exception(msg)
        raise

    try:
        demo.run()
    except Exception:
        msg = (
            "\nEncountered error while running VLM or STT. "
            "Check run.log for more info. Exiting..."
        )
        print(msg, flush=True)
        sima_log_exception(msg)
        raise
    except KeyboardInterrupt:
        msg = "\nUser interrupt detected. Exiting..."
        print(msg, flush=True)
    except BaseException:
        msg = "\nSystem interrupt detected. Exiting..."
        print(msg, flush=True)
        sima_log_exception(msg)
        raise
    finally:
        sima_log_info("Finalize starting...")
        del demo
        disconnect()
        sima_log_info("Finalize done.")

    return 0


def search_models(args: argparse.Namespace) -> int:
    results = model_manager.search_models(args.term)
    if not results:
        print("No models found.", flush=True)
    else:
        for info in results:
            print(info.model_id)
    return 0


def pull_model(args: argparse.Namespace) -> int:
    model_dir = model_manager.pull_model(args.model)
    print(f"Downloaded to {model_dir}", flush=True)
    return 0


def list_models(args: argparse.Namespace) -> int:
    del args
    models = model_manager.list_models()
    if not models:
        print("No models found.", flush=True)
    else:
        for model_dir in models:
            print(model_dir.name)
    return 0


def rm_model(args: argparse.Namespace) -> int:
    removed = model_manager.rm_model(args.model)
    if not removed:
        print(f"Model not found: {args.model}", flush=True)
        return 1
    print(f"Removed {args.model}", flush=True)
    return 0


def benchmark_model(args: argparse.Namespace) -> int:
    model_path = _resolve_run_model_path(args.model)
    connect(logging.INFO)

    # Create a ZMQServer.
    try:
        server = ZMQServer(model_path, args.port)
    except Exception:
        msg = "Failed to create server"
        print(msg, flush=True)
        sima_log_exception(msg)
        raise

    # Start the server.
    try:
        server.run()
    except KeyboardInterrupt:
        msg = "\nUser interrupt detected. Exiting..."
        print(msg, flush=True)
    except BaseException:
        msg = "\nSystem interrupt detected. Exiting..."
        print(msg, flush=True)
        sima_log_exception(msg)
        raise
    finally:
        sima_log_info("Finalize starting...")
        del server
        disconnect()
        sima_log_info("Finalize done.")


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    run_parser = subparsers.add_parser("run", help="Run a model")
    run_parser.add_argument("model", type=str, help="Model path or model ID")
    run_parser.add_argument(
        "--mode",
        type=str,
        choices=["cli", "web"],
        default="cli",
    )
    run_parser.add_argument(
        "--stt_model_path",
        type=Path,
        default=None,
        help="Path to the elf files for speech-to-text model.",
    )

    group = run_parser.add_mutually_exclusive_group()
    group.add_argument("--system_prompt", type=str, default="", help="Use system prompt.")
    group.add_argument(
        "--system_prompt_file",
        type=Path,
        default=None,
        help="Path to the file with system prompt.",
    )
    group.add_argument("--chat_template", type=str, default="", help="Use chat template.")
    group.add_argument(
        "--chat_template_file",
        type=Path,
        default=None,
        help="Path to the file with chat template.",
    )
    run_parser.add_argument(
        "--parallel_load",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Load multiple models in parallel.",
    )
    run_parser.add_argument("--log_level", type=str, default=None, help="Logging level")
    run_parser.set_defaults(func=run_model)


def _add_search_parser(subparsers: argparse._SubParsersAction) -> None:
    search_parser = subparsers.add_parser("search", help="Search remote models")
    search_parser.add_argument("term", nargs="?", default="", help="Search term (optional)")
    search_parser.set_defaults(func=search_models)


def _add_pull_parser(subparsers: argparse._SubParsersAction) -> None:
    pull_parser = subparsers.add_parser("pull", help="Download a model")
    pull_parser.add_argument("model", type=str, help="Model ID")
    pull_parser.set_defaults(func=pull_model)


def _add_list_parser(subparsers: argparse._SubParsersAction) -> None:
    list_parser = subparsers.add_parser("list", help="List local models")
    list_parser.set_defaults(func=list_models)


def _add_rm_parser(subparsers: argparse._SubParsersAction) -> None:
    rm_parser = subparsers.add_parser("rm", help="Remove a local model")
    rm_parser.add_argument("model", type=str, help="Model ID or path")
    rm_parser.set_defaults(func=rm_model)


def _add_benchmark_parser(subparsers: argparse._SubParsersAction) -> None:
    bm_parser = subparsers.add_parser(
        "benchmark-server", help="Start a server for benchmarking a model"
    )
    bm_parser.add_argument("model", type=str, help="Model ID or path")
    bm_parser.add_argument("--port", type=int, help="Listening port")
    bm_parser.set_defaults(func=benchmark_model)


def main() -> None:
    if (
        len(sys.argv) > 1
        and sys.argv[1] in ("run", "benchmark-server")
        and hasattr(os, "geteuid")
        and os.geteuid() != 0
    ):
        llima_cmd = [
            "sudo",
            "-E",
            sys.executable,
            "-m",
            "sima_lmm.devkit.devkit_demo",
            *sys.argv[1:],
        ]
        print("Re-executing with sudo for device access...", flush=True)
        result = subprocess.run(llima_cmd)
        sys.exit(result.returncode)

    parser = argparse.ArgumentParser(description="Llima unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    _add_search_parser(subparsers)
    _add_pull_parser(subparsers)
    _add_list_parser(subparsers)
    _add_rm_parser(subparsers)
    _add_benchmark_parser(subparsers)

    args = parser.parse_args()

    try:
        exit_code = args.func(args)
    except FileNotFoundError:
        sys.exit(1)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
