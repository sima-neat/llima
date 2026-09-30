import argparse
import logging
import os
import sys
import time
import subprocess

from enum import Enum
from pathlib import Path

from sima_lmm.devkit import model_manager
from sima_lmm.devkit.model_manager import ModelManager
from sima_lmm.devkit.pcie_config import (
    RecvRootBusyError,
    claim_recv_root,
    recv_root_from_pep_conf,
)
from sima_lmm.devkit.utils import CLI, WEB, ZMQServer, connect, disconnect
from sima_lmm.logging import (
    configure_runtime_logging,
    sima_log_exception,
    sima_log_info,
)


MODEL_MANAGER = ModelManager()


# Common codes for both CLI and WEB modes.
class DemoMode(str, Enum):
    CLI = "cli"
    WEB = "web"


def _init_logging(mode: DemoMode, log_level: str | None) -> None:
    configure_runtime_logging(
        log_level=log_level,
        console=mode == DemoMode.WEB,
    )


def _resolve_run_model_path(model: str) -> Path:
    resolved = MODEL_MANAGER.resolve(model)
    if resolved is None:
        print("Model not found.", flush=True)
        available = MODEL_MANAGER.list()
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
        [
            "pkill",
            "--older",
            "60",
            "-f",
            "[l]lima (run|benchmark-server)( |$)",
        ],
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

    if args.pcie:
        # Only the CLI builds the PCIe file provider; WEB would get an empty
        # recv-root as its model path and fail with an unclear config error.
        if args.mode != DemoMode.CLI:
            print("--pcie is supported only with --mode cli.", flush=True)
            return 1
        # Whisper is not pulled over PCIe: it always reads a local folder.
        if args.stt_model_path and not Path(args.stt_model_path).is_dir():
            print(
                f"--stt_model_path '{args.stt_model_path}' is not a local folder. "
                "The speech model is not pulled over PCIe; give a local path.",
                flush=True,
            )
            return 1
        # The model's remote subfolder is just the positional model name (the
        # host serves it under <serve-root>/<model>), so no separate flag.
        # The receive dir defaults to the pep-daemon's default-recv root, so the
        # value the daemon already owns need not be repeated on the command line.
        pcie_recv_root = args.pcie_recv_root or recv_root_from_pep_conf()
        if not pcie_recv_root:
            print(
                "--pcie could not determine the receive directory. Pass "
                "--pcie-recv-root, or set 'default-recv' (and its [recv] path) "
                "in /etc/simaai/simaai-pep-daemon.conf.",
                flush=True,
            )
            return 1
        # Parallel load needs the whole batch on disk at once; PCIe pulls one ELF
        # at a time, so force the serial load path. This env var is read once into
        # a static (mla_model.hpp), so it MUST be set before connect()/CLI() below.
        os.environ["SIMA_LLIMA_RUN_DISABLE_PARALLEL_LOAD"] = "1"
        model_path = Path(pcie_recv_root)        # provider recv-root == model_path
        draft_model_path = None                  # draft-over-PCIe out of scope
        model_path.mkdir(parents=True, exist_ok=True)
        # Skip the local model-dir existence checks: files are not on the board yet.
    else:
        user_model_path = _resolve_run_model_path(args.model)
        try:
            model_path, draft_model_path = model_manager.resolve_target_and_draft_paths(
                user_model_path
            )
        except RuntimeError as e:
            print(str(e), flush=True)
            return 1
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

    # One PCIe model user per card: pcie-genai-backend and other --pcie runs
    # stage the same file names in the same recv root. Taken after the kill
    # above, so a replaced old session has released it. The kernel drops the
    # lock when this process exits.
    recv_claim = None
    if args.pcie:
        try:
            recv_claim = claim_recv_root()
        except RecvRootBusyError as e:
            print(
                "Another PCIe model user (pcie-genai-backend or 'llima run --pcie') is "
                f"running on this card: {e}. Only one can run at a time, because they "
                "share the PCIe receive directory. Stop it first.",
                flush=True,
            )
            return 1
        except OSError as e:
            print(f"--pcie could not take the card's receive-directory lock: {e}", flush=True)
            return 1

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
                draft_model_path,
                system_prompt,
                chat_template,
                args.pcie_serve_root if args.pcie else None,
                args.model if args.pcie else None,
                str(model_path) if args.pcie else None,
            )
        else:
            demo = WEB(
                model_path,
                args.stt_model_path,
                draft_model_path,
                system_prompt,
                chat_template,
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
        if recv_claim is not None:
            recv_claim.close()
        sima_log_info("Finalize done.")

    return 0


def search_models(args: argparse.Namespace) -> int:
    results = MODEL_MANAGER.search(args.term)
    if not results:
        print("No models found.", flush=True)
    else:
        for info in results:
            print(info.model_id)
    return 0


def pull_model(args: argparse.Namespace) -> int:
    model_dir = MODEL_MANAGER.pull(args.model)
    print(f"Downloaded to {model_dir}", flush=True)
    return 0


def list_models(args: argparse.Namespace) -> int:
    del args
    models = MODEL_MANAGER.list()
    if not models:
        print("No models found.", flush=True)
    else:
        for model_dir in models:
            print(model_dir.name)
    return 0


def rm_model(args: argparse.Namespace) -> int:
    removed = MODEL_MANAGER.remove(args.model)
    if not removed:
        print(f"Model not found: {args.model}", flush=True)
        return 1
    print(f"Removed {args.model}", flush=True)
    return 0


def benchmark_model(args: argparse.Namespace) -> int:
    user_model_path = _resolve_run_model_path(args.model)
    model_path, draft_model_path = model_manager.resolve_target_and_draft_paths(user_model_path)
    connect(logging.INFO)

    # Create a ZMQServer.
    try:
        server = ZMQServer(model_path, args.port, draft_model_path)
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
    run_parser.add_argument("--log_level", type=str, default=None, help="Logging level")
    run_parser.add_argument("--pcie", action="store_true",
        help="Pull model files over PCIe from the host instead of reading local disk.")
    run_parser.add_argument("--pcie-serve-root", default="models",
        help="Host serve-root name registered with simaai_svc (default: models).")
    run_parser.add_argument("--pcie-recv-root", default=None,
        help="Board directory where pulled files are staged one ELF at a time "
             "(default: the pep-daemon's default-recv root from "
             "/etc/simaai/simaai-pep-daemon.conf).")
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
