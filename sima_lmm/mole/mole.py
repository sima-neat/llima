"""
Core logic for MoLE - Modalix Language Model Evaluator

Extends `lm_eval` from EleutherAI/lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness).
"""

import dataclasses
import traceback
from pathlib import Path
from typing import Any

import huggingface_hub
import huggingface_hub.errors
import lm_eval.evaluator
import lm_eval.loggers
import lm_eval.tasks
import rich
import rich.pretty
import torch
from loguru import logger

from sima_lmm.mole.modalixboard import ModalixBoard
from sima_lmm.mole.modalixlm import HuggingFaceLM, ModalixLM
from sima_lmm.mole.utils import make_table


@dataclasses.dataclass(frozen=True)
class Config:
    """
    Holds all configuration options for a Modalix Evaluation run. This is a frozen dataclass.

    Attributes:
        model_id: The HF repository id of the model to be evaluated.
        backend: Evaluation Hardware, has to be `hf` for huggingface, or `modalix` for Modalix.
        tasks: A list of task names to be performed during the evaluation.
        output_path: The directory where the evaluation results will be saved.
        board: A ModalixBoard instance, required when using the 'modalix' backend.
        limit: An optional limit on the number of samples to use for each task, useful for quick
            tests. Note: This option MUST NOT be used for benchmarking LLMs.
        device: The device to run the evaluation on, e.g., 'cuda:0' or 'cpu'.
        do_start_server: Whether to start the server on the board.
        do_perf: Whether to perform performance benchmarking.
        max_num_tokens: Max number of tokens. Used only for perf benchmarking.
        max_new_tokens: Max number of new tokens. Used only for perf benchmarking.
        batch_size: The batch size to use for the evaluation, defaults to 1.
        random_seed: The random seed to ensure reproducibility, defaults to 42.
    """

    model_id: str
    backend: str
    tasks: list[str]
    output_path: Path
    board: ModalixBoard | None
    limit: int | None
    device: str
    do_start_server: bool
    do_perf: bool
    max_num_tokens: int
    max_new_tokens: int | None
    batch_size: int = dataclasses.field(init=False, default=1)
    random_seed: int = dataclasses.field(init=False, default=42)

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)


def prepare_mole(
    model_id: str,
    backend: str,
    tasks: list[str] | None = None,
    output_path: str | Path | None = None,
    board: ModalixBoard | None = None,
    limit: int | None = None,
    do_start_server: bool = False,
    do_perf: bool = False,
    max_num_tokens: int = 1024,
    max_new_tokens: int = 256
) -> Config | None:
    """
    prepare_mole Prepares and validates the configuration for a MOLE evaluation run.

    Args:
        model_id: The Hugging Face model ID to be evaluated.
        backend: The evaluation backend to use ('hf' or 'modalix').
        tasks: A list of task names. If None, defaults to a standard set.
        output_path: The path to save results. If None, defaults to a 'results' subdirectory.
        board: A ModalixBoard object, required for the 'modalix' backend.
        limit: An optional limit on the number of evaluation samples.
        do_start_server: Start the server automatically.
        do_perf: Whether to perform performance benchmarking.
        max_num_tokens: Max number of tokens.
        max_new_tokens: Max number of new tokens. Used only for perf benchmarking.

    Returns:
        A Config object if the setup is successful, otherwise None.
    """
    if not Path(model_id).is_dir():
        hfapi = huggingface_hub.HfApi()
        try:
            hfapi.model_info(model_id)
        except huggingface_hub.errors.RepositoryNotFoundError as e:
            logger.error(f"Invalid model_id: {model_id}")
            logger.error(traceback.format_exception(e))
            return None

    if not do_perf and not tasks:
        logger.error("No tasks is specified")
        return None
    metadata = {"pretrained": model_id}
    task_manager = lm_eval.tasks.TaskManager(metadata=metadata)
    for task in tasks:
        if task not in task_manager.all_tasks:
            logger.error("Specified task is invalid: {}", task)
            return None

    if output_path is None:
        output_path = Path("results") / model_id
        logger.warning(f"You have not specified an output path, will default to: {output_path}")

    if isinstance(output_path, str):
        output_path = Path(output_path)

    if do_perf and limit is None:
        limit = 1
    elif not do_perf and limit:
        logger.warning(
            f"Setting a limit of {limit}, do not use these metrics for any dashboarding!"
        )
    if do_perf:
        backend = "modalix"
    match backend.strip().lower():
        case "hf":
            backend = "hf"
        case "modalix":
            backend = "modalix"
            if board is None:
                logger.error(
                    f"You have not specified a board to run board inference"
                )
                return None
            if do_start_server:
                if board.model is None:
                    logger.error(
                        "To start the server on the board, the model ID or path to the model on the"
                        "board needs to be provided"
                    )
                    return None
                if board.address is None:
                    logger.error(
                        "To start the server on the board, the hostname or IP needs to be provided"
                    )
                    return None
        case _:
            logger.error(f"Unsupported backend: {backend}")
            return None

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return Config(
        model_id=model_id,
        backend=backend,
        tasks=tasks,
        output_path=output_path,
        board=board,
        limit=limit,
        device=device,
        do_start_server=do_start_server,
        do_perf=do_perf,
        max_num_tokens=max_num_tokens,
        max_new_tokens=max_new_tokens,
    )


def _table_title(config: Config, grouped: bool = False) -> str:
    suffix = "Grouped Evaluation Results" if grouped else "Evaluation Results"
    if config.backend == "modalix":
        board = config.board.board_name if config.board is not None else "Modalix"
        board_model = (
            f" (board model: {config.board.model})"
            if config.board and config.board.model else ""
        )
        return f"Modalix backend on {board}{board_model}: {config.model_id} {suffix}"
    return f"HF backend: {config.model_id} {suffix}"


def run(config: Config) -> dict[str, Any]:
    """
    run Runs LLM Evaluation based on the provided Config object.

    This function sets up the evaluation environment, initializes the correct
    language model backend (HuggingFaceLM or ModalixLM), runs the evaluation
    using `lm_eval.evaluator.simple_evaluate`, and then prints the results in a
    formatted table.

    Args:
        config: The configuration object for the evaluation run.

    Returns:
        A dictionary containing the evaluation results.
    """

    # Eval Setup
    metadata = {"pretrained": config.model_id}
    task_manager = lm_eval.tasks.TaskManager(metadata=metadata)
    evaluation_tracker = lm_eval.loggers.EvaluationTracker(output_path=str(config.output_path))

    need_exit = False
    detected_error = False
    lm = None
    try:
        if config.backend == "modalix":
            if config.do_start_server:
                config.board.start_server()
            lm = ModalixLM(
                board=config.board, pretrained=config.model_id, batch_size=config.batch_size,
                max_length=config.max_num_tokens, device=config.device
            )
        else:
            lm = HuggingFaceLM(
                pretrained=config.model_id, batch_size=config.batch_size,
                max_length=config.max_num_tokens, device=config.device
            )

        # Eval
        results = lm_eval.evaluator.simple_evaluate(
            model=lm,
            model_args={},
            tasks=config.tasks,
            task_manager=task_manager,
            evaluation_tracker=evaluation_tracker,
            batch_size=config.batch_size,
            limit=config.limit,
            random_seed=config.random_seed,
            numpy_random_seed=config.random_seed,
            torch_random_seed=config.random_seed,
            fewshot_random_seed=config.random_seed,
            metadata=metadata,
        )
    except KeyboardInterrupt:
        logger.info("User interrupted")
        need_exit = True
    except BaseException:
        logger.exception("Unexpected error occurred")
        need_exit = True
        detected_error = True
    finally:
        if config.backend == "modalix" and config.do_start_server:
            # Ask the server to stop cleanly so model buffers are released before the next run.
            if lm is not None:
                try:
                    lm.zmq_socket.send_multipart(["stop".encode("utf-8")])
                    if not config.board.wait_for_server_stop(timeout=30):
                        logger.warning("Timed out waiting for benchmark server to stop cleanly")
                except Exception as exc:
                    logger.warning("Failed to stop benchmark server cleanly: {}", exc)
            config.board.stop_server()
        if need_exit:
            exit(int(detected_error))

    assert results, "Results were not generated!"
    evaluation_tracker.save_results_aggregated(results=results, samples=None)

    table = make_table(result_dict=results, table_title=_table_title(config))
    rich.print(table)
    if "groups" in results:
        table = make_table(
            result_dict=results,
            column="group",
            table_title=_table_title(config, grouped=True)
        )
        rich.print(table)

    return results


def perf_bench(config: Config):
    if config.do_start_server:
        config.board.start_server()
    lm = ModalixLM(
        board=config.board, pretrained=config.model_id, batch_size=config.batch_size,
        max_length=config.max_num_tokens, device=config.device
    )
    lm.perf_bench(config.limit, config.max_new_tokens)

    if config.do_start_server:
        # Stop the server.
        lm.zmq_socket.send_multipart(["stop".encode('utf-8')])
        if not config.board.wait_for_server_stop(timeout=30):
            logger.warning("Timed out waiting for benchmark server to stop cleanly")
        config.board.stop_server()
