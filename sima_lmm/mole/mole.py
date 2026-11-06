"""
Core logic for MoLE - Modalix Language Model Evaluator

Extends `lm_eval` from EleutherAI/lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness).
"""

import dataclasses
import json
import traceback
from contextlib import contextmanager
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
from sima_lmm.mole.perf_runner import PerformanceOptions, run_performance, write_results


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
        input_lengths: Exact input-token lengths to benchmark, or None to generate
            automatic buckets.
        num_fewshot: Number of few-shot examples per prompt, or None for the task default.
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
    input_lengths: tuple[int, ...] | None
    performance: PerformanceOptions = dataclasses.field(default_factory=PerformanceOptions)
    num_fewshot: int | None = None
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
    max_new_tokens: int = 256,
    input_lengths: list[int] | None = None,
    performance: PerformanceOptions | None = None,
    num_fewshot: int | None = None,
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
        input_lengths: Exact input-token lengths for performance benchmarking.
            When omitted, power-of-two buckets are generated automatically.
        num_fewshot: Number of few-shot examples per prompt. Uses the task default if None.

    Returns:
        A Config object if the setup is successful, otherwise None.
    """
    performance = performance or PerformanceOptions()
    try:
        performance.validate()
        if do_perf and (max_new_tokens < 1 or max_num_tokens > 65535 or max_num_tokens <= max_new_tokens):
            raise ValueError("Performance token budgets must be positive and fit within context <= 65535")
        if do_perf and limit is not None and limit < 1:
            raise ValueError("Number of performance samples must be positive")
        if performance.dataset and input_lengths:
            raise ValueError("Dataset workloads cannot use exact --input_lengths buckets")
    except ValueError as exc:
        logger.error(str(exc))
        return None
    tasks = tasks or []
    backend = "modalix" if do_perf else backend.strip().lower()
    if backend == "modalix":
        if board is None:
            logger.error("Modalix evaluation requires a board target")
            return None
        try:
            board.host
        except ValueError as exc:
            logger.error(str(exc))
            return None
        if do_start_server and not board.model:
            logger.error("Starting a benchmark server requires --board_model")
            return None
    elif backend != "hf":
        logger.error("Unsupported backend: {}", backend)
        return None

    if not Path(model_id).is_dir():
        hfapi = huggingface_hub.HfApi()
        try:
            info = hfapi.model_info(model_id, revision=performance.model_revision)
            if do_perf:
                performance = dataclasses.replace(performance, model_revision=info.sha)
        except huggingface_hub.errors.RepositoryNotFoundError as e:
            logger.error(f"Invalid model_id: {model_id}")
            logger.error(traceback.format_exception(e))
            return None

    if not do_perf and not tasks:
        logger.error("No tasks is specified")
        return None
    normalized_input_lengths = None
    if input_lengths is not None:
        if not do_perf:
            logger.error("--input_lengths is only supported for performance benchmarking")
            return None
        if len(input_lengths) != len(set(input_lengths)):
            logger.error("--input_lengths values must be unique")
            return None
        if any(length <= 0 for length in input_lengths):
            logger.error("--input_lengths values must be positive")
            return None
        if any(length + max_new_tokens > max_num_tokens for length in input_lengths):
            logger.error(
                "Every --input_lengths value plus --max_new_tokens must be less than "
                "or equal to --max_num_tokens"
            )
            return None
        normalized_input_lengths = tuple(sorted(input_lengths))
    if num_fewshot is not None and num_fewshot < 0:
        logger.error("--num_fewshot must be zero or positive")
        return None

    if tasks:
        metadata = {"pretrained": model_id}
        task_manager = lm_eval.tasks.TaskManager(metadata=metadata)
        for task in tasks:
            if task not in task_manager.all_tasks:
                logger.error("Specified task is invalid: {}", task)
                return None

    if output_path is None:
        output_path = Path("results") / (Path(model_id).name if Path(model_id).is_dir() else model_id)
        logger.warning(f"You have not specified an output path, will default to: {output_path}")

    if isinstance(output_path, str):
        output_path = Path(output_path)

    if do_perf and not performance.dataset and limit is None:
        limit = 1
    elif not do_perf and limit:
        logger.warning(
            f"Setting a limit of {limit}, do not use these metrics for any dashboarding!"
        )
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
        input_lengths=normalized_input_lengths,
        performance=performance,
        num_fewshot=num_fewshot,
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


@contextmanager
def _benchmark_model(config: Config):
    lm = None
    started_server = False
    try:
        if config.backend == "modalix":
            if config.do_start_server:
                config.board.start_server(log_path=config.output_path / "server.log")
                started_server = True
            lm = ModalixLM(
                board=config.board, pretrained=config.model_id, batch_size=config.batch_size,
                max_length=config.max_num_tokens, device="cpu" if config.do_perf else config.device,
                revision=config.performance.model_revision or "main",
            )
        else:
            lm = HuggingFaceLM(
                pretrained=config.model_id, batch_size=config.batch_size,
                max_length=config.max_num_tokens, device=config.device,
            )
        yield lm
    finally:
        try:
            if started_server:
                config.board.stop_server()
        finally:
            if config.backend == "modalix" and lm is not None:
                lm.zmq_socket.close(linger=0)
                lm.zmq_context.term()


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

    try:
        with _benchmark_model(config) as lm:
            results = lm_eval.evaluator.simple_evaluate(
                model=lm,
                model_args={},
                tasks=config.tasks,
                task_manager=task_manager,
                evaluation_tracker=evaluation_tracker,
                batch_size=config.batch_size,
                limit=config.limit,
                num_fewshot=config.num_fewshot,
                random_seed=config.random_seed,
                numpy_random_seed=config.random_seed,
                torch_random_seed=config.random_seed,
                fewshot_random_seed=config.random_seed,
                metadata=metadata,
            )
    except KeyboardInterrupt:
        logger.info("User interrupted")
        raise SystemExit(130)
    except Exception:
        logger.exception("Unexpected error occurred")
        raise SystemExit(1)

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
    output = config.output_path / "performance.json"
    write_results(output, {"schema_version": 1, "status": "starting", "requests": [], "summary": {}})
    try:
        with _benchmark_model(config) as lm:
            return run_performance(lm, config)
    except BaseException as exc:
        # Startup/adapter failures happen before run_performance can write its report.
        report = json.loads(output.read_text())
        if report["status"] in ("starting", "complete"):
            report.update(status="failed", error=str(exc))
            write_results(output, report)
        if isinstance(exc, KeyboardInterrupt):
            raise SystemExit(130) from exc
        if isinstance(exc, Exception):
            logger.error("Performance benchmark failed: {}", exc)
            raise SystemExit(1) from exc
        raise
