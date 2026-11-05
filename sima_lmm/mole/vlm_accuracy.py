"""Reproducible image/text accuracy evaluation through upstream lmms-eval."""

import copy
import importlib.metadata
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from sima_lmm.mole.multimodal import (
    MultimodalClient, UnsupportedInput, generation_settings, tree_identity,
)


@dataclass(frozen=True)
class AccuracyOptions:
    manifest: Path | None = None
    max_new_tokens: int | None = None
    enable_thinking: bool = False


def _evaluator_info() -> dict:
    return {"name": "lmms-eval", "version": importlib.metadata.version("lmms-eval")}


def choose_evaluator(tasks: list[str], text_manager, backend: str) -> str:
    """Keep text tasks on lm-eval; route image tasks to the image evaluator internally."""
    if backend != "modalix":
        return "lm-eval"
    try:
        from lm_eval.tasks._yaml_loader import load_yaml

        def load_task_yaml(path):
            return load_yaml(path, resolve_func=False, recursive=True)
    except ImportError:
        from lm_eval.utils import load_yaml_config

        def load_task_yaml(path):
            return load_yaml_config(str(path), mode="simple")

    kinds = []
    for task in tasks:
        if task not in text_manager.all_tasks:
            kinds.append("lmms-eval")
            continue
        entry = text_manager.task_index[task]
        yaml_path = entry.yaml_path if hasattr(entry, "yaml_path") else entry.get("yaml_path")
        cfg = load_task_yaml(yaml_path) if yaml_path else {}
        kinds.append("lmms-eval" if cfg.get("doc_to_image") is not None else "lm-eval")
    if len(set(kinds)) > 1:
        raise UnsupportedInput("Run text and image tasks in separate accuracy invocations")
    return kinds[0] if kinds else "lm-eval"


def _manifest(options) -> tuple[dict, Path]:
    if options.manifest is None:
        return {}, Path.cwd()
    path = options.manifest.resolve()
    manifest = json.loads(path.read_text())
    if set(manifest) - {"schema_version", "tasks"} or manifest.get("schema_version") != 1:
        raise UnsupportedInput("Accuracy manifest requires schema_version: 1 and tasks")
    if not isinstance(manifest.get("tasks"), dict):
        raise UnsupportedInput("Accuracy manifest tasks must map upstream task names to dataset declarations")
    return manifest["tasks"], path.parent


def _tasks(config, report: dict, options=None, task_names=None):
    from huggingface_hub import HfApi
    from lmms_eval.api.task import ConfigurableTask
    from lmms_eval.tasks import TaskManager

    options = options or config.accuracy
    task_names = config.tasks if task_names is None else task_names
    manager = TaskManager(model_name="modalix")
    unknown = set(task_names) - set(manager.all_subtasks)
    if unknown:
        raise UnsupportedInput(f"Unknown image tasks (use individual upstream task names): {sorted(unknown)}")
    declarations, root = _manifest(options)
    if declarations and set(declarations) != set(task_names):
        raise UnsupportedInput("Accuracy manifest tasks must exactly match the requested tasks")
    tasks = {}
    for name in task_names:
        # This pinned upstream API resolves includes/functions before dataset loading.
        task_config = copy.deepcopy(manager._get_config(name))
        if (task_config.get("output_type", "generate_until") != "generate_until"
            or not task_config.get("doc_to_visual") or "class" in task_config):
            raise UnsupportedInput(f"{name}: only single-round image/text generation tasks are supported")
        spec = declarations.get(name, {})
        if not isinstance(spec, dict) or set(spec) - {"revision", "local_path", "split"}:
            raise UnsupportedInput(f"{name}: unsupported dataset declaration")
        if task_config.get("repeats", 1) != 1 or task_config.get("num_fewshot", 0):
            raise UnsupportedInput(f"{name}: repeated/few-shot visual evaluation is not yet supported")
        split = task_config.get("test_split") or task_config.get("validation_split")
        if not split or (spec.get("split") is not None and spec["split"] != split):
            raise UnsupportedInput(f"{name}: use the upstream scoring split {split!r}")
        kwargs = task_config.get("dataset_kwargs") or {}
        if any(kwargs.get(key) for key in ("video", "From_YouTube", "builder_script")):
            raise UnsupportedInput(f"{name}: audio/video/custom dataset builders are unsupported")
        source = {"repository": task_config["dataset_path"], "subset": task_config.get("dataset_name"),
                  "split": split}
        revision = spec.get("revision") or kwargs.get("revision")
        if spec.get("local_path"):
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise UnsupportedInput(f"{name}: prepared datasets require their immutable source revision")
            local = (root / spec["local_path"]).resolve()
            source["prepared_files"] = tree_identity(local)
            if not source["prepared_files"]:
                raise UnsupportedInput(f"{name}: local dataset directory is empty or missing")
            task_config["dataset_path"] = str(local)
            task_config["dataset_kwargs"] = {"load_from_disk": True}
        else:
            revision = HfApi().dataset_info(source["repository"], revision=revision).sha
            task_config["dataset_kwargs"] = {**kwargs, "revision": revision}
        source["revision"] = revision
        report["datasets"][name] = source
        task = ConfigurableTask(config=task_config, model_name="modalix")
        if split not in task.dataset:
            raise UnsupportedInput(f"{name}: prepared dataset lacks split {split!r}")
        count = len(task.dataset[split])
        if count == 0:
            raise UnsupportedInput(f"{name}: scoring split is empty")
        source.update(rows=count, fingerprint=task.dataset[split]._fingerprint,
                      selected=min(count, config.limit) if config.limit is not None else count)
        # The upstream config may add delimiter stops during initialization.
        generation_settings(task.config.generation_kwargs, getattr(options, "max_new_tokens", None))
        tasks[name] = task
    return tasks


def _counts(report, records):
    counts = {}
    for task, source in report["datasets"].items():
        rows = [row for row in records if row["task"] == task]
        selected = source.get("selected", 0)
        counts[task] = {
            "selected": selected, "attempted": len(rows),
            "answered": sum(row["status"] == "ok" for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows),
            "unsupported": sum(row["status"] == "unsupported" for row in rows),
            "truncated": sum(row.get("runtime", {}).get("finish_reason") == "length" for row in rows),
            "not_run": max(0, selected - len(rows)),
            "scored": selected if report.get("scoring_complete") else 0,
        }
    return counts


def run_vlm_accuracy(config):
    import numpy as np
    import torch

    from sima_lmm.mole.perf_runner import write_results

    report = {"schema_version": 1, "status": "starting", "model_id": config.model_id,
              "evaluator": {}, "datasets": {},
              "settings": {"max_num_tokens": config.max_num_tokens,
                           "max_new_tokens_override": config.accuracy.max_new_tokens,
                           "enable_thinking": config.accuracy.enable_thinking,
                           "seed": config.random_seed, "limit": config.limit,
                           "cache_policy": "empty prompt cache per sample",
                           "stop_sequences": "trim answer after native-stop/budget generation"},
              "counts": {}, "results": {}, "runtime": {},
              "execution": {"endpoint": config.board.tcp_uri, "managed_server": config.do_start_server,
                            "mode": "host" if config.board.address else "devkit-local"}}
    output = config.output_path / "accuracy.json"
    records = []
    client = None
    started = False
    write_results(output, report)
    # Open fresh files; a failed run never leaves a previous run's successful sample export.
    samples_file = config.output_path / "accuracy-samples.jsonl"
    samples_file.write_text("")
    with (config.output_path / "accuracy-answers.jsonl").open("w") as answers:
        def record(row):
            records.append(row)
            answers.write(json.dumps(row, ensure_ascii=False, default=str, allow_nan=False) + "\n")
            answers.flush()
            if len(records) % 100 == 0:
                report["counts"] = _counts(report, records)
                write_results(output, report)

        try:
            report["evaluator"] = _evaluator_info()
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
            torch.manual_seed(config.random_seed)
            tasks = _tasks(config, report)
            if config.do_start_server:
                config.board.start_server(log_path=config.output_path / "server.log")
                started = True
            client = MultimodalClient(config.board, config.max_num_tokens, config.accuracy.enable_thinking)
            report["runtime"] = client.info()
            from lmms_eval.evaluator import simple_evaluate
            from sima_lmm.mole.lmms_adapter import ModalixMultimodalLM

            lm = ModalixMultimodalLM(client, record, output_budget=config.accuracy.max_new_tokens)
            report["status"] = "running"
            write_results(output, report)
            results = simple_evaluate(
                model=lm, tasks=list(tasks.values()), batch_size=1, limit=config.limit,
                task_manager=None, bootstrap_iters=0, log_samples=True, process_docs_parallel=1,
                random_seed=config.random_seed, numpy_random_seed=config.random_seed,
                torch_random_seed=config.random_seed, fewshot_random_seed=config.random_seed,
            )
            if not results or set(results.get("samples", {})) != set(tasks):
                raise RuntimeError("Evaluator did not return samples for every requested task")
            with samples_file.open("w") as samples:
                for task, rows in results["samples"].items():
                    expected = report["datasets"][task]["selected"]
                    if len(rows) != expected:
                        raise RuntimeError(f"{task}: evaluator returned {len(rows)} of {expected} selected samples")
                    for row in rows:
                        samples.write(json.dumps({"task": task, **row}, ensure_ascii=False, default=str) + "\n")
            report["results"] = {key: value for key, value in results.items() if key != "samples"}
            report.update(status="complete", scoring_complete=True)
            return report
        except BaseException as exc:
            report.update(status="failed", error=str(exc), error_type=type(exc).__name__)
            raise
        finally:
            try:
                try:
                    if client is not None:
                        client.close()
                finally:
                    if started:
                        config.board.stop_server()
            except BaseException as exc:
                report.update(status="failed", cleanup_error=str(exc), results={})
                raise
            finally:
                report["counts"] = _counts(report, records)
                write_results(output, report)
