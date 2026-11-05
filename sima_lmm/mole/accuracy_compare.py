"""Compare completed VLM accuracy exports without loading an evaluator or model."""

import copy
import json
import re
from pathlib import Path


class InvalidRun(ValueError):
    """A result directory cannot supply a complete comparison."""


def _read(path: Path):
    def invalid_constant(value):
        raise ValueError(f"Invalid JSON number: {value}")

    try:
        return json.loads(path.read_text(), parse_constant=invalid_constant)
    except (OSError, ValueError) as exc:
        raise InvalidRun(f"{path}: {exc}") from exc


def _rows(path: Path) -> dict:
    rows = {}
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise InvalidRun(f"{path}: {exc}") from exc
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
            key = (row["task"], row["doc_id"])
            if not isinstance(key[0], str) or type(key[1]) is not int:
                raise ValueError("task must be a string and doc_id must be an integer")
            if key in rows:
                raise ValueError(f"Duplicate sample: {key}")
            rows[key] = row
        except (ValueError, KeyError, TypeError) as exc:
            raise InvalidRun(f"{path}:{number}: {exc}") from exc
    return rows


def _load(directory: Path) -> dict:
    report = _read(directory / "accuracy.json")
    try:
        if (report["schema_version"] != 1 or report["status"] != "complete"
            or report["scoring_complete"] is not True):
            raise ValueError("Requires a completed schema-version-1 accuracy run")
        for field in ("datasets", "settings", "counts", "runtime", "assets", "evaluator", "results"):
            if not isinstance(report[field], dict) or not report[field]:
                raise ValueError(f"Missing or empty {field}")
        for field in ("files", "chat_template"):
            if not report["assets"].get(field):
                raise ValueError(f"Missing asset identity: {field}")
        if not report["evaluator"].get("source_tree_sha256"):
            raise ValueError("Missing scorer source identity")
        if not report["runtime"].get("artifact_sha256") or not report["runtime"].get("runtime", {}).get("library_sha256"):
            raise ValueError("Missing artifact or runtime identity")
        if not isinstance(report["results"].get("results"), dict) or not report["results"]["results"]:
            raise ValueError("Missing aggregate metrics")
        answers = _rows(directory / "accuracy-answers.jsonl")
        samples = _rows(directory / "accuracy-samples.jsonl")
        expected = set()
        for task, dataset in report["datasets"].items():
            ids = dataset["selected_sample_ids"]
            keys = {(task, item["doc_id"]) for item in ids}
            if not ids or len(keys) != len(ids) or len(ids) != dataset["selected"]:
                raise ValueError(f"{task}: invalid selected sample IDs")
            expected.update(keys)
            count = report["counts"][task]
            if any(count[name] != len(ids) for name in ("selected", "attempted", "answered", "scored")):
                raise ValueError(f"{task}: incomplete sample counts")
            if any(count[name] != 0 for name in ("failed", "unsupported", "not_run")):
                raise ValueError(f"{task}: run contains failed or missing samples")
            for item in ids:
                row = answers[(task, item["doc_id"])]
                if row["sample_id"] != item["sample_id"] or row["status"] != "ok":
                    raise ValueError(f"{task}: answer does not match selected sample")
                for field in ("split", "generation", "images", "content", "answer"):
                    row[field]
                for field in ("raw_answer", "prompt_sha256", "input_tokens", "generated_tokens", "finish_reason"):
                    row["runtime"][field]
        if set(answers) != expected or set(samples) != expected:
            raise ValueError("Answer and scoring sample IDs must match the selected samples")
        for row in samples.values():
            for field in ("target", "filtered_resps", "doc_hash"):
                row[field]
        return {"report": report, "answers": answers, "samples": samples}
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise InvalidRun(f"{directory}: {exc}") from exc


def _task_config(value):
    if isinstance(value, dict):
        return {key: _task_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_task_config(item) for item in value]
    if isinstance(value, str):
        # dump_config serializes callables with process-specific addresses.
        return re.sub(r"^<function ([^<>]+) at 0x[0-9a-fA-F]+>$", r"<function \1>", value)
    return value


def _identity(report: dict) -> dict:
    datasets = copy.deepcopy(report["datasets"])
    for dataset in datasets.values():
        # Prepared dataset directories can differ between the host and DevKit.
        dataset.get("task_config", {}).pop("dataset_path", None)
        if "task_config" in dataset:
            dataset["task_config"] = _task_config(dataset["task_config"])
    return {
        "datasets": datasets,
        "settings": report["settings"],
        "runtime": report["runtime"],
        "assets": {key: report["assets"][key] for key in ("files", "chat_template")},
        "evaluator": {key: value for key, value in report["evaluator"].items()
                      if key not in ("packages", "python", "machine")},
    }


def _differences(left, right, path: str, output: list):
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(left.keys() | right.keys()):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                output.append({"path": child, "kind": "missing_right" if key in left else "missing_left"})
            else:
                _differences(left[key], right[key], child, output)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            output.append({"path": path, "kind": "length", "left": len(left), "right": len(right)})
        for index, (a, b) in enumerate(zip(left, right)):
            _differences(a, b, f"{path}/{index}", output)
    elif type(left) is not type(right) or left != right:
        output.append({"path": path, "kind": "value", "left": left, "right": right})


def compare_accuracy(left: Path, right: Path) -> dict:
    """Return exact differences. Row order does not affect sample matching."""
    runs = [_load(left), _load(right)]
    differences = []
    _differences(_identity(runs[0]["report"]), _identity(runs[1]["report"]), "identity", differences)
    _differences(runs[0]["report"]["counts"], runs[1]["report"]["counts"], "counts", differences)
    _differences(runs[0]["report"]["results"]["results"],
                 runs[1]["report"]["results"]["results"], "metrics", differences)
    for source in ("answers", "samples"):
        projected = []
        for run in runs:
            rows = {}
            for (task, doc_id), original in run[source].items():
                row = copy.deepcopy(original)
                if source == "answers":
                    row["runtime"] = {key: row["runtime"][key] for key in (
                        "raw_answer", "prompt_sha256", "input_tokens", "generated_tokens", "finish_reason")}
                else:
                    # Large source objects are represented by doc_hash and ordered answer inputs.
                    for field in ("doc", "arguments", "input_media", "token_counts"):
                        row.pop(field, None)
                rows.setdefault(task, {})[str(doc_id)] = row
            projected.append(rows)
        _differences(*projected, source, differences)
    return {"schema_version": 1, "status": "different" if differences else "match",
            "left": str(left), "right": str(right), "differences": differences}


def run_comparison(args) -> int:
    try:
        result = compare_accuracy(args.left, args.right)
    except (InvalidRun, KeyError, TypeError) as exc:
        result = {"schema_version": 1, "status": "invalid", "error": str(exc)}
    if args.output:
        try:
            # Do not overwrite either input export.
            if args.output.resolve() in {
                (directory / name).resolve() for directory in (args.left, args.right)
                for name in ("accuracy.json", "accuracy-answers.jsonl", "accuracy-samples.jsonl")
            }:
                raise ValueError("Comparison output must not overwrite an input export")
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        except (OSError, ValueError) as exc:
            print(f"Invalid comparison output: {exc}")
            return 2
    print(f"Accuracy comparison: {result['status']}")
    if result["status"] == "invalid":
        print(result["error"])
        return 2
    for difference in result["differences"][:20]:
        print(f"  {difference['path']}: {difference['kind']}")
    if len(result["differences"]) > 20:
        print(f"  {len(result['differences']) - 20} more differences. Use --output for the full report.")
    return 1 if result["differences"] else 0


def add_comparison_parser(subparsers):
    parser = subparsers.add_parser("compare-accuracy", help="Compare two completed VLM accuracy runs")
    parser.add_argument("left", type=Path, help="First accuracy result directory")
    parser.add_argument("right", type=Path, help="Second accuracy result directory")
    parser.add_argument("-o", "--output", type=Path, help="Comparison JSON file")
