"""Exact comparison and input validation for saved VLM runs."""

import json
from types import SimpleNamespace

import pytest

from sima_lmm.mole.accuracy_compare import InvalidRun, compare_accuracy, run_comparison


def make_run(path):
    path.mkdir()
    report = {
        "schema_version": 1, "status": "complete", "scoring_complete": True,
        "datasets": {"chartqa": {"selected": 1, "revision": "a" * 40,
            "split": "test", "selected_sample_ids": [{"doc_id": 0, "sample_id": "q1"}],
            "task_config": {"dataset_path": "/host/data", "process_results": "<function scorer at 0xabc>"}}},
        "settings": {"seed": 42},
        "counts": {"chartqa": {"selected": 1, "attempted": 1, "answered": 1,
            "scored": 1, "failed": 0, "unsupported": 0, "not_run": 0, "truncated": 0}},
        "runtime": {"artifact_sha256": "artifact", "runtime": {"library_sha256": "runtime"}},
        "assets": {"files": {"tokenizer.json": "hash"}, "chat_template": "native"},
        "evaluator": {"name": "lmms-eval", "version": "0.7.3", "python": "3.12", "machine": "x86"},
        "results": {"results": {"chartqa": {"relaxed_overall,none": 1.0}}},
    }
    answer = {"task": "chartqa", "doc_id": 0, "sample_id": "q1", "status": "ok",
        "split": "test", "generation": {"max_new_tokens": 16}, "images": [{"sha256": "image"}],
        "content": [{"type": "image"}, {"type": "text", "text": "Question?"}], "answer": "10",
        "runtime": {"raw_answer": "10", "prompt_sha256": "prompt", "input_tokens": 40,
            "generated_tokens": 1, "finish_reason": "stop", "infer_time_ns": 10}}
    sample = {"task": "chartqa", "doc_id": 0, "target": "10", "filtered_resps": ["10"],
              "doc_hash": "doc", "relaxed_overall": 1.0}
    save(path, report, answer, sample)
    return report, answer, sample


def save(path, report, answer, sample):
    (path / "accuracy.json").write_text(json.dumps(report))
    for name, row in (("answers", answer), ("samples", sample)):
        (path / f"accuracy-{name}.jsonl").write_text(json.dumps(row) + "\n")


def test_host_local_match(tmp_path):
    left, right = tmp_path / "host", tmp_path / "local"
    make_run(left)
    report, answer, sample = make_run(right)
    report["datasets"]["chartqa"]["task_config"]["dataset_path"] = "/devkit/data"
    report["datasets"]["chartqa"]["task_config"]["process_results"] = "<function scorer at 0xdef>"
    report["execution"] = {"mode": "devkit-local"}
    report["evaluator"].update(python="3.13", machine="arm64", packages={"extra": "1"})
    report["assets"]["model_id"] = "/devkit/model"
    answer["runtime"]["infer_time_ns"] = 99
    save(right, report, answer, sample)
    assert compare_accuracy(left, right)["status"] == "match"


def test_reports_identity_answer_and_metric_differences(tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    make_run(left)
    report, answer, sample = make_run(right)
    report["runtime"]["runtime"]["library_sha256"] = "changed"
    report["results"]["results"]["chartqa"]["relaxed_overall,none"] = 0.0
    answer["answer"] = "20"
    answer["runtime"]["prompt_sha256"] = "changed"
    sample["filtered_resps"] = ["20"]
    save(right, report, answer, sample)
    result = compare_accuracy(left, right)
    assert result["status"] == "different"
    paths = {item["path"] for item in result["differences"]}
    assert paths == {"identity/runtime/runtime/library_sha256", "metrics/chartqa/relaxed_overall,none",
                     "answers/chartqa/0/answer", "answers/chartqa/0/runtime/prompt_sha256",
                     "samples/chartqa/0/filtered_resps/0"}


@pytest.mark.parametrize("damage", ["failed", "missing", "duplicate", "count", "json", "schema"])
def test_invalid_runs(tmp_path, damage):
    left, right = tmp_path / "a", tmp_path / "b"
    make_run(left)
    report, answer, sample = make_run(right)
    if damage == "failed":
        report["status"] = "failed"
    elif damage == "schema":
        report["schema_version"] = 2
    elif damage == "count":
        report["counts"]["chartqa"]["scored"] = 0
    save(right, report, answer, sample)
    if damage == "missing":
        (right / "accuracy-samples.jsonl").write_text("")
    elif damage == "duplicate":
        (right / "accuracy-answers.jsonl").write_text((json.dumps(answer) + "\n") * 2)
    elif damage == "json":
        (right / "accuracy.json").write_text("{")
    with pytest.raises(InvalidRun):
        compare_accuracy(left, right)


def test_cli_exit_codes_and_output_protection(tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    make_run(left)
    report, answer, sample = make_run(right)
    args = SimpleNamespace(left=left, right=right, output=tmp_path / "comparison.json")
    assert run_comparison(args) == 0
    assert json.loads(args.output.read_text())["status"] == "match"
    answer["answer"] = "20"
    save(right, report, answer, sample)
    assert run_comparison(args) == 1
    args.output = left / "accuracy.json"
    original = args.output.read_bytes()
    assert run_comparison(args) == 2
    assert args.output.read_bytes() == original
    args.output = None
    (right / "accuracy.json").unlink()
    assert run_comparison(args) == 2


def test_cli_dispatch(tmp_path, monkeypatch):
    from sima_lmm.host.benchmark import main

    left, right = tmp_path / "a", tmp_path / "b"
    make_run(left)
    make_run(right)
    monkeypatch.setattr("sys.argv", ["llima-benchmark", "compare-accuracy", str(left), str(right)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


@pytest.mark.parametrize("section,field", [
    ("runtime", "artifact_sha256"), ("evaluator", "version"),
    ("assets", "files"), ("results", "results"),
])
def test_missing_identity_is_invalid(tmp_path, section, field):
    left, right = tmp_path / "a", tmp_path / "b"
    make_run(left)
    report, answer, sample = make_run(right)
    del report[section][field]
    save(right, report, answer, sample)
    with pytest.raises(InvalidRun):
        compare_accuracy(left, right)


def test_jsonl_order_does_not_change_comparison(tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    for path in (left, right):
        report, answer, sample = make_run(path)
        report["datasets"]["chartqa"].update(selected=2, selected_sample_ids=[
            {"doc_id": 0, "sample_id": "q1"}, {"doc_id": 1, "sample_id": "q2"}])
        for field in ("selected", "attempted", "answered", "scored"):
            report["counts"]["chartqa"][field] = 2
        save(path, report, answer, sample)
        answers = [answer, {**answer, "doc_id": 1, "sample_id": "q2"}]
        samples = [sample, {**sample, "doc_id": 1}]
        for name, rows in (("answers", answers), ("samples", samples)):
            if path == right:
                rows.reverse()
            (path / f"accuracy-{name}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
    assert compare_accuracy(left, right)["status"] == "match"
