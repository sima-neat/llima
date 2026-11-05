"""Hermetic task/scoring integration with the installed lmms-eval package."""

import base64
import io
import json
import threading
from types import SimpleNamespace

import datasets
import msgpack
import pytest
import zmq
from PIL import Image

from sima_lmm.mole import modalixboard
from sima_lmm.mole.modalixboard import ModalixBoard
from sima_lmm.mole.multimodal import (
    MultimodalClient, UnsupportedInput, generation_settings, ordered_content,
)
from sima_lmm.mole.vlm_accuracy import AccuracyOptions, run_vlm_accuracy


class BenchmarkPeer:
    """Exercise the real wire client; this peer does not stand in for hardware validation."""

    def __init__(self, runtime, *, fail_on=None):
        self.runtime = runtime
        self.fail_on = fail_on
        self.profiles = {}
        self.requests = []
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.linger = 0
        self.socket.rcvtimeo = 100
        self.socket.curve_secretkey = b"+ZyM3eTG#JY(P)U75QLd<jK&f>ZL0/4XSO.M(+]8"
        self.socket.curve_server = True
        self.port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.serve)
        self.thread.start()

    def serve(self):
        try:
            while not self.done.is_set():
                try:
                    frames = self.socket.recv_multipart()
                except zmq.Again:
                    continue
                metadata = msgpack.unpackb(frames[0], raw=False)
                self.requests.append((metadata, frames[1]))
                request_type = metadata["type"]
                reply = {"tensor_dtype": "utf8", "result_type": request_type}
                if request_type == "multimodal_info":
                    payload = json.dumps(self.runtime).encode()
                else:
                    request = json.loads(frames[1])
                    content = request["messages"][0]["content"]
                    assert content[0]["type"] == "image"
                    encoded = content[0]["image"].removeprefix("data:image/png;base64,")
                    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
                    assert image.size == (12, 8)
                    text = "".join(part.get("text", "") for part in content)
                    if "Scene" in text:
                        answer = "A"
                    elif "Document" in text:
                        answer = "invoice"
                    elif "Chart10" in text:
                        answer = "10"
                    else:
                        answer = "20"
                    payload = answer.encode()
                    reply.update(finish_reason="length" if "Chart20" in text else "stop",
                                 input_tokens=40, generated_tokens=1, infer_time_ns=10,
                                 prompt_sha256="a" * 64, raw_answer=answer)
                    if request_type == "generate_multimodal_for_perf":
                        profile = next((value for key, value in self.profiles.items() if key in text), {
                            "token_durations": [0.2, 0.02, 0.03], "input_tokens": 40,
                            "generated_tokens": 3, "accepted_draft_tokens": None,
                        })
                        payload = json.dumps(profile).encode()
                        reply.update({key: profile[key] for key in
                                      ("input_tokens", "generated_tokens", "accepted_draft_tokens")})
                        reply.pop("raw_answer")
                    if self.fail_on and self.fail_on in text:
                        reply["error"] = "deliberate inference failure"
                reply["tensor_shape"] = [len(payload)]
                self.socket.send_multipart([msgpack.packb(reply, use_bin_type=True), payload])
        finally:
            self.socket.close()

    def close(self):
        self.done.set()
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()
        self.ctx.term()


def prepared_config(tmp_path, monkeypatch, *, local=False, fail_on=None):
    # Any unexpected download fails this test, including task/metric downloads.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(modalixboard, "_is_modalix", lambda: local)
    runtime = {"protocol_version": 1, "decoding": "greedy", "max_num_tokens": 1024,
               "model_path": "/models/qwen3.5-2b", "model_name": "qwen3.5-2b",
               "model_type": "qwen3_5", "runtime_version": "0.6.0"}
    peer = BenchmarkPeer(runtime, fail_on=fail_on)
    image = Image.new("RGB", (12, 8), "red")
    rows = {
        "realworldqa": ("test", [{"image": image, "question": "Scene: choose A or B", "answer": "A"}]),
        "docvqa_val": ("validation", [{"image": image, "question": "Document: type?", "answers": ["invoice"], "questionId": 123}]),
        "chartqa": ("test", [
            {"image": image, "question": "Chart10?", "answer": "10", "type": "human_test"},
            {"image": image, "question": "Chart20?", "answer": "20", "type": "augmented_test"},
        ]),
    }
    declarations = {}
    for task, (split, samples) in rows.items():
        path = tmp_path / task
        datasets.DatasetDict({split: datasets.Dataset.from_list(samples)}).save_to_disk(path)
        declarations[task] = {"revision": "a" * 40, "local_path": task, "split": split}
    manifest = tmp_path / "accuracy-manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "tasks": declarations}))
    output = tmp_path / "results"
    output.mkdir()
    board = ModalixBoard(address=None if local else "127.0.0.1", port=peer.port, model="qwen3.5-2b")
    lifecycle = []
    monkeypatch.setattr(board, "start_server", lambda **kwargs: lifecycle.append("start"))
    monkeypatch.setattr(board, "stop_server", lambda: lifecycle.append("stop"))
    return SimpleNamespace(
        model_id="qwen3.5-2b", tasks=list(rows), accuracy=AccuracyOptions(manifest=manifest),
        max_num_tokens=1024, limit=None, random_seed=42, board=board,
        output_path=output, do_start_server=True,
    ), peer, lifecycle


@pytest.mark.parametrize("local", [False, True], ids=["host", "devkit-local"])
def test_upstream_tasks_score_and_export_in_both_modes(tmp_path, monkeypatch, local):
    config, peer, lifecycle = prepared_config(tmp_path, monkeypatch, local=local)
    try:
        report = run_vlm_accuracy(config)
    finally:
        peer.close()
    assert report["status"] == "complete"
    assert report["execution"]["mode"] == ("devkit-local" if local else "host")
    assert lifecycle == ["start", "stop"]
    results = report["results"]["results"]
    assert results["realworldqa"]["exact_match,none"] == 1.0
    assert results["docvqa_val"]["anls,none"] == 1.0
    assert results["chartqa"]["relaxed_overall,none"] == 1.0
    assert report["counts"]["chartqa"]["truncated"] == 1
    assert all(count["failed"] == count["unsupported"] == count["not_run"] == 0
               for count in report["counts"].values())
    answers = [json.loads(line) for line in (config.output_path / "accuracy-answers.jsonl").read_text().splitlines()]
    scored = [json.loads(line) for line in (config.output_path / "accuracy-samples.jsonl").read_text().splitlines()]
    assert len(answers) == len(scored) == 4
    assert all("filtered_resps" in row for row in scored)
    assert all(row["images"][0]["width"] == 12 for row in answers)
    assert all(row["runtime"]["raw_answer"] == row["answer"] for row in answers)
    assert sorted(metadata["max_new_tokens"] for metadata, _ in peer.requests[1:]) == [16, 16, 16, 32]
    assert next(row for row in answers if row["task"] == "docvqa_val")["sample_id"] == 123
    assert report["runtime"] == peer.runtime


def test_failed_sample_is_reported_without_changing_evaluated_set(tmp_path, monkeypatch):
    config, peer, lifecycle = prepared_config(tmp_path, monkeypatch, fail_on="Document")
    try:
        with pytest.raises(RuntimeError, match="deliberate inference failure"):
            run_vlm_accuracy(config)
    finally:
        peer.close()
    report = json.loads((config.output_path / "accuracy.json").read_text())
    assert report["status"] == "failed"
    assert report["results"] == {}
    assert report["counts"]["docvqa_val"]["failed"] == 1
    assert lifecycle == ["start", "stop"]
    assert (config.output_path / "accuracy-samples.jsonl").read_text() == ""


def test_multi_image_order_and_lossless_transport():
    first, second = Image.new("RGB", (3, 2), "red"), Image.new("RGB", (3, 2), "blue")
    content, identities = ordered_content("Before <image 2> between <image 1> after", [first, second])
    assert [part["type"] for part in content] == ["text", "image", "text", "image", "text"]
    assert content[0]["text"] == "Before "
    decoded = Image.open(io.BytesIO(base64.b64decode(content[1]["image"].split(",", 1)[1])))
    assert decoded.getpixel((0, 0)) == (0, 0, 255)
    assert identities[0]["sha256"] != identities[1]["sha256"]


def test_teardown_failure_marks_report_failed(tmp_path, monkeypatch):
    config, peer, _ = prepared_config(tmp_path, monkeypatch)

    def fail_stop():
        raise RuntimeError("server cleanup failed")

    monkeypatch.setattr(config.board, "stop_server", fail_stop)
    try:
        with pytest.raises(RuntimeError, match="server cleanup failed"):
            run_vlm_accuracy(config)
    finally:
        peer.close()
    report = json.loads((config.output_path / "accuracy.json").read_text())
    assert report["status"] == "failed"
    assert report["cleanup_error"] == "server cleanup failed"
    assert report["results"] == {}
    assert sum(count["scored"] for count in report["counts"].values()) == 4


@pytest.mark.parametrize("context,visuals", [
    ("<image>", []), ("<image><image>", [Image.new("RGB", (2, 2))]),
    ("<image 2>", [Image.new("RGB", (2, 2))]), ("hello", ["video.mp4"]),
    ("hello", [Image.new("RGB", (2, 2)), None]),
])
def test_unsupported_visuals_are_rejected(context, visuals):
    with pytest.raises(UnsupportedInput):
        ordered_content(context, visuals)


@pytest.mark.parametrize("kwargs", [{"do_sample": True}, {"temperature": 0.3}, {"top_p": 0.9},
                                   {"num_beams": 2}, {"max_new_tokens": 0}, {"unknown": True}])
def test_unsupported_generation_is_rejected(kwargs):
    with pytest.raises(UnsupportedInput):
        generation_settings(kwargs)


def test_text_tasks_stay_on_original_evaluator(tmp_path):
    from sima_lmm.mole.mole import prepare_mole

    config = prepare_mole(str(tmp_path), "hf", tasks=["lambada_openai"], output_path=tmp_path / "out")
    assert config.evaluator == "lm-eval"


def test_runtime_context_limit_is_rejected():
    peer = BenchmarkPeer({"protocol_version": 1, "decoding": "greedy", "max_num_tokens": 32})
    board = ModalixBoard(address="127.0.0.1", port=peer.port)
    client = MultimodalClient(board, 64)
    try:
        with pytest.raises(UnsupportedInput, match="exceeds compiled"):
            client.info()
    finally:
        client.close()
        peer.close()


def test_real_task_manager_routes_text_and_images(tmp_path, monkeypatch):
    from lm_eval.tasks import TaskManager
    from sima_lmm.mole.mole import prepare_mole
    from sima_lmm.mole.vlm_accuracy import choose_evaluator

    manager = TaskManager()
    assert choose_evaluator(["gsm8k"], manager, "modalix") == "lm-eval"
    assert choose_evaluator(["chartqa"], manager, "modalix") == "lmms-eval"
    with pytest.raises(UnsupportedInput, match="separate"):
        choose_evaluator(["gsm8k", "chartqa"], manager, "modalix")

    def unexpected_lookup(*args, **kwargs):
        raise AssertionError("Image accuracy must use deployed assets without a host model lookup")

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", unexpected_lookup)
    config = prepare_mole(
        "deployment-label", "modalix", tasks=["chartqa"], output_path=tmp_path,
        board=ModalixBoard(address="127.0.0.1", port=8002, model="compiled-vlm"),
    )
    assert config.evaluator == "lmms-eval"


@pytest.mark.parametrize("accepted", [None, 0, 2])
def test_image_profile_preserves_canonical_timings_and_draft_counters(tmp_path, monkeypatch, accepted):
    config, peer, _ = prepared_config(tmp_path, monkeypatch)
    peer.runtime["speculative_decoding"] = accepted is not None
    peer.profiles["Document"] = {
        "token_durations": [0.2, 0.02, 0.03], "input_tokens": 40,
        "generated_tokens": 3, "accepted_draft_tokens": accepted,
    }
    client = MultimodalClient(config.board, 1024)
    try:
        assert client.info(allow_speculative=True)["speculative_decoding"] == (accepted is not None)
        content, _ = ordered_content("Document", [Image.new("RGB", (12, 8))])
        profile = client.profile([{"role": "user", "content": content}], {"max_new_tokens": 8})
    finally:
        client.close()
        peer.close()
    assert profile["token_durations"] == [0.2, 0.02, 0.03]
    assert profile["generated_tokens"] == 3
    assert profile["accepted_draft_tokens"] == accepted
    assert profile["zmq_roundtrip_time_ns"] > 0
    assert profile["infer_time_ns"] == 10
    assert peer.requests[-1][0]["max_new_tokens"] == 8


def test_speculative_image_accuracy_still_rejected(tmp_path, monkeypatch):
    config, peer, lifecycle = prepared_config(tmp_path, monkeypatch)
    peer.runtime["speculative_decoding"] = True
    try:
        with pytest.raises(UnsupportedInput, match="non-speculative"):
            run_vlm_accuracy(config)
    finally:
        peer.close()
    assert len(peer.requests) == 1
    assert lifecycle == ["start", "stop"]


def test_image_performance_workload_uses_upstream_prompts_without_scoring(tmp_path, monkeypatch):
    from sima_lmm.mole.perf_runner import PerformanceOptions
    from sima_lmm.mole.workloads import load_image_workload, prepare_image_sample

    config, peer, _ = prepared_config(tmp_path, monkeypatch)
    manifest = config.accuracy.manifest
    data = json.loads(manifest.read_text())
    data["tasks"] = {"chartqa": data["tasks"]["chartqa"]}
    manifest.write_text(json.dumps(data))
    config.performance = PerformanceOptions(dataset="chartqa", manifest=manifest)
    monkeypatch.setattr("lmms_eval.evaluator.simple_evaluate", lambda **kwargs: pytest.fail("Performance must not score"))
    try:
        samples, metadata = load_image_workload(config)
        rows = list(samples)
        assert len(rows) == metadata["available_samples"] == 2
        assert metadata["selection"] == "all"
        messages, images = prepare_image_sample(rows[0])
    finally:
        peer.close()
    assert rows[0]["id"] == "chartqa:0"
    assert messages[0]["content"][0]["type"] == "image"
    assert "Chart10" in messages[0]["content"][-1]["text"]
    assert images[0]["width"] == 12


@pytest.mark.parametrize("accepted", [None, 0, 1])
def test_image_performance_exports_weighted_metrics_in_both_modes(tmp_path, monkeypatch, accepted):
    from sima_lmm.mole.mole import perf_bench
    from sima_lmm.mole.perf_runner import PerformanceOptions

    config, peer, lifecycle = prepared_config(tmp_path, monkeypatch)
    manifest = config.accuracy.manifest
    data = json.loads(manifest.read_text())
    data["tasks"] = {"chartqa": data["tasks"]["chartqa"]}
    manifest.write_text(json.dumps(data))
    config.performance = PerformanceOptions(dataset="chartqa", manifest=manifest)
    config.max_new_tokens, config.input_lengths = 8, None
    config.do_perf, config.backend, config.batch_size, config.device = True, "modalix", 1, "cpu"
    config.tasks = []
    peer.runtime["speculative_decoding"] = accepted is not None
    peer.profiles = {
        "Chart10": {"token_durations": [0.2, 0.02, 0.03], "input_tokens": 40,
                    "generated_tokens": 3, "accepted_draft_tokens": accepted},
        "Chart20": {"token_durations": [0.3, 0.04, 0.06, 0.1], "input_tokens": 40,
                    "generated_tokens": 4, "accepted_draft_tokens": accepted},
    }

    def unexpected_host_model(*args, **kwargs):
        pytest.fail("Image performance must use the deployed VLM without a host model or scoring")

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", unexpected_host_model)
    monkeypatch.setattr("sima_lmm.mole.mole.ModalixLM", unexpected_host_model)
    monkeypatch.setattr("lmms_eval.evaluator.simple_evaluate", unexpected_host_model)
    try:
        perf_bench(config)
    finally:
        peer.close()
    report_text = (config.output_path / "performance.json").read_text()
    report = json.loads(report_text)
    measured = [row for row in report["requests"] if row["phase"] == "measure"]
    assert len(measured) == 2
    assert len([row for row in report["requests"] if row["phase"] == "warmup"]) == 1
    assert [row["ttft_s"] for row in measured] == [0.2, 0.3]
    assert [row["generated_tokens"] for row in measured] == [3, 4]
    group = next(row for row in report["summary"]["groups"] if row["category"] == "all")
    assert group["requests"] == 2
    assert group["decoding"] == ("ordinary" if accepted is None else "speculative")
    assert group["decode_tps"] == pytest.approx(20.0)
    assert group["accepted_draft_tokens"] == (None if accepted is None else 2 * accepted)
    assert group["draft_contribution"] == (None if accepted is None else pytest.approx(2 * accepted / 7))
    assert "base64" not in report_text
    assert lifecycle == ["start", "stop"]
    requests = [metadata for metadata, _ in peer.requests if metadata["type"] == "generate_multimodal_for_perf"]
    assert len(requests) == 3
    assert all(metadata["max_new_tokens"] == 8 for metadata in requests)
