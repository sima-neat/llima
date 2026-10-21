"""Hermetic task/scoring integration with the packaged image-only lmms-eval profile."""

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
    MultimodalClient, UnsupportedInput, file_identity, generation_settings, ordered_content,
)
from sima_lmm.mole.vlm_accuracy import AccuracyOptions, run_vlm_accuracy


class BenchmarkPeer:
    """Exercise the real wire client; this peer does not stand in for hardware validation."""

    def __init__(self, runtime, *, fail_on=None):
        self.runtime = runtime
        self.fail_on = fail_on
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
    assets = tmp_path / "assets"
    assets.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
        (assets / name).write_text("{}")
    files = {"devkit/" + file.name: file_identity(file) for file in assets.iterdir()}
    runtime = {"protocol_version": 1, "decoding": "greedy", "max_num_tokens": 1024, "files": files}
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
        model_id=str(assets), tasks=list(rows), accuracy=AccuracyOptions(manifest=manifest),
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
    assert report["datasets"]["docvqa_val"]["selected_sample_ids"] == [{"doc_id": 0, "sample_id": 123}]


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


def test_mismatched_assets_fail_before_inference_and_stop_owned_server(tmp_path, monkeypatch):
    config, peer, lifecycle = prepared_config(tmp_path, monkeypatch)
    try:
        (tmp_path / "assets/tokenizer.json").write_text('{"wrong":true}')
        with pytest.raises(UnsupportedInput, match="does not match deployed"):
            run_vlm_accuracy(config)
    finally:
        peer.close()
    report = json.loads((config.output_path / "accuracy.json").read_text())
    assert report["status"] == "failed"
    assert len(peer.requests) == 1  # capability query, no inference
    assert lifecycle == ["start", "stop"]
    assert sum(count["not_run"] for count in report["counts"].values()) == 4


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
