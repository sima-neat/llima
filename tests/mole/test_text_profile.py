"""Text profiling across ordinary and speculative ZMQ server versions."""

import struct
from types import MethodType, SimpleNamespace

import msgpack
import pytest
import torch

from sima_lmm.mole import perf_runner
from sima_lmm.mole.modalixlm import ModalixLM


@pytest.mark.parametrize("tagged", [False, True], ids=["legacy", "current"])
@pytest.mark.parametrize("accepted", [None, 0], ids=["ordinary", "speculative"])
@pytest.mark.parametrize("dataset", [None, "ifeval"], ids=["bucket", "dataset"])
@pytest.mark.parametrize("new_tokens", [1, 8])
def test_profile_measures_exact_budget(tmp_path, monkeypatch, tagged, accepted, dataset, new_tokens):
    tokens = [12, 34, 56, 78]
    budget = new_tokens + (dataset is None)
    requests = []

    def send(frames):
        request = msgpack.unpackb(frames[0], raw=False)
        assert request["type"] == "generate_for_perf"
        assert request["stop_token_ids"] == [2]
        assert list(struct.unpack("=4I", frames[1])) == tokens
        requests.append(request["max_num_tokens"])

    def reply(**kwargs):
        count = requests[-1] - len(tokens) + (not tagged and accepted is None)
        # The first real request stops early at EOS, independent of the token limit.
        if len(requests) == 1:
            count = min(1, count)
        metadata = {"tensor_dtype": "float64", "tensor_shape": [1, count],
                    "generated_tokens": count, "accepted_draft_tokens": accepted, "infer_time_ns": 1}
        if tagged:
            metadata["result_type"] = "token_durations"
        return [msgpack.packb(metadata), struct.pack("=d", 0.01) * count]

    lm = SimpleNamespace(pretrained=str(tmp_path), config=SimpleNamespace(eos_token_id=2),
                         tokenizer=object(), profile_data=[],
                         zmq_socket=SimpleNamespace(send_multipart=send, recv_multipart=reply))
    lm._zmq_infer = MethodType(ModalixLM._zmq_infer, lm)
    config = SimpleNamespace(performance=perf_runner.PerformanceOptions(dataset=dataset),
                             board=SimpleNamespace(tcp_uri="test", model="test"), do_start_server=False,
                             output_path=tmp_path, limit=1, random_seed=42, input_lengths=None,
                             max_new_tokens=new_tokens, max_num_tokens=len(tokens) + budget)
    monkeypatch.setattr(perf_runner, "load_workload",
                        lambda *_: ([{"id": "prompt", "category": "test", "input_ids": tokens}], {}))
    # The longer bucket cannot fit TTFT plus all requested decode steps.
    monkeypatch.setattr(perf_runner.performance_bench, "create_sample_data",
                        lambda **_: {len(tokens): [tokens], len(tokens) + 1: [tokens + [90]]})
    monkeypatch.setattr(perf_runner.performance_bench, "summarize", lambda **_: {})

    report = perf_runner.run_performance(lm, config)
    warmup, measured = report["requests"]
    legacy_ordinary = not tagged and accepted is None
    assert requests == [len(tokens) + new_tokens, len(tokens) + budget - legacy_ordinary]
    assert warmup["output_budget"] == new_tokens + legacy_ordinary
    assert warmup["generated_tokens"] == 1
    assert measured["output_budget"] == measured["generated_tokens"] == budget
    assert measured["decode_tokens"] == budget - 1
    assert measured["accepted_draft_tokens"] == accepted
    assert measured["output_budget_reached"]


@pytest.mark.parametrize(("shape", "payload"), [([1, 0], b""), ([1, 1], b""), ([1, 0], bytes(8))])
def test_profile_empty_tensor_requires_matching_shape(shape, payload):
    metadata = {"tensor_dtype": "float64", "tensor_shape": shape, "result_type": "token_durations"}
    socket = SimpleNamespace(send_multipart=lambda _: None,
                             recv_multipart=lambda **_: [msgpack.packb(metadata), payload])
    lm = SimpleNamespace(zmq_socket=socket, profile_data=[])
    args = lm, torch.tensor([[12, 34]]), "generate_for_perf"
    if shape == [1, 0] and not payload:
        durations = ModalixLM._zmq_infer(*args, max_num_tokens=2)
        assert list(durations.shape) == shape
        assert durations.dtype == torch.float64
        assert lm.last_profile_data["out_tokens"] == 0
    else:
        with pytest.raises((RuntimeError, ValueError)):
            ModalixLM._zmq_infer(*args, max_num_tokens=2)
