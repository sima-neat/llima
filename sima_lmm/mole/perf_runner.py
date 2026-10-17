"""Dataset performance measurements and JSON results."""

import dataclasses
import importlib.metadata
import json
import math
import platform
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError
from loguru import logger

from sima_lmm.mole import performance_bench
from sima_lmm.mole.workloads import load_workload


@dataclasses.dataclass(frozen=True)
class PerformanceOptions:
    dataset: str | None = None
    model_revision: str | None = None

    def validate(self):
        if self.dataset not in (None, "bfcl-v3", "ifeval", "alpacaeval", "gsm8k", "dialogsum"):
            raise ValueError("Unsupported performance dataset")


def write_results(path: Path, report: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def _aggregate(records):
    groups = defaultdict(list)
    for record in records:
        if record.get("phase") == "measure" and record.get("status") == "ok":
            groups[(record["decoding"], "all")].append(record)
            if record["category"] != "all":
                groups[(record["decoding"], record["category"])].append(record)
    summaries = []
    for (mode, category), rows in groups.items():
        decode_tokens = sum(row["decode_tokens"] for row in rows)
        decode_time = sum(row["decode_time_s"] for row in rows)
        generated = sum(row["generated_tokens"] for row in rows)
        accepted = [row["accepted_draft_tokens"] for row in rows]
        summaries.append({
            "decoding": mode, "category": category, "requests": len(rows),
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "generated_tokens": generated, "decode_tokens": decode_tokens,
            "ttft_mean_s": statistics.mean(row["ttft_s"] for row in rows),
            "ttft_median_s": statistics.median(row["ttft_s"] for row in rows),
            "generation_time_s": sum(row["generation_time_s"] for row in rows),
            "decode_tps": decode_tokens / decode_time if decode_time else None,
            "accepted_draft_tokens": sum(accepted) if all(value is not None for value in accepted) else None,
            "draft_contribution": sum(accepted) / generated if generated and all(value is not None for value in accepted) else None,
        })
    return {"groups": summaries,
            "failures": sum(record.get("status") == "failed" for record in records)}


def run_performance(lm, config):
    options = config.performance
    report = {"schema_version": 1, "status": "running", "requests": [], "summary": {},
              "settings": {**dataclasses.asdict(options), "max_num_tokens": config.max_num_tokens,
                           "max_new_tokens": config.max_new_tokens, "num_samples": config.limit,
                           "sampling": "greedy", "seed": config.random_seed,
                           "warmup": 1, "repetitions": 1, "cache_policy": "reset",
                           "chat_template": "native defaults", "cpu_threads": 1},
              "evaluator": {"python": platform.python_version(), "machine": platform.machine(),
                            "packages": {name: importlib.metadata.version(name) for name in
                                         ("sima-lmm", "torch", "transformers", "datasets")}},
              "runtime": {"endpoint": config.board.tcp_uri, "model_selection": config.board.model,
                          "managed_server": config.do_start_server},
              "timing": {"ttft": "target prefill to first generated token; excludes tokenization and input-buffer preparation",
                         "decode_tps": "tokens after the first / sum of their durations",
                         "speculative_tokens": "verification-round duration divided equally across emitted tokens; not individual arrival timestamps",
                         "generation_time": "sum of TTFT and decoding durations; excludes network transport",
                         "cache": "empty KV cache before every warmup and measured request"}}
    output = config.output_path / "performance.json"
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        report["tokenizer"] = lm.pretrained
        mode = None
        stops = getattr(lm.config, "eos_token_id", None) or lm.tokenizer.eos_token_id
        generation_config = Path(lm.pretrained) / "generation_config.json"
        if not Path(lm.pretrained).is_dir():
            try:
                generation_config = Path(hf_hub_download(
                    lm.pretrained, "generation_config.json", revision=lm.revision
                ))
            except RemoteEntryNotFoundError:
                generation_config = None
        if generation_config is not None and generation_config.is_file():
            stops = json.loads(generation_config.read_text()).get("eos_token_id", stops)
        stops = [stops] if isinstance(stops, int) else list(stops or [])
        report["settings"]["stop_token_ids"] = stops
        if options.dataset:
            samples, report["dataset"] = load_workload(options.dataset, config.limit)
        else:
            data = performance_bench.create_sample_data(
                tokenizer=lm.tokenizer, max_toks=config.max_num_tokens,
                n_samples=config.limit, input_lengths=config.input_lengths,
            )
            if config.input_lengths is None:
                data = {length: rows for length, rows in data.items()
                        if length + config.max_new_tokens <= config.max_num_tokens}
            samples = [{"id": f"bucket-{length}-{index}", "category": f"n{length}", "input_ids": tokens}
                       for length, rows in data.items() for index, tokens in enumerate(rows)]
            report["dataset"] = {"name": "lambada-buckets", "repository": "cimec/lambada"}
        report["selected_sample_ids"] = [sample["id"] for sample in samples]
        prepared = []
        for sample in samples:
            try:
                if "input_ids" in sample:
                    tokens = sample["input_ids"]
                else:
                    if not lm.tokenizer.chat_template:
                        raise ValueError("Dataset workloads require the model's native chat template")
                    if sample.get("tools") and "tools" not in str(lm.tokenizer.chat_template):
                        raise ValueError("The selected tokenizer's chat template does not support tools")
                    tokens = lm.tokenizer.apply_chat_template(sample["messages"], tools=sample.get("tools"),
                        tokenize=True, add_generation_prompt=True, return_dict=False)
                # Existing buckets measure TTFT plus max_new_tokens decode steps; dataset budgets
                # count all generated tokens, including the first.
                budget = config.max_new_tokens + (0 if options.dataset else 1)
                context_tokens = len(tokens) + config.max_new_tokens
                if not tokens or context_tokens > config.max_num_tokens:
                    raise ValueError(f"Input {len(tokens)} + output {budget} exceeds context {config.max_num_tokens}; no truncation applied")
                prepared.append((sample, tokens, budget))
            except Exception as exc:
                report["requests"].append({"sample_id": sample["id"], "category": sample["category"],
                                           "status": "failed", "phase": "prepare", "error": str(exc)})
        if not prepared:
            raise ValueError("No valid performance requests were prepared")
        for phase in ("warmup", "measure"):
            selected = prepared[:1] if phase == "warmup" else prepared
            for sample, tokens, budget in selected:
                context = torch.tensor([tokens], dtype=torch.int64)
                record = {"sample_id": sample["id"], "category": sample["category"],
                          "source": sample.get("source"),
                          "phase": phase, "input_tokens": len(tokens),
                          "output_budget": budget, "status": "failed"}
                try:
                    # The existing ordinary endpoint returns the first token plus decode steps;
                    # speculative decoding counts all output tokens in max_num_tokens.
                    # The unmeasured warmup identifies the mode from its draft counter.
                    max_length = len(tokens) + (config.max_new_tokens if phase == "warmup"
                                               else budget - (mode == "ordinary"))
                    max_length = min(max_length, config.max_num_tokens)
                    durations = lm._zmq_infer(context, "generate_for_perf", max_num_tokens=max_length,
                        stop_token_ids=stops).cpu().tolist()[0]
                    metadata = lm.last_profile_data
                    count, accepted = metadata.get("generated_tokens"), metadata.get("accepted_draft_tokens")
                    if not durations or count != len(durations):
                        raise RuntimeError("Runtime must return one timing per generated token")
                    if any(not math.isfinite(duration) or duration <= 0 for duration in durations):
                        raise RuntimeError("Runtime returned invalid generation durations")
                    mode = "speculative" if accepted is not None else "ordinary"
                    report["settings"]["decoding"] = mode
                    record["decoding"] = mode
                    if phase == "warmup":
                        record["output_budget"] = max_length - len(tokens) + (mode == "ordinary")
                    if count > record["output_budget"]:
                        raise RuntimeError("Runtime returned more tokens than the requested output budget")
                    decode_time = sum(durations[1:])
                    record.update(status="ok", generated_tokens=count, token_durations_s=durations,
                        ttft_s=durations[0], decode_tokens=count - 1, decode_time_s=decode_time,
                        decode_tps=(count - 1) / decode_time if decode_time else None,
                        generation_time_s=sum(durations), accepted_draft_tokens=accepted,
                        draft_contribution=accepted / count if accepted is not None else None,
                        output_budget_reached=count == record["output_budget"],
                        server_time_s=metadata["infer_time_ns"] / 1e9,
                        roundtrip_time_s=metadata["zmq_roundtrip_time_ns"] / 1e9)
                    logger.info("{} {} {}: {} tokens, {:.1f} ms TTFT, {} tok/s", phase,
                                sample["id"], mode, count, durations[0] * 1000,
                                round(record["decode_tps"], 2) if record["decode_tps"] else None)
                except Exception as exc:
                    record.update(status="failed", error=str(exc))
                    logger.error("{} {}: {}", sample["id"], mode, exc)
                    # A timed-out REQ socket cannot safely service subsequent requests.
                    import zmq
                    if isinstance(exc, zmq.ZMQError):
                        report["requests"].append(record)
                        raise
                report["requests"].append(record)
                write_results(output, report)
        report["status"] = "failed" if any(row["status"] == "failed" for row in report["requests"]) else "complete"
        if report["status"] == "failed":
            raise RuntimeError("Performance requests failed; see performance.json")
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = str(exc) or type(exc).__name__
        raise
    finally:
        torch.set_num_threads(prior_threads)
        report["summary"] = _aggregate(report["requests"])
        write_results(output, report)
    if not options.dataset:
        buckets = defaultdict(lambda: defaultdict(list))
        for row in report["requests"]:
            if row.get("phase") != "measure" or row.get("status") != "ok":
                continue
            metrics = buckets[row["input_tokens"]]
            metrics["ttft"].append(row["ttft_s"])
            metrics["tps"].append(row["decode_tps"] or 0)
            if row["accepted_draft_tokens"] is not None:
                metrics["generated_tokens"].append(row["generated_tokens"])
                metrics["accepted_draft_tokens"].append(row["accepted_draft_tokens"])
                metrics["acceptance_rate_pct"].append(100 * row["draft_contribution"])
        report["summary"]["buckets"] = performance_bench.summarize(
            data=buckets, model_name=lm.pretrained
        )
        write_results(output, report)
    for group in report["summary"]["groups"]:
        if group["category"] == "all":
            logger.info("{}: {} requests, {:.2f} decode tok/s", group["decoding"], group["requests"], group["decode_tps"] or 0)
    return report
