"""
A highly sophisticated, artisinal LLM  Stopwatch.
"""

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, List, Tuple, TypeAlias, Union

import datasets
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table
from tqdm import tqdm, trange
from transformers import AutoTokenizer, PreTrainedTokenizer

# {sequence_length: [[sample1_tok1, ...], [sample2_tok1, ...]]}
TokenizedSamples: TypeAlias = dict[int, list[list[int]]]

# {sequence_length: {'metric': [value1, value2, ...]}}
PerfStatsData: TypeAlias = dict[int, defaultdict[str, list[float]]]

# {sequence_length: {'metric': summary_value}}
SummaryStats: TypeAlias = dict[int, dict[str, float]]


def format_number(num: int | float | np.floating) -> str:
    """
    format_number Format a number in scientific notation or decimal notation.

    Args:
        num: An integer or float input.

    Returns:
        Formatted output.
    """
    if 0 < abs(num) < 1e-2 or (abs(num) >= 1e3):
        return f"{num:.2e}"
    else:
        return f"{num:.2f}"


def create_sample_data(
    tokenizer: PreTrainedTokenizer,
    max_toks: int = 1024,
    n_samples: int = 128,
    input_lengths: tuple[int, ...] | None = None,
) -> TokenizedSamples:
    """
    create_sample_data Creates a dataset token sample for LLM evaluation. Data is sampled from
    `cimec/lambada`. When available, the tokenizer's chat template wraps each passage in a
    continuation request; otherwise, raw text is used. Each sample is truncated to a given token
    length. This function is to be used when benchmarking LLM performance only. Please do not use
    this for benchmarking quality metrics.

    Args:
        tokenizer: The HF Tokenizer for the model.
        max_toks: The maximum context length to benchmark against.
            The function will generate buckets for powers of two up to this value.
            Defaults to 1024.
        n_samples: The number of unique text samples to generate
            for *each* context length bucket. Defaults to 128.
        input_lengths: Exact input-token lengths to generate. When omitted,
            powers-of-two buckets are generated from 128 up to max_toks.

    Returns:
        A dictionary where keys are integer context lengths and values are lists of tokenized
        samples.
    """
    if n_samples < 100:
        logger.warning(f"n_samples = {n_samples}. Do not use this run for benchmarking!")
    if input_lengths is None:
        # 128 -> max token len
        tok_lens = [2**x for x in range(7, math.floor(math.log2(max_toks)))]
    else:
        tok_lens = list(input_lengths)
    if not tok_lens:
        raise ValueError("No performance input lengths were generated")
    longest_input = max(tok_lens)
    required_samples = n_samples * len(tok_lens)

    chat_template = getattr(tokenizer, "chat_template", None)
    chat_prefix_len = 0
    chat_suffix_len = 0

    def tokenize_text(text: str) -> list[int]:
        if not chat_template:
            return tokenizer(text, padding=False, truncation=False)["input_ids"]
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Continue the following passage:\n\n{text}"}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )

    if chat_template:
        marker_a = tokenize_text("alpha")
        marker_b = tokenize_text("bravo")
        chat_prefix_len = next(
            (
                i for i, (token_a, token_b) in enumerate(zip(marker_a, marker_b))
                if token_a != token_b
            ),
            min(len(marker_a), len(marker_b)),
        )
        max_suffix_len = min(len(marker_a), len(marker_b)) - chat_prefix_len
        while (
            chat_suffix_len < max_suffix_len
            and marker_a[-chat_suffix_len - 1] == marker_b[-chat_suffix_len - 1]
        ):
            chat_suffix_len += 1
        if chat_suffix_len == 0:
            raise RuntimeError("The chat template has no assistant-generation suffix")

    def truncate_tokens(tokens: list[int], target_length: int) -> list[int]:
        if not chat_template:
            return tokens[:target_length]
        content_end = len(tokens) - chat_suffix_len
        content_length = target_length - chat_prefix_len - chat_suffix_len
        if content_length < 0 or content_end - chat_prefix_len < content_length:
            raise RuntimeError(
                f"Cannot create a {target_length}-token chat prompt with this template"
            )
        return (
            tokens[:chat_prefix_len]
            + tokens[chat_prefix_len:content_end][:content_length]
            + tokens[content_end:]
        )

    def tok_and_len(batch):
        tok = tokenize_text(batch["text"])
        return {"input_ids": tok, "len": len(tok)}

    tokenized_dataset = (
        datasets.load_dataset("cimec/lambada", split=f"train[:{required_samples}]")
        .map(tok_and_len, batched=False, remove_columns=["text", "domain"])
        .filter(lambda data: data["len"] >= longest_input)
    )
    if len(tokenized_dataset) < required_samples:
        raise RuntimeError(
            f"LAMBADA provided {len(tokenized_dataset)} samples with at least "
            f"{longest_input} tokens; {required_samples} are required"
        )

    samples = {
        tok_len: [
            truncate_tokens(tokenized_dataset[j]["input_ids"], tok_len)
            for j in range(i * n_samples, (i + 1) * n_samples)
        ]
        for i, tok_len in enumerate(sorted(tok_lens))
    }

    return samples


def save_samples(
    model_name: str,
    data: TokenizedSamples,
    save_dir: str | Path,
):
    """
    save_samples Save tokenized samples to a file in `save_dir`.
        ```
            save_dir/
                MODEL - plaintext model name
                tokz.npz - tokens
        ```
    Args:
        model_name: The name of the model that generated the samples.
        data: A dictionary where keys are input sequence lengths, and values are lists of token
            lists (input samples), the same length as the keys.
        save_dir: Save location. Will be created if it doesn't exist.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Saving {model_name} samples to {save_dir}")
    with (save_dir / "MODEL").open("w") as fp:
        fp.write(model_name)

    np.savez(file=save_dir / "toks.npz", **{str(k): v for k, v in data.items()})


def load_samples(path: str | Path) -> tuple[str, TokenizedSamples]:
    """
    load_samples Loads samples generated by `save_samples`.

    Args:
        path: Path containing the saved samples, or the tok.npz file.

    Returns:
        A tuple containing the model name, and tokenized sample data.
    """
    path = Path(path)

    if path.is_file():
        path = path.parent

    npzdata = np.load(path / "toks.npz")
    data = {int(k): npzdata[k] for k in npzdata.files}

    with (path / "MODEL").open("r") as fp:
        model_name = fp.read()
    logger.info(f"Loaded {model_name} samples from {path}.")
    return (model_name, data)


def perf_bench(data: TokenizedSamples, inference_fn: Callable[[Any], list[float]]) -> PerfStatsData:
    """
    perf_bench Benchmark LLM Performance.
      input      outtok1    outtok2     ...     outtokN
        |   ->      |          |                   |
        t0          t1         t2       ...        tN
        |   <->     |               <->            |
            ttft                tok_gen_time
          1 out tok             N - 1 out toks
    Args:
        data: Pre-truncated, tokenized samples
        inference_fn: LLM inference function - should accept input tokens and return a list of time
            to the next tokens. The first entry is the time to the first token (TTFT).

    Returns:
        Performance statistics, containing values for each individual input set.
    """
    perf_stats = {}

    # Perf Bench
    for seq_len, tok_list in data.items():
        stats = defaultdict(list)
        for toks in tqdm(tok_list, desc=f"seq_len: {seq_len}"):
            time_data = inference_fn(toks)

            assert time_data
            ttft = time_data[0]
            if len(time_data) > 1:
                tps = (len(time_data) - 1) / sum(time_data[1:])
            else:
                tps = 0
            stats["ttft"].append(ttft)
            stats["tps"].append(tps)

        perf_stats[seq_len] = stats
    return perf_stats


def summarize(data: PerfStatsData, model_name: str) -> SummaryStats:
    """
    summarize Summarizes performance data, prints a table, and returns a dictionary of summary
    stats.

    Args:
        data: Performance statistics, as returned by `perf_bench()`.
        model_name: Model name.

    Returns:
        Summarized statistics (mean, median, p5, p95) for each seq len.
    """
    res_dict = defaultdict(lambda: defaultdict(list))
    console = Console()
    table = Table(
        title=f"{model_name} Performance Results",
        show_header=True,
    )
    # Define the columns for the table
    table.add_column("Token Length", style="dim", width=12, justify="center")
    table.add_column("Metric", justify="center")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("P5", justify="right")
    table.add_column("P95", justify="right")

    # Sort by token length to ensure ordered output
    for length, metrics in sorted(data.items()):
        for metric_name, values in metrics.items():
            if not values:
                continue

            mean = np.mean(values)
            median = np.median(values)
            p5 = np.percentile(values, 5)
            p95 = np.percentile(values, 95)

            table.add_row(
                str(length),
                metric_name,
                format_number(mean),
                format_number(median),
                format_number(p5),
                format_number(p95),
            )
            res_dict[length][metric_name] = {"mean": mean, "median": median, "p95": p95, "p5": p5}
        if length != sorted(data.keys())[-1]:
            table.add_row()

    console.print(table)

    return res_dict
