# MoLE - Modalix Language Model Evaluator

## Overview

MoLE (Modalix Language Model Evaluator) is a benchmarking tool for evaluating the accuracy and performance of LLMs running on the Modalix platform.

It extends [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and supports two backends:

- **hf** — runs evaluation on the host using HuggingFace transformers (baseline reference)
- **modalix** — runs evaluation on a Modalix board via the `llima benchmark-server`

## Installation

MoLE is a host-side benchmarking tool. Install and run it on the host or SDK container, not on the Modalix device. The Modalix device only needs the LLiMa runtime and the `llima benchmark-server` process. See [Neat Framework installation](/getting-started/neat-library/) for the runtime installation flow.

Install MoLE on the host using `sima-cli`:

``` console
host:~$ sima-cli install tools/mole
```

This installs MoLE into a host virtual environment at `~/sima-mole-venv`.

## Usage

First, activate the MoLE virtual environment:

``` console
host:~$ source ~/sima-mole-venv/bin/activate
```

MoLE is then invoked via the `llima-benchmark` CLI with two subcommands. The `<model_id>` argument is always the HuggingFace model ID (e.g., `meta-llama/Llama-3.2-3B-Instruct`). In `-b modalix` mode this is not just a display label: it must match the tokenizer and config used to compile the deployed board model, because the board returns token scores only and does not provide tokenizer metadata.

### Accuracy Benchmarking

Evaluates model quality against standard tasks:

``` console
(sima-mole-venv) host:~$ llima-benchmark accuracy <model_id> -b modalix \
    -t <task> \
    -o <output_dir> \
    --max_num_tokens <max_num_tokens> \
    --board_ip <board_ip> \
    --board_model <model_path_on_board>
```

| Argument | Description |
|----|----|
| `model_id` | HuggingFace model ID (e.g., `meta-llama/Llama-3.2-3B-Instruct`). For `-b modalix`, this must match the deployed model's tokenizer/config. |
| `-b` | Backend to use: `modalix` (run on board) or `hf` (run on host as reference baseline). |
| `-t` | **Required.** One or more evaluation tasks. Example tasks: `hellaswag`, `triviaqa`, `piqa`, `winogrande`, `wikitext`. See the [task list](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/tasks/README.md) for all available tasks. |
| `-o` | Output directory for benchmark results. |
| `--board_ip` | IP address of the Modalix board. Required for `-b modalix`. |
| `--board_model` | Path to the compiled model directory on the Modalix device (e.g., `/media/nvme/llima/models/Llama-3.2-3B-Instruct-a16w4`). Required for `-b modalix`. |
| `--max_num_tokens` | Maximum context length. Must be equal to or smaller than the value used during compilation. |
| `--modalix_group_prefill` | Optional Modalix loglikelihood mode that uses grouped prefix prefill before scoring continuation tokens. This can improve multiple-choice benchmarks with long shared contexts, but the default path remains the per-token scalar scorer. |
| `-n, --num_samples` | Number of samples to evaluate. Runs the full task set if not specified. |
| `--board_ssh_user` | SSH username for the Modalix board. Optional, default: `sima`. \# |
| `--board_ssh_pass` | SSH password for the Modalix board. Optional. Set to enable non-interactive automated benchmarking. |

:::important
Accuracy and loglikelihood benchmarking with `-b modalix` requires the deployed model to be compiled with `--return_logits`. This flag is off by default. See [Model Compilation](compilation_genai.md). If the model was compiled without this flag, the benchmark fails with: `model not compiled with --return_logits; accuracy/loglikelihood tasks are unsupported`.
:::

In `-b modalix` mode, result tables are labeled as Modalix backend results and include the board target. The HuggingFace `model_id` still appears because MoLE uses it for tokenization and task metadata.

For models compiled with grouped prefill layers, `--modalix_group_prefill` can reuse those layers for the prompt prefix of each loglikelihood request, then score the continuation tokens exactly as scalar log probabilities on the board. This is opt-in because very short prefixes may not benefit from grouped execution.

To use the HuggingFace backend as a reference baseline:

``` console
(sima-mole-venv) host:~$ llima-benchmark accuracy <model_id> -b hf -t <task> -o <output_dir>
```

For all available options, run `llima-benchmark accuracy -h`.

### Performance Benchmarking

Measures Time To First Token (TTFT) and Tokens Per Second (TPS) on a Modalix board for different input lengths:

``` console
(sima-mole-venv) host:~$ llima-benchmark perf <model_id> \
    -o <output_dir> \
    --board_ip <board_ip> \
    --board_model <model_path_on_board> \
    --max_num_tokens <max_num_tokens> --max_new_tokens <max_new_tokens>
```

| Argument | Description |
|----|----|
| `model_id` | HuggingFace model ID (e.g., `meta-llama/Llama-3.2-3B-Instruct`). For Modalix performance runs, this should match the tokenizer/config for the deployed model. |
| `-o` | Output directory for benchmark results. |
| `--board_ip` | IP address of the Modalix board. |
| `--board_model` | Path to the compiled model directory on the Modalix device (e.g., `/media/nvme/llima/models/Llama-3.2-3B-Instruct-a16w4`). |
| `--max_num_tokens` | Maximum context length. Must be equal to or smaller than the value used during compilation. |
| `--max_new_tokens` | Maximum number of tokens to generate in the output. |
| `--board_ssh_user` | SSH username for the Modalix board. Optional, default: `sima`. |
| `--board_ssh_pass` | SSH password for the Modalix board. Optional. Set to enable non-interactive automated benchmarking. |

For all available options, run `llima-benchmark perf -h`.
