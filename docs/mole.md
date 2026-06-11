# MoLE - Modalix Language Model Evaluator

## Overview

MoLE (Modalix Language Model Evaluator) is a benchmarking tool for evaluating the accuracy and performance of LLMs running on the Modalix platform.

It extends [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and supports two backends:

- **hf** — runs evaluation on the host using HuggingFace transformers (baseline reference)
- **modalix** — runs evaluation on a Modalix board via the `llima benchmark-server`

## Installation

MoLE requires the LLiMa runtime on the Modalix device. See [Neat Framework installation](/getting-started/installation/neat-framework/) for the runtime installation flow.

Install MoLE on your Modalix device using the `sima-cli`:

``` console
host:~$ sima-cli install tools/mole
```

This installs MoLE into a virtual environment at `~/sima-mole-venv`.

## Usage

First, activate the MoLE virtual environment:

``` console
host:~$ source ~/sima-mole-venv/bin/activate
```

MoLE is then invoked via the `llima-benchmark` CLI with two subcommands. The `<model_id>` argument is always the HuggingFace model ID (e.g., `meta-llama/Llama-3.2-3B-Instruct`).

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
| `model_id` | HuggingFace model ID (e.g., `meta-llama/Llama-3.2-3B-Instruct`). |
| `-b` | Backend to use: `modalix` (run on board) or `hf` (run on host as reference baseline). |
| `-t` | **Required.** One or more evaluation tasks. Example tasks: `hellaswag`, `triviaqa`, `piqa`, `winogrande`, `wikitext`. See the [task list](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/tasks/README.md) for all available tasks. |
| `-o` | Output directory for benchmark results. |
| `--board_ip` | IP address of the Modalix board. Required for `-b modalix`. |
| `--board_model` | Path to the compiled model directory on the Modalix device (e.g., `/media/nvme/llima/models/Llama-3.2-3B-Instruct-a16w4`). Required for `-b modalix`. |
| `--max_num_tokens` | Maximum context length. Must be equal to or smaller than the value used during compilation. |
| `-n, --num_samples` | Number of samples to evaluate. Runs the full task set if not specified. |
| `--board_ssh_user` | SSH username for the Modalix board. Optional, default: `sima`. \# |
| `--board_ssh_pass` | SSH password for the Modalix board. Optional. Set to enable non-interactive automated benchmarking. |

:::important
Accuracy benchmarking with `-b modalix` requires the model to be compiled with the `--return_logits` flag. See [Model Compilation](compilation_genai.md). If the model was compiled without this flag, benchmarking will fail at runtime.
:::

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
| `model_id` | HuggingFace model ID (e.g., `meta-llama/Llama-3.2-3B-Instruct`). |
| `-o` | Output directory for benchmark results. |
| `--board_ip` | IP address of the Modalix board. |
| `--board_model` | Path to the compiled model directory on the Modalix device (e.g., `/media/nvme/llima/models/Llama-3.2-3B-Instruct-a16w4`). |
| `--max_num_tokens` | Maximum context length. Must be equal to or smaller than the value used during compilation. |
| `--max_new_tokens` | Maximum number of tokens to generate in the output. |
| `--board_ssh_user` | SSH username for the Modalix board. Optional, default: `sima`. |
| `--board_ssh_pass` | SSH password for the Modalix board. Optional. Set to enable non-interactive automated benchmarking. |

For all available options, run `llima-benchmark perf -h`.
