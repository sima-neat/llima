# MoLE - Modalix Language Model Evaluator

## Overview

MoLE (Modalix Language Model Evaluator) is a benchmarking tool for evaluating the accuracy and performance of LLMs running on the Modalix platform.

It extends [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and supports two backends:

- **hf** — runs evaluation on the host using HuggingFace transformers (baseline reference)
- **modalix** — runs evaluation on a Modalix board via the `llima benchmark-server`

## Installation

Install MoLE in a separate virtual environment on a host or a Modalix DevKit, outside the SDK Docker container. DevKit execution also requires the installed LLiMa runtime. See [Neat Framework installation](/getting-started/neat-library/) for the runtime installation flow.

Install MoLE using `sima-cli`:

``` console
host:~$ sima-cli neat install llima/mole
```

This installs MoLE into a virtual environment at `~/sima-mole-venv`.

## Usage

First, activate the MoLE virtual environment:

``` console
host:~$ source ~/sima-mole-venv/bin/activate
```

For Modalix evaluation from a host, `--board_ip` is required. On a Modalix DevKit, omit it to use the local runtime. Provide `--board_model` when starting a server; local server logs are saved to `<output_dir>/server.log`. Use `--no-board_start_server` to attach to an existing server without starting or stopping it.

MoLE provides the `llima-benchmark accuracy` and `llima-benchmark perf` subcommands. For Modalix, `<model_id>` accepts a HuggingFace model ID or a local tokenizer/config directory matching the deployed model. MoLE uses that tokenizer to prepare requests.

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
| `--board_ip` | Remote board IP or host name. Required when running from a host; omit for local Modalix execution. |
| `--board_model` | Compiled model directory on the DevKit. Required when starting a server. |
| `--max_num_tokens` | Maximum context length. Must be equal to or smaller than the value used during compilation. |
| `-n, --num_samples` | Number of samples to evaluate. Runs the full task set if not specified. |
| `--board_ssh_user` | SSH username for the Modalix board. Optional, default: `sima`. \# |
| `--board_ssh_pass` | SSH password for the Modalix board. Optional. Set to enable non-interactive automated benchmarking. |

:::important
Accuracy and loglikelihood benchmarking with `-b modalix` requires the deployed model to be compiled with `--return_logits`. This flag is off by default. See [Model Compilation](compilation_genai.md). If the model was compiled without this flag, the benchmark fails with: `model not compiled with --return_logits; accuracy/loglikelihood tasks are unsupported`.
:::

In `-b modalix` mode, result tables are labeled as Modalix backend results and include the board target. The HuggingFace `model_id` still appears because MoLE uses it for tokenization and task metadata.

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
    --max_num_tokens <max_num_tokens> --max_new_tokens <max_new_tokens> \
    --input_lengths 1024 2048 3072 4096
```

| Argument | Description |
|----|----|
| `model_id` | HuggingFace model ID (e.g., `meta-llama/Llama-3.2-3B-Instruct`). For Modalix performance runs, this should match the tokenizer/config for the deployed model. |
| `-o` | Output directory for benchmark results. |
| `--board_ip` | Remote board IP or host name. Required when running from a host; omit for local Modalix execution. |
| `--board_model` | Compiled model directory on the DevKit. Required when starting a server. |
| `--max_num_tokens` | Maximum context length. Must be equal to or smaller than the value used during compilation. |
| `--max_new_tokens` | Maximum number of tokens to generate in the output. |
| `--input_lengths` | Optional exact input-token lengths to benchmark. Values must be unique and each value plus `--max_new_tokens` must fit within `--max_num_tokens`. When omitted, MoLE generates automatic power-of-two buckets. |
| `--board_ssh_user` | SSH username for the Modalix board. Optional, default: `sima`. |
| `--board_ssh_pass` | SSH password for the Modalix board. Optional. Set to enable non-interactive automated benchmarking. |

For all available options, run `llima-benchmark perf -h`.

Use `--dataset bfcl-v3|ifeval|alpacaeval|gsm8k|dialogsum` for native chat/tool prompts instead of token-length buckets. Omitting `-n` runs the full selected dataset; `-n 20` runs its first 20 samples. Datasets download automatically and are cached. BFCL covers simple/multiple cases; tools are not executed.

`--board_model` determines ordinary or speculative decoding. Compare models with separate runs using the same dataset and settings, changing only `--board_model` and `-o`. Each run excludes one warmup and resets KV state before every request. The model's native chat template and EOS stopping apply. Dataset output budgets include the first token; existing buckets measure the first token plus `--max_new_tokens` decode steps.

`-o` saves `performance.json` with timings, token counts, aggregate metrics, draft contribution, failures, benchmark settings and dataset sources. Decode TPS excludes TTFT; generation timings exclude input preparation and network transport. These workloads measure performance, not accuracy.
