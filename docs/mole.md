# MoLE - Modalix Language Model Evaluator

## Overview

MoLE (Modalix Language Model Evaluator) is a benchmarking tool for evaluating the accuracy and performance of LLMs and VLMs running on the Modalix platform.

Text accuracy uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Modalix image-plus-text accuracy uses [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), reusing upstream tasks, extraction and scoring. Both use `llima-benchmark accuracy`. MoLE supports two backends:

- **hf** — runs text accuracy evaluation on the host using HuggingFace transformers (baseline reference)
- **modalix** — runs evaluation on a Modalix board via the `llima benchmark-server`

## Installation

Install MoLE in a separate virtual environment on a host or a Modalix DevKit, outside the SDK Docker container. DevKit execution also requires the installed LLiMa runtime. See [Neat Framework installation](/getting-started/neat-library/) for the runtime installation flow.

Install MoLE using `sima-cli`:

``` console
host:~$ sima-cli neat install llima/mole
```

This installs MoLE into a virtual environment at `~/sima-mole-venv`.

### Manual installation

Without `sima-cli`, run the bundled installer from the downloaded MoLE package directory:

```bash
bash ./install_mole.sh --venv ~/sima-mole-venv
```

For direct wheel installation, activate a Python 3.11–3.13 virtual environment and run the following, replacing the wheel filename with the downloaded version:

```bash
python -m pip install './sima_lmm-<version>-py3-none-any.whl[mole]'
python -m pip install --no-deps "lmms-eval==0.7.3" "tenacity>=8.3.0"
```

The second command enables image benchmarking while reusing MoLE's shared dependencies.

## Usage

First, activate the MoLE virtual environment:

``` console
host:~$ source ~/sima-mole-venv/bin/activate
```

For Modalix evaluation from a host, `--board_ip` is required. On a Modalix DevKit, omit it to use the local runtime. Provide `--board_model` when starting a server; local server logs are saved to `<output_dir>/server.log`. Use `--no-board_start_server` to attach to an existing server without starting or stopping it.

MoLE provides the `llima-benchmark accuracy` and `llima-benchmark perf` subcommands. Text workloads use a HuggingFace model ID or local tokenizer/config directory matching the deployed model. Image workloads use the deployed VLM's assets; `<model_id>` labels the run.

### Accuracy Benchmarking

#### Text accuracy

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
| `model_id` | HuggingFace model ID or local tokenizer/config directory for text accuracy; a descriptive label for image accuracy. |
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
Tasks that consume logits or loglikelihoods require `--return_logits` during compilation. Generation-based tasks, including image-plus-text evaluation, do not require it. See [Model Compilation](compilation_genai.md).
:::

In `-b modalix` mode, result tables are labeled as Modalix backend results and include the board target. The HuggingFace `model_id` still appears because MoLE uses it for tokenization and task metadata.

To use the HuggingFace backend as a reference baseline:

``` console
(sima-mole-venv) host:~$ llima-benchmark accuracy <model_id> -b hf -t <task> -o <output_dir>
```

For all available options, run `llima-benchmark accuracy -h`.

#### Image-plus-text accuracy

The validated tasks are `realworldqa`, `docvqa_val` and `chartqa`, using Qwen3.5-2B on Modalix. The evaluator is installed with MoLE and uses the deployed VLM's native chat template and image preprocessing.

```bash
llima-benchmark accuracy qwen3.5-2b -b modalix \
    -t realworldqa docvqa_val chartqa \
    --board_ip <devkit-address> --board_model <compiled-model-on-devkit> \
    --max_num_tokens 4096 -o results/vlm
```

On the DevKit, omit `--board_ip`. No host model assets are needed. Use matching runtime and MoLE releases.

| Image accuracy option | Description |
|----|----|
| `--max_new_tokens` | Override the task's output budget; otherwise retain its upstream default. |
| `--enable_thinking` | Enable the model's native thinking template; disabled by default. |
| `--accuracy_manifest` | Use pinned dataset revisions or prepared local datasets, as below. |

Run text and image tasks separately, using individual task names. Image accuracy supports ordinary VLMs with greedy decoding. Audio/video, multi-turn, visual few-shot and repeated-request tasks are unsupported. Inputs plus the output budget must fit the requested and compiled context; inputs are not truncated. Multiple images retain their task-defined order. Native EOS stops generation; task string stops trim the returned answer.

#### Prepared image datasets

Datasets download automatically. For pinned or offline evaluation, pass `--accuracy_manifest manifest.json`. For example, with `-t chartqa`:

```json
{
  "schema_version": 1,
  "tasks": {
    "chartqa": {
      "revision": "<40-character-dataset-commit>",
      "split": "test",
      "local_path": "datasets/chartqa"
    }
  }
}
```

Prepare local datasets with HuggingFace `DatasetDict.save_to_disk()`, preserving upstream columns, images and scoring splits. Paths are relative to the manifest, and its task names must match `-t`. Omit `local_path` to download the pinned revision. For offline runs, set `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`.

#### Image accuracy results

The output directory contains:

- `accuracy.json`: aggregate metrics, sample counts, dataset revisions, deployment path, runtime version and settings.
- `accuracy-answers.jsonl`: generated answers, image/prompt identities, token counts, finish reasons and errors.
- `accuracy-samples.jsonl`: upstream per-sample scoring results.

Omitting `-n` runs the full dataset. Failures stop evaluation, preserve partial answers and produce no aggregate metrics. Output-budget truncations are counted and still scored.

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
| `model_id` | HuggingFace model ID or local tokenizer/config directory matching the deployed model for text workloads; a descriptive run label for image workloads. |
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

#### Image-plus-text performance

Use `--dataset realworldqa|docvqa_val|chartqa` to measure ordinary or speculative VLMs with the same upstream image/text prompts used for accuracy:

```bash
llima-benchmark perf qwen3.5-vlm --dataset chartqa \
    --board_ip <devkit-address> --board_model <compiled-model-or-speculative-bundle> \
    --max_num_tokens 4096 --max_new_tokens 256 -o results/vlm-perf
```

Omitting `-n` runs the full dataset once, after one excluded warmup. `--dataset_manifest` accepts the same pinned/local dataset format as `--accuracy_manifest`; its task name must match `--dataset`. Images retain their dataset resolution and use native runtime preprocessing. No host model assets are needed. Accuracy evaluation remains limited to ordinary decoding.

`-o` saves `performance.json` with per-request timings, token counts, aggregate metrics, draft contribution, failures, benchmark settings and dataset sources. Decode TPS excludes TTFT. Image TTFT includes native image preprocessing and vision encoding; text TTFT starts at language prefill. Both exclude network transport and host dataset preparation. Draft contribution is accepted draft tokens divided by all generated tokens. These workloads measure performance, not accuracy.
