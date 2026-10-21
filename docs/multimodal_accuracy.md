# Image-plus-text accuracy with MoLE

Install matching candidate runtime and MoLE packages using the existing installation flows. `sima-cli neat install llima/mole` installs the image evaluator in MoLE's separate virtual environment. No compiler environment or second evaluator command is required. The benchmark server must support the additive `multimodal_info` and `generate_multimodal` requests; existing text-token requests remain compatible.

Image accuracy uses the deployed model's native C++ chat template, image preprocessing and vision/language execution. Images travel inline as lossless RGB PNGs, so host paths never become DevKit paths. Generation tasks do not require compilation with `--return_logits`.

## Validation status

The initial targets are `realworldqa`, `docvqa_val` and `chartqa`, starting with a compiled Qwen3.5-2B VLM. Their unchanged upstream protocols/scoring have integration coverage and a six-sample Qwen3.5-2B smoke evaluation on a Neat 3.0 DevKit in both host-driven and DevKit-local modes. Answers, prompt/image hashes, token counts and metrics matched across modes. See [validation evidence](multimodal_accuracy_validation.md). Full-dataset accuracy has not been measured; these smoke scores must not be published as registry accuracy.

Other upstream image tasks use the same adapter when their protocols fit, but TextVQA, OCRBench, MMMU, MMBench and MMStar are not yet validated support claims. Use individual upstream task names, rather than groups or wildcards. Run text and image tasks in separate invocations. Text accuracy continues to use lm-evaluation-harness, including existing HF host workflows. Performance benchmarking is separate.

## Host and DevKit invocation

Activate `~/sima-mole-venv/bin/activate` after installation. On a **host**, supply a directory containing the exact deployed tokenizer/processor assets and the remote target:

```bash
llima-benchmark accuracy /data/qwen3.5-2b-assets -b modalix \
  -t realworldqa docvqa_val chartqa \
  --board_ip <devkit-address> --board_model <compiled-model-on-devkit> \
  --max_num_tokens 4096 -o results/vlm-host
```

On the **DevKit**, omit `--board_ip`. The model directory's `devkit/` assets can be used directly:

```bash
llima-benchmark accuracy /media/nvme/llima/models/<compiled-qwen3.5-2b> -b modalix \
  -t realworldqa docvqa_val chartqa \
  --board_model /media/nvme/llima/models/<compiled-qwen3.5-2b> \
  --max_num_tokens 4096 -o results/vlm-local
```

This starts the installed `/usr/bin/llima` runtime locally, without SSH or another host. The MoLE environment remains separate from the runtime and compiler. `--board_venv_path` can select a different runtime installation; `--no-board_start_server` attaches without taking ownership of the server. Managed runs stop only their owned server, including on failures and interruptions.

Host-driven server startup first tries SSH agent/key authentication. If authentication fails and no explicit password was supplied, it prompts for a password. `--board_ssh_pass` retains the existing explicit-password workflow. DevKit-local execution does not use SSH.

`model_id` also accepts a Hugging Face repository, with optional `--model_revision`. MoLE resolves an immutable revision and downloads only tokenizer/config/processor assets; weights are not downloaded. These files must match the deployed assets. The compiled `vlm_config.json`, including any chat-template override, remains authoritative.

Each task retains its upstream generation settings. `--max_new_tokens` explicitly overrides output budgets and is recorded. Context plus output must fit both `--max_num_tokens` and the compiled context; inputs are never truncated to make them fit. The current adapter supports greedy decoding. Unsupported sampling/beam settings, logits tasks, multiround tasks, audio/video and non-image visuals fail clearly. Thinking defaults off; `--enable_thinking` selects native thinking behavior. The runtime's existing reasoning parser supplies final content to scoring while raw output is retained.

Native EOS stops generation. Upstream string stops trim the returned answer after generation, rather than being flattened to individual stop-token IDs. Output-budget termination is recorded as `length`, even when the answer can still be scored.

Multiple images retain upstream order. Literal `<image>` placeholders consume images in order; `<image 1>`, `<image 2>` explicitly interleave them with text. Without placeholders, images precede the task's text, matching the upstream simple-adapter convention. Missing, duplicated, unrecognized or mismatched placeholders fail instead of discarding images. Few-shot visual examples and repeated requests are not yet supported.

## Prepared datasets and offline CI

Registry owns dataset declarations, preparation and CI orchestration. MoLE owns inference, upstream scoring and result exports. Online runs resolve dataset revisions before loading. For CI, provide a JSON accuracy manifest with pinned source revisions and prepared Hugging Face `DatasetDict.save_to_disk()` directories. Preserve upstream columns, image features and scoring splits. Local paths are relative to the manifest:

```json
{
  "schema_version": 1,
  "tasks": {
    "realworldqa": {
      "revision": "<40-character-source-dataset-commit>",
      "split": "test",
      "local_path": "datasets/realworldqa"
    },
    "docvqa_val": {
      "revision": "<40-character-source-dataset-commit>",
      "split": "validation",
      "local_path": "datasets/docvqa"
    },
    "chartqa": {
      "revision": "<40-character-source-dataset-commit>",
      "split": "test",
      "local_path": "datasets/chartqa"
    }
  }
}
```

Pass `--accuracy_manifest accuracy-manifest.json` in either mode. Omit `local_path` to load the specified remote dataset revision. Manifest task names must exactly match `-t`. For offline runs, use local matching model assets and set `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`. Prepared file hashes and fingerprints are recorded alongside the declared source revision; the CI preparing datasets is responsible for preserving that source's contents.

## Exports and failure policy

The output directory contains:

- `accuracy.json`: aggregate upstream metrics, revisions/splits/fingerprints, selected sample IDs, task configs, scorer source identity, package versions, compiled model file hashes, loaded runtime library hash/platform identity, tokenizer/processor hashes, image resolution/preprocessing configuration, context/output/thinking settings and counts.
- `accuracy-answers.jsonl`: attempted requests with sample IDs, ordered text/image descriptors, image hashes/resolutions, raw and final generated answers, generation settings, actual prompt hashes, token counts, finish reasons or errors.
- `accuracy-samples.jsonl`: upstream per-sample targets, responses, filtered predictions, task metrics and hashes for a completed scoring run. Task-specific internal extraction can occur within upstream scoring; MoLE does not implement a second extractor.

Failures stop the run with a nonzero exit code and preserve partial answers. Failed runs publish no aggregate metrics. Counts distinguish selected, attempted, answered, failed, unsupported, truncated, not-run and scored samples. Truncated answers remain in the evaluated set and are scored normally. `-n` selects a prefix for smoke testing; limited or failed runs must not be published as full-dataset registry accuracy.

Compare host and local runs using identical dataset files/revisions and sample IDs, deployed artifact, runtime identity and generation settings. Check prompt/image hashes, per-sample answers, counts and metrics; investigate any differences. Do not infer VLM accuracy from compilation or text-only results.

## Evaluator dependency assessment

The MoLE build pins lmms-eval's exact source commit in `deps/manifest.json`, builds it on the packaging host and bundles a pure-Python wheel. `tools/build_lmms_profile.py` preserves every upstream code/task asset and license, records their hashes, and narrows wheel dependency metadata to imports required for image task evaluation. Its local version identifies the commit. Upstream's `v0.7.3` tag currently declares distribution version `0.7.2`; commit and source-tree hashes resolve that ambiguity.

The profile reuses MoLE's Torch, Transformers, datasets, Pillow and text-harness dependencies and includes evaluate, sacrebleu, accelerate, aiohttp, pandas, pytz and tenacity. It excludes unrelated model, audio/video, judge-service and training extras. Missing task-specific optional dependencies fail rather than installing software during evaluation. Import, dependency-consistency and prepared-task scoring checks passed on ARM64 Python 3.12 and on the target Neat 3.0 DevKit with Python 3.13.5. The evaluator stays separate from runtime and compiler dependencies.

Run the hermetic checks with the candidate MoLE environment:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m pytest -o addopts='' tests/mole -q
```

Runtime tests require installed candidate packages and real Modalix hardware, following [Test Suites](../tests/README.md). The wire/scoring tests do not substitute for image-grounded hardware validation.
