# Multimodal accuracy validation

Validated on 2026-10-21 using a Modalix DevKit running Neat 3.0 (`3.0.0_develop_B2121`), Python 3.13.5 and the packaged `simaai/Qwen3.5-2B-Autoround-a16w4` artifact at immutable revision `9e3720332190687093bdf7a00a8ba4a4d42dcb5f`. The artifact has context 4096 and `return_logits: false`.

Candidate runtime Debian packages were built natively on the device using matching B2121 development headers, then extracted into a private prefix selected through `--board_venv_path`. The installed system runtime was untouched. This validates packaged binaries, but does not replace a system-installed package/upgrade acceptance run. MoLE was installed in a separate virtual environment. The native vision-generation runtime test passed. The ZMQ hardware check also passed for single-image and two-image requests, runtime identity, and recovery after unsupported/malformed requests. This reused the runtime test assertions with only its executable-prefix guard changed in a temporary copy to select the extracted candidate; the repository test still requires a system-installed runtime.

## Host and local evaluation

Both modes used the first two original samples of each prepared dataset, seed 42, context 4096, greedy decoding, thinking disabled and unchanged upstream task output budgets/scoring. Each prepared DatasetDict contained the first three rows of its first scoring shard; `-n 2` selected two. Datasets were loaded offline from `save_to_disk()` directories with their immutable source declarations.

| Task | Immutable dataset revision | Split | Samples | Smoke metric |
| --- | --- | --- | --- | --- |
| RealWorldQA | `lmms-lab-encoder/RealWorldQA@907c4e5228fd1703c710ed937601cb5f89ab8d5c` | test | 2 | exact_match 0.0 |
| DocVQA | `lmms-lab-encoder/DocVQA@539088ef8a8ada01ac8e2e6d4e372586748a265e` | validation | 2 | ANLS 0.875 |
| ChartQA | `lmms-lab-encoder/ChartQA@9e63b7df1592a1c2158e735cc1725454aef0d6d9` | test | 2 | relaxed_overall 0.0 |

These are execution smoke scores, not estimates of full-dataset accuracy. No answer normalization beyond the upstream task scorer was introduced to improve scores.

All six samples were selected, attempted, answered and scored in each mode; failed, unsupported, truncated and not-run counts were zero. An exact comparison passed for sample IDs, answers, raw answers, ordered content, image hashes, generation settings, native prompt hashes, actual input/output token counts, finish reasons, task metrics and counts. Both runs verified the same upstream scorer source tree and compiled artifact/runtime identities:

- Compiled artifact manifest SHA256: `1a6153df5a706cbc10c7c8b22a192bd61cd8a0d8da23ba9e5a4fcb6e0cb946c9`.
- Loaded runtime library SHA256: `6dfad02b7b25ed4b51cb7330c379b5c2cab5fa7f3dca1c442089f8965d47102c`.

Each run exported `accuracy.json`, `accuracy-answers.jsonl` and upstream `accuracy-samples.jsonl`. Host execution managed the remote server through SSH key authentication. Local execution omitted `--board_ip` and launched its server through a local subprocess; no separate evaluator host was needed.

## Compatibility and remaining coverage

The existing text-only GSM8K path completed one generation sample through lm-evaluation-harness on Qwen2.5-0.5B using the candidate server, also without logits. Existing performance benchmarking completed one 128-token prompt with an eight-token output budget. A 32-token prompt was explicitly rejected because that chat template could not fit it. These are compatibility checks, not accuracy/performance measurements.

The isolated DevKit evaluator installation passed `pip check` and 22 tests. The final host suite, including the real task-manager routing regression, passed 23 tests. The profile uses the pinned upstream evaluator and excludes unrelated audio/video/model backends.

## Full-dataset RealWorldQA validation

Subsequent DevKit-local runs used all 765 samples from the RealWorldQA revision
and split above, context 4096, non-thinking greedy decoding, a 16-token output
budget and unchanged upstream scoring. Both models used fixed 448 × 448 native
image processing. The candidate runtime was installed system-wide as
`0.6.0+codex.mole.vlm.accuracy.4a2f249`; its library hash matched the value above.
System-installed single/multi-image request and recovery acceptance also passed.

| Artifact | Correct / scored | Exact-match accuracy | Truncated |
| --- | --- | --- | --- |
| Packaged Qwen3.5-2B AutoRound A16W4, revision above | 451 / 765 | 58.95% | 3 |
| Original Qwen3.5-2B compiled in BF16 | 462 / 765 | 60.39% | 0 |

Both runs answered and scored all selected samples, with zero failed,
unsupported or not-run samples. Truncated answers remained in scoring. The BF16
source was `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`.
All 119 compiler units declared BF16 precision; embedding and KV-cache
quantization were disabled. Compilation used Model SDK 3.0.0, Frontend
`3.0.0.dev0+master.396`, MLC `3.0.0.dev0+master.189` and LLiMa branch source
at `b65ef8c4fafffa40f96e982fbf717dbcb5177cdb`.

Across all 765 samples, selected IDs, ordered image/text descriptors, generation
settings, native prompt hashes and input token counts matched. The two-sample
BF16 pilot also matched the captured native image tensors and the original
PyTorch BF16 answers. This does not establish numerical equivalence over the
full dataset. BF16 host/local parity has not been repeated.

These scores are regression baselines for the recorded deployments. They do
not reproduce Qwen's published evaluation protocol. Release image processing,
prompt and sampling details remain unverified, and compiler provenance differs
between the two deployed artifacts. The score difference cannot be attributed
solely to quantization.

Individual validation of TextVQA, OCRBench, MMMU, MMBench and MMStar remains
outstanding. Sampling, visual few-shot examples, multiround protocols and
audio/video are explicitly unsupported.
