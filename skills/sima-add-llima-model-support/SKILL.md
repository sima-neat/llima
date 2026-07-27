---
name: sima-add-llima-model-support
description: Add LLM and VLM model support in the sima-neat/llima repository. Use when a contribution introduces a new language architecture, a new vision encoder or multimodal projector, a Hugging Face or GGUF tensor-layout mapping, or tokenizer/chat-template compatibility. Do not use for ASR architecture work, compiling a compatible checkpoint on an existing architecture, or ordinary ONNX model compilation.
---

# Add LLiMa Model Support

Add model support through the smallest correct extension point. Normalize
upstream differences at ingestion or configuration boundaries before adding
architecture-specific compiler or runtime branches.

## Start

1. Read `AGENTS.md`, `CONTRIBUTING.md`, and `docs/contributing.md`.
2. Record the public model ID, immutable revision, source format, modality,
   upstream configuration, tensor index, tokenizer files, and processor files.
3. Compare the model with the closest supported case in
   `tests/compilation/cases.py`.
4. If the model matches an existing architecture, source layout, tokenizer,
   and chat-template contract, stop this repository-change workflow and use
   `sima-llima-compile-run`. Do not add checkpoint-specific source branches.
5. Otherwise, classify the actual compatibility gap using the routes below.
6. Define the expected generated configuration, compiler units, runtime input
   contract, validation cases, cache entry, and documentation impact before
   editing.

## Choose a Route

- **New LLM architecture**: read `references/llm-architecture.md`.
  Example: use the existing LFM2 integration as the pattern for a hybrid model
  that adds short-convolution layers and `w1`/`w2`/`w3` MLP weights.
- **New VLM encoder or projector**: read `references/vlm-architecture.md`.
  Example: use Qwen3-VL support to see where a new vision encoder, projector,
  image preprocessing, and language-embedding contract diverge from Qwen2.5-VL.
- **Hugging Face or GGUF layout mapping**: read
  `references/source-layouts.md`.
  Example: normalize a fused Hugging Face QKV tensor, or apply the existing
  Mistral GGUF model-type and Q/K permutation rules without changing language
  graph semantics.
- **Tokenizer or chat-template compatibility**: read
  `references/tokenizer-chat-template.md`.
  Example: make a Gemma template using standard adjacent string literals render
  through Minja without hardcoding a replacement template for that checkpoint.

Read `references/validation-matrix.md` after the route-specific reference.
Combine routes only when evidence shows independent differences. For example,
a new VLM may need both the VLM and source-layout routes, while a compatible
new size of an existing LLM leaves this workflow and uses
`sima-llima-compile-run`.

## Implement

- Add the smallest reusable configuration, ingestion, graph, preprocessing, or
  runtime change that expresses the upstream contract.
- Keep Hugging Face and GGUF normalization outside shared model computation.
- Keep compiler-only dependencies out of `sima_lmm/devkit/`.
- Keep ONNX and direct Model SDK graph behavior aligned when both paths support
  the affected compiler unit.
- Reject missing, ambiguous, or unsupported configuration and tensor layouts
  with actionable errors.
- Add hermetic tests for pure logic and model-backed tests for the affected
  compiler surface. A skipped required model case is not validation.
- Add immutable model inputs to the appropriate
  `tools/hf-safetensors/*.txt` manifest only when CI needs them.
- Update `docs/index.md` only after the support and validation level is clear.

## Boundaries

- Do not add ASR architecture support through this skill. The current ASR path
  is Whisper-specific and is not an extensible architecture framework.
- Do not claim GGUF support because Hugging Face support works; validate GGUF
  configuration, tensor mapping, quantization, and tokenizer behavior
  separately.
- Do not commit downloaded weights or generated ONNX, NumPy, MPK, ELF, or
  runtime model trees.
- Do not silently substitute another checkpoint, revision, precision, image
  shape, or source format.
- Treat a checkpoint conforming to an existing architecture and its ingestion
  contracts as already supported. Use `sima-llima-compile-run` to compile and
  smoke-test it instead of adding checkpoint-specific repository changes.

## Finish

Report:

- the selected route and closest supported analogue;
- source model and immutable revision;
- compiler and runtime surfaces changed;
- tests run for configuration, ingestion, numerical behavior, compilation,
  and Modalix execution;
- unavailable model or hardware checks;
- supported source formats and remaining limitations; and
- documentation and CI-cache changes.
