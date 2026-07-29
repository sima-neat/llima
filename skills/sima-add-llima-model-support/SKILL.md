---
name: sima-add-llima-model-support
description: Add LLM or VLM model support in sima-neat/llima. Use for a new language architecture, vision encoder or multimodal projector, Hugging Face or GGUF tensor layout, or tokenizer/prompt contract. Do not use for ASR architecture work, a compatible checkpoint on an existing architecture, or ordinary ONNX compilation.
---

# Add LLiMa Model Support

Normalize upstream differences at ingestion/configuration boundaries before
adding architecture-specific compiler or runtime branches.

## Classify

1. Read `AGENTS.md` and `docs/contributing.md`.
2. Record model ID/revision, format, modality, config, tensor index, tokenizer,
   template, and processor files.
3. Compare with the closest case in `tests/compilation/cases.py`.
4. If architecture, layout, tokenizer, and prompt contracts already match, use
   `sima-llima-compile-run`; do not add checkpoint-specific branches. Example:
   a new size or fine-tune of an existing Llama architecture needs compilation,
   not source changes.
5. Otherwise choose the smallest route:

| Route | Reference | Example |
| --- | --- | --- |
| New LLM architecture | `references/llm-architecture.md` | LFM2-style hybrid attention/convolution layers |
| New VLM encoder/projector | `references/vlm-architecture.md` | Qwen3-VL vision/projector differences from Qwen2.5-VL |
| New HF/GGUF layout | `references/source-layouts.md` | Fused HF QKV or Mistral GGUF normalization |
| Tokenizer/prompt compatibility | `references/tokenizer-prompt-contract.md` | Tool-aware template requiring a standards-compatible Minja fix |

Combine routes only for independently proven differences. Then read
`references/validation-matrix.md`.

## Implement

- Express the upstream contract with the smallest reusable config, ingestion,
  graph, preprocessing, or runtime change.
- Keep format-specific normalization out of shared model computation.
- Keep ONNX and direct Model SDK graphs aligned where both apply.
- Reject missing or ambiguous config/tensor layouts.
- Add hermetic tests for pure logic and model-backed coverage for each affected
  compiler surface.
- Add immutable CI inputs only to the appropriate
  `tools/hf-safetensors/*.txt` manifest.
- Update `docs/index.md` only after support and validation are clear.

## Boundaries

- Existing Whisper maintenance uses `sima-contribute-to-llima`,
  `scripts/gen_models--openai--whisper.py`, and
  `sima_lmm/devkit/cpp/whisper_*`. New ASR architectures require a separate
  compiler/runtime design.
- Validate Hugging Face and GGUF independently.
- Never substitute another checkpoint, revision, precision, image shape, or
  source format.
- Follow repository artifact/security policy; do not commit models or
  generated compiler/runtime binaries.

## Finish

Report the route and analogue, model/revision, changed surfaces, source formats,
tests and Modalix evidence, unavailable checks, limitations, and docs/cache
changes.
