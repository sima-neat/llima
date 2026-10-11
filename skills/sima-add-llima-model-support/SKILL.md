---
name: sima-add-llima-model-support
description: Assess LLM/VLM checkpoint compatibility with LLiMa or implement missing architecture, source-layout, tokenizer, or prompt support.
---

# Assess or Add LLiMa Model Support

Assess only the compatibility boundaries relevant to the request, using the
exact model/revision and source artifacts. For source changes, work in a LLiMa
checkout under its `AGENTS.md`; repository docs are not installed with this
skill. Already confirmed compatible models use `sima-llima-compile-run` when
compilation or deployment is requested. Ordinary repository maintenance uses
`sima-contribute-to-llima`.

## Choose the Relevant Reference

| Question | Reference |
| --- | --- |
| Does this checkpoint fit existing support? Which boundary fails? | Relevant sections of the [compatibility audit](references/compatibility-audit.md) |
| Does language computation or persistent state differ? | [LLM architecture](references/llm-architecture.md) |
| Does vision, the projector, or image preprocessing differ? | [VLM architecture](references/vlm-architecture.md) |
| Do HF/GGUF names, config, or tensor storage differ? | [Source layouts](references/source-layouts.md) |
| Do tokenizer, templates, tools, or runtime assets differ? | [Tokenizer and prompt contract](references/tokenizer-prompt-contract.md) |
| Which implementation checks and CI inputs are affected? | [Validation matrix](references/validation-matrix.md) |

Combine routes only for independent differences established by evidence. A
narrow loader or tokenizer assessment does not require an architecture audit,
model compilation, or Modalix access.

## Essential Contracts

Normalize differences at ingestion/configuration boundaries and extend existing
resolvers before adding model-specific graph or runtime branches. Use ModelGraph
for native graphs and preserve staged/direct quantization where applicable.
Align reference tensor layouts as documented in the relevant architecture
reference. Reject ambiguous configuration or tensor layouts.

Preserve checkpoint, revision, format, precision, and image shape. Validate HF
and GGUF independently. Never equate host loading or CPU/reference execution
with complete runtime support; label mirror/derivative evidence as provisional.
Keep credentials, private data, weights, and generated binaries out of Git and
reports. New ASR architectures need a separate design; existing Whisper
maintenance uses the contributor skill.

## Completion

For assessment, report supported and failing boundaries with evidence, unknowns,
and the smallest required change; do not implement or compile unless requested.
For implementation, include affected tests and required-unit/asset generation
using the validation matrix. Pin CI inputs in the relevant cache manifests and
update `docs/index.md` only for established support. Continue through compilation
or Modalix validation when requested, reporting unavailable checks explicitly.
