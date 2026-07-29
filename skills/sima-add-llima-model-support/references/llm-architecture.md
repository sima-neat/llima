# Add an LLM Architecture

Use this route only when existing language graph or state semantics cannot
represent the model. Config-key or tensor-name differences alone belong in
ingestion/source-layout handling.

## Implementation Order

1. Use the matching
   [Transformers model implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/)
   at a pinned commit as the architecture and numerical reference. If it is
   absent, use the model repository's pinned remote-code implementation and
   document that provenance.
2. Add the LLiMa architecture identifier and source detection.
3. Add only configuration fields required to express new behavior.
4. Adapt existing pre/cache/post, attention, or convolution modeling files
   when their tensor flow and state contract already fit.
5. Add a new layer type and modeling file only for genuinely new computation
   or persistent state; include it in the compiler-unit matrix.
6. Generate and compile affected units, comparing them with the reference
   implementation. Establish generated config, names, tensor layouts, and
   state interfaces before changing runtime code.
7. Adapt the C++ runtime only for a new execution contract. Drive selection,
   dimensions, and behavior from generated configuration.

## Define the Contract

Compare the upstream config, implementation, and tensor index with the closest
supported architecture. Document:

- attention type; Q/K/V shape, heads, scaling, masks, and biases;
- position/RoPE behavior and full/sliding-window rules;
- normalization type, placement, epsilon, and unit offset;
- MLP topology, activation, tensor names, and widths;
- ordered attention/convolution/other layer schedule;
- KV or convolution state shapes and prefill/decode updates;
- embedding/LM-head tying, logits shape, and stop-token behavior; and
- required group, single, per-layer, or speculative compiler units.

Reject incomplete or contradictory evidence instead of inferring semantics
from a model name.

## Configuration and Compiler Graphs

- Register/parse the architecture in `sima_lmm/config/vlm_config.py`.
- Normalize Hugging Face or GGUF config/tensors at the source boundary.
- Validate required fields, tensor rank/shape/dtype, schedule length, tied
  weights, and optional features.
- Ensure generated `vlm_config.json` carries every runtime-required value.
- Extend `VlmConfig.get_layer_ids()` only for a different unit matrix.
- Reuse `language_part_base.py` and the nearest
  `language_{pre,cache,post}_model.py` implementation.
- Reuse shared attention, MLP, normalization, embedding, and LM-head builders;
  branch only where computation or state differs.
- Keep ONNX and direct Model SDK implementations aligned where both apply.
- Preserve deterministic layer and model-part ordering.

## Runtime Contract

Audit `sima_lmm/devkit/cpp/vlm_config.hpp` and `language_model.*` for:

- generated part names and group/single selection;
- state allocation, update, rollover, reset, and prefill/decode transition;
- masks, RoPE/head dimensions, sliding boundaries, and partial final groups;
- logits, sampling, stop tokens, and context limits; and
- cleanup after completion, cancellation, and failure.

Prefer data-driven generated config. Add a runtime model-type branch only when
the execution contract cannot be expressed by existing configuration.

## Validation

- Test config parsing/defaults/serialization and invalid inputs.
- Test source ingestion and deterministic tensor transforms.
- Compare affected prefill, decode, and state outputs with the reference.
- Exercise group/single variants, boundary lengths, and cache transitions.
- Run ONNX regression and direct graph parity where supported.
- Verify complete required-unit generation and graph integration.
- Run multi-turn generation and clean teardown on Modalix.

Add cases through `tests/compilation/cases.py`; follow `validation-matrix.md`
for manifests, regression mode, and audited counts.

## Example: LFM2-Style Hybrid

For alternating attention/short-convolution layers with `w1`/`w3`/`w2` MLP:

- use LFM2 as the closest analogue and derive ordered `layer_types` from
  upstream config/tensors, never filesystem order;
- keep rolling convolution state separate from transformer KV cache;
- reuse shared MLP and normal pre/cache/post models for attention;
- add a fused convolution model only for convolution layers and preserve its
  state across group-prefill/decode; and
- compare deterministic outputs, compile every required unit, and run
  multi-token Modalix generation.
