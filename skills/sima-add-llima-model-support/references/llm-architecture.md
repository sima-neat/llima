# Add an LLM Architecture

Use this route only when existing language graph semantics cannot represent the
model.

## Define the Contract

Document:

- attention type, Q/K/V shapes, head layout, scaling, masks, and biases;
- positional encoding and RoPE variant;
- normalization type, placement, epsilon, and unit offset;
- MLP topology, activation, tensor names, and intermediate widths;
- full, sliding, convolutional, or other layer schedule;
- KV-cache tensors, update rules, and prefill/decode differences;
- embedding and LM-head tying; and
- required compiler units for group and single-token execution.

## Implementation Surfaces

- Register and parse the architecture in `sima_lmm/config/vlm_config.py`.
- Normalize source configuration and tensor names under `sima_lmm/hf/` or
  `sima_lmm/gguf/`.
- Reuse `language_part_base.py` primitives and extend
  `language_pre_model.py`, `language_cache_model.py`, and
  `language_post_model.py` only where computation differs.
- Add convolution or per-layer compiler units only when the architecture
  requires them.
- Keep ONNX and direct Model SDK graph implementations numerically aligned.
- Audit C++ runtime assumptions about generated part names, cache buffers, RoPE,
  stop tokens, and output shapes. Prefer generated configuration over new
  runtime model-type branches.

## Example: LFM2-Style Hybrid Layers

Request:

> Add a language architecture that alternates attention and short-convolution
> layers and uses `w1`, `w3`, and `w2` MLP tensors.

Investigation:

- Use LFM2 as the nearest supported analogue.
- Derive the ordered `layer_types` from upstream configuration or tensor
  evidence.
- Confirm convolution state and cache behavior differ from transformer KV
  cache rather than treating every layer as attention.

Implementation:

- Add a language architecture identifier and config normalization.
- Reuse the shared MLP builder by selecting `w1`/`w3`/`w2` when those tensors
  exist.
- Route convolution layers through the convolution model while retaining the
  normal pre/cache/post path for attention layers.
- Preserve the generated layer order; do not infer it from filesystem order.

Validation:

- Unit-test layer-schedule parsing and invalid schedules.
- Compare prefill, decode, post, and convolution outputs with the upstream
  implementation using deterministic inputs.
- Exercise group and single-token compiler units.
- Run compiler graph integration, a complete build, and multi-token generation
  on Modalix.

