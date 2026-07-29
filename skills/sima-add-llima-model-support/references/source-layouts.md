# Add Hugging Face or GGUF Layouts

Use when computation is supported but source config or tensor storage differs.

## Rules

- Normalize names, shapes, packing, permutations, and metadata at ingestion.
- Trace existing fallback resolution in both ONNX and direct Model SDK paths
  before implementing a new split or transform.
- Validate required tensor rank, shape, dtype, and fused dimension.
- Reject ambiguous layouts instead of guessing.
- Keep format-specific logic out of shared language/vision graphs.
- Validate Hugging Face and GGUF independently.

## Existing Bundled-Weight Resolver

`sima_lmm/model/onnx_builder.py::find_alternate_weight` already resolves:

- `q_proj`, `k_proj`, and `v_proj` from `qkv_proj`; and
- `gate_proj` and `up_proj` from `gate_up_proj`.

Both `onnx_builder.py` and `sima_builder.py` call it, so reuse it to keep ONNX
and direct Model SDK behavior aligned. Verify its assumptions against the exact
tensor index: source prefix, module name, packing order, output axis, Q/K/V
sizes, and equal gate/up widths.

If the source uses a different block or module prefix, add the smallest name
normalization that lets the existing resolver operate. Do not reimplement its
split merely because the original requested name is absent.

## Example: Hugging Face Fused QKV

Goal: ingest `qkv_proj.weight` for a graph expecting separate Q/K/V tensors.

- First trace the requested `q_proj`/`k_proj`/`v_proj` name through
  `find_alternate_weight`.
- Confirm packing order from upstream code/config.
- Reuse the resolver's split with validated Q, K, and V output dimensions.
- Expose normalized tensors through the existing loader.
- Reject a packed dimension that differs from the expected sum.
- Test exact shapes/values with marked synthetic ranges, then run affected
  source-ingestion and ONNX cases.

## Example: Mistral GGUF

Goal: recognize Mistral stored with GGUF `llama` architecture metadata.

- Normalize to Mistral only from explicit immutable metadata such as
  `general.name` or `general.basename`.
- Apply the Llama/Mistral Q/K permutation and map names to normalized
  Hugging Face-style parameters.
- Keep the shared Mistral graph unchanged.
- Compare HF/GGUF configs, test permutations with synthetic weights/scales,
  and run GGUF ingestion, quantization, and ONNX parity cases.
