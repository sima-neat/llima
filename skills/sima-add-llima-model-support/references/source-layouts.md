# Add Hugging Face or GGUF Layouts

Use when computation is supported but source config or tensor storage differs.

## Rules

- Normalize names, shapes, packing, permutations, and metadata at ingestion.
- Validate required tensor rank, shape, dtype, and fused dimension.
- Reject ambiguous layouts instead of guessing.
- Keep format-specific logic out of shared language/vision graphs.
- Validate Hugging Face and GGUF independently.

## Example: Hugging Face Fused QKV

Goal: ingest `qkv_proj.weight` for a graph expecting separate Q/K/V tensors.

- Confirm packing order from upstream code/config.
- Split using validated Q, K, and V output dimensions.
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
