# Add Hugging Face or GGUF Source Layouts

Use this route when model computation is already supported but upstream
configuration or tensor storage differs.

## Rules

- Normalize names, shapes, packing, and metadata at the source boundary.
- Validate every required tensor and its expected rank, shape, and dtype.
- Reject ambiguous fused layouts instead of guessing split order.
- Keep format-specific permutations out of shared language and vision graph
  code.
- Validate Hugging Face and GGUF independently; support for one does not imply
  support for the other.

## Example: Hugging Face Fused QKV

Request:

> Support a checkpoint that stores one `qkv_proj.weight` instead of separate
> `q_proj`, `k_proj`, and `v_proj` tensors.

Implementation:

- Read the upstream packing order from model code or published configuration.
- Split the fused tensor using validated Q, K, and V output dimensions.
- Expose the normalized tensors through the existing parameter-loading
  interface.
- Reject a tensor whose packed dimension does not equal the expected sum.

Validation:

- Create a small deterministic fused tensor with distinguishable Q, K, and V
  ranges.
- Assert exact split values and shapes.
- Run source-ingestion and affected ONNX numerical cases.

## Example: Mistral GGUF Normalization

Request:

> Add a Mistral GGUF that reports the GGUF `llama` architecture.

Implementation:

- Use immutable GGUF metadata such as `general.name` or `general.basename` to
  normalize the model type to Mistral only when the evidence is explicit.
- Apply the Llama/Mistral Q/K permutation policy at GGUF loading.
- Map GGUF tensor names to the normalized Hugging Face-style parameter names.
- Keep the shared Mistral language graph unchanged.

Validation:

- Compare generated GGUF and Hugging Face configurations for the same
  architecture.
- Test Q/K permutation with synthetic weights and scales.
- Run GGUF source-ingestion, quantization-specific, and ONNX parity cases.

