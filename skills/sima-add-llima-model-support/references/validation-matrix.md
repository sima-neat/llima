# Model-Support Validation Matrix

Select every row affected by the chosen route. Run the exact commands documented
in `docs/contributing.md` and `tests/README.md`.

| Surface | Minimum evidence | Example |
| --- | --- | --- |
| LLM configuration | Hermetic parsing and invalid-input tests | Reject a hybrid layer schedule whose length differs from `num_hidden_layers` |
| LLM numerical behavior | Deterministic prefill/cache/post comparisons | Compare LFM2-style convolution and attention units against upstream outputs |
| VLM preprocessing | Exact processor input comparison | Compare Qwen3-VL patch tensors for one fixed image and resolution |
| VLM numerical behavior | Vision/projector ONNX comparison | Compare the final projected image embeddings before language insertion |
| Hugging Face layout | Synthetic transform plus source-ingestion case | Split a marked fused QKV tensor and assert every output slice |
| GGUF layout | Config parity, permutation, and quantization cases | Compare Mistral HF/GGUF config and Q/K tensors |
| Tokenizer/template | Exact rendered prompt and token IDs | Render a Gemma adjacent-literal template through Jinja and Minja |
| Compiler integration | Required units generated without unintended skips | Generate all selected group/single or vision units |
| Modalix runtime | Representative task succeeds and exits cleanly | Run two LLM turns or one image-grounded VLM prompt |

## CI Inputs

- Pin source models by immutable revision in the appropriate
  `tools/hf-safetensors/manifest.txt`, `config-manifest.txt`, or
  `gguf-manifest.txt`.
- Add the smallest maintained case to `tests/compilation/cases.py`.
- Generate ONNX and numerical comparison artifacts during the test run.
- Do not treat cache absence or a skipped required case as a pass.

## Completion Example

For a new Qwen3-VL-style VLM, report:

- source model ID and revision;
- VLM and source-layout routes selected;
- configuration, processor, vision ONNX, and graph-integration tests passed;
- complete compilation options and artifact location;
- Modalix image-prompt result;
- GGUF not supported because it was not implemented or validated; and
- supported-model documentation updated only for Hugging Face input.
