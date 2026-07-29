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
| Tokenizer/prompt contract | Exact rendered prompt and token IDs | Render ordinary and tool-enabled messages through the source Jinja implementation and Minja |
| Compiler integration | Required units generated without unintended skips | Generate all selected group/single or vision units |
| Modalix runtime | Representative task succeeds and exits cleanly | Run two LLM turns or one image-grounded VLM prompt |

## Model-Support CI Checklist

1. Add the smallest affected cases to `tests/compilation/cases.py`.
2. Pin inputs by immutable revision in the matching cache manifest:
   - weighted HF: `tools/hf-safetensors/manifest.txt`;
   - metadata-only HF: `tools/hf-safetensors/config-manifest.txt`;
   - GGUF: `tools/hf-safetensors/gguf-manifest.txt` (name files explicitly
     when needed).
3. Extend `tools/hf-safetensors/selection-policy.json` only for required files
   not already selected; avoid broad cache patterns.
4. Set ONNX regression mode:
   - `required` when the published `develop` compiler supports the baseline;
   - `informative` for candidate-only support, promoted after publication;
   - `disabled` only with a documented reason; it runs neither revision.
5. When case counts change, update `tests/README.md` and the matching
   `expected_test_count` in `.github/workflows/model-compiler-tests.yml`.
6. Verify complete output includes required runtime config and
   tokenizer/template/processor assets. Generate numerical artifacts during
   the run; never commit them. Missing inputs, cache misses, and unexpected
   skips fail.

## Completion Example

For a new Qwen3-VL-style VLM, report:

- source model ID and revision;
- VLM and source-layout routes selected;
- manifest/policy/count changes and temporary `informative` mode;
- config, processor, ONNX, graph, output-asset, and Modalix results;
- compilation options and artifact location;
- GGUF not supported because it was not implemented or validated; and
- supported-model documentation updated only for Hugging Face input.
