# Host reference runtime

The host reference runtime executes the generated language-model parts in the same normal
prefill/decode order as the C++ runtime. It can execute either ONNX parts or the quantized `.sima`
graphs, which makes it useful for separating graph-generation errors from C++/MLA runtime errors.

The reference path mirrors the C++ behavior for:

- group-prefill scheduling and single-token decode;
- the full group cache followed by the final valid-row `n1` post model;
- full and sliding-attention cache selection;
- normal, long-context, group, and sliding future-token masks;
- quantized embedding inputs and quantized KV caches;
- shared-KV source layers; and
- split LM-head output handling.

Speculative decoding, recurrent/conv layers, Gemma4 per-layer embeddings, and Qwen3-VL deepstack
inputs are rejected explicitly until their C++ orchestration is represented in the reference
runtime. This prevents an incomplete reference execution from being mistaken for a runtime
comparison.

## Quantized `.sima` example

```bash
source ../afe_env/bin/activate

python scripts/run_reference_model.py \
  /path/to/source-model \
  /path/to/compiled-model/sima_files \
  "Why is the sky blue?" \
  --max-new-tokens 1 \
  --use-jax
```

The script reads compilation options from `devkit/vlm_config.json`, so group size, context length,
embedding quantization, KV-cache quantization, and split-MLP settings match the compiled graphs.

The NumPy evaluator remains the default because JAX 0.4.30 can abort on ARM Linux. On supported
AMD64 hosts, `--use-jax` sets `SIMA_LMM_SDK_EVAL_USE_JAX=1` and substantially reduces reference
runtime. Split LM-head post graphs remain on NumPy because the same JAX version cannot execute
their multi-output graph reliably.

Use `--mode onnx --onnx-path /path/to/onnx` to execute generated ONNX parts instead.
