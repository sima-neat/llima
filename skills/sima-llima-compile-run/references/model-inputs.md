# LLiMa Model Inputs

## Formats and Tradeoffs

Supported source families:

- Hugging Face safetensors for supported LLMs/VLMs;
- supported compressed-tensors GPTQ/AutoRound-style safetensors; and
- GGUF for supported LLMs.

| Input | Typical use | Main tradeoff |
| --- | --- | --- |
| HF BF16/FP, quantized by LLiMa | Native INT8/INT4 runtime path | Quantization may reduce accuracy |
| Supported prequantized compressed-tensors, symmetric 4/8-bit, group size 128/256 | Often strong speed/accuracy balance | Layout and quantization compatibility are model-specific |
| GGUF | Convenient existing quantized LLM | Common Q4_0/Q8_0 block-32 formats are usually slower than native/prequantized paths |

Check the
[SiMa.ai Pre-Quantized Models collection](https://huggingface.co/collections/simaai/pre-quantized-models)
first. It provides GPTQ/AutoRound safetensor checkpoints for most supported
models; prefer an exact compatible match when available. These are
pre-quantized compiler inputs, not compiled Modalix runtime models, so they
still require `llima-compile`.

If the collection has no compatible checkpoint, start from original HF weights
and choose LLiMa INT8/INT4 by accuracy, memory, and performance. Use GGUF when
availability outweighs peak performance.

Do not infer support from `GPTQ` or `AutoRound` in a repository name.
Inspect `quantization_config`: the direct path requires supported
compressed-tensors layout, symmetric 4/8-bit weights, and no unsupported zero
points. Measure group sizes other than 128/256. Inspect each GGUF quantization
type; non-Q4_0/Q8_0 formats may convert or dequantize differently.

## Check Support First

Read `docs/index.md` and `docs/compilation_genai.md` for supported
architectures/sizes, formats, required assets, vision shapes, quantization, and
flags. Reject unsupported input rather than choosing a similarly named model.
GGUF is not the default VLM path.

## Resolve Access

Prefer:

1. a complete local directory;
2. an approved immutable organization cache; or
3. an authorized direct Hugging Face download.

For gated models, use the standard Hugging Face credential store/environment;
never print or embed tokens. Pin direct downloads:

```bash
hf download <organization/model> \
  --revision <immutable-revision> \
  --local-dir <model-directory>
```

Record public model ID/revision, not credentials, private weights, or customer
data.

## Options and Output

Start with defaults. Add context length, group size, precision, vision size,
embedding quantization, LoRA, or speculative decoding only for a stated need;
record choices because they affect accuracy, size, compilation time, TTFT, and
TPS.

A deployable output contains:

```text
<output>/sima_files/
├── devkit/
├── mpk/
└── npy_files/  # optional LoRA material
```

Keep `onnx_files/` and compiler intermediates on the host. Deploy with
`llima-deploy`; do not copy the entire compiler workspace as a runtime model.
