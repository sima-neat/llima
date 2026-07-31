# LLiMa Model Inputs

## Required Selection Order

Always check the
[SiMa.ai Pre-Quantized Models collection](https://huggingface.co/collections/simaai/pre-quantized-models)
first. It is an index of model-specific repositories, not a shared
quantization workflow.

1. Prefer an exact matching pre-quantized checkpoint directly by default.
2. For a custom fine-tune, use the exact matching pre-quantized model
   repository's own `quantize.py` as described in
   `references/prequantized-models.md`, then compile its output.
3. Use original FP/BF16 Hugging Face weights or GGUF for an explicit
   requirement—such as maximum fidelity, source-weight reproduction, or a
   mandated format—or with user agreement.
4. When no exact pre-quantized match exists, report the gap before agreeing on
   an FP/BF16 or GGUF fallback.

Match architecture, size, base/instruction variant, and modality. For VLMs,
also match the vision encoder and projector contract. Never substitute the
nearest model or reuse its script.

## Formats and Tradeoffs

Supported source families:

- Hugging Face safetensors for supported LLMs/VLMs;
- supported compressed-tensors GPTQ/AutoRound-style safetensors; and
- GGUF for supported LLMs.

| Input | Typical use | Main tradeoff |
| --- | --- | --- |
| Supported prequantized compressed-tensors, symmetric 4/8-bit, group size 128/256 | Default and preferred LLiMa compiler input | Requires an exact compatible model-specific layout |
| HF BF16/FP, quantized by LLiMa | Explicit fidelity/source requirement or agreed fallback | Requires the additional LLiMa quantization stage; quantization may reduce accuracy |
| GGUF | Convenient existing quantized LLM | Common Q4_0/Q8_0 block-32 formats are usually slower than native/prequantized paths |

Collection checkpoints are pre-LLiMa quantized Hugging Face artifacts, not
compiled Modalix runtime models. They still require `llima-compile`.

Do not infer support from `GPTQ` or `AutoRound` in a repository name.
Inspect `quantization_config`: the direct path requires supported
compressed-tensors layout, symmetric 4/8-bit weights, and no unsupported zero
points. The matching repository's script and `recipe.yaml` define the
supported targets, exceptions, method, group size, and calibration contract.
Inspect each GGUF quantization type; non-Q4_0/Q8_0 formats may convert or
dequantize differently.

## Check Support First

Read `docs/index.md` and `docs/compilation_genai.md` for supported
architectures/sizes, formats, required assets, vision shapes, quantization, and
flags. Reject unsupported input rather than choosing a similarly named model.
GGUF is not the VLM path.

## Resolve Access

Prefer:

1. a complete local directory;
2. an approved immutable organization cache; or
3. an authorized direct Hugging Face download.

Absent an explicit source, fidelity, or format requirement, preserve the
required selection order: exact pre-quantized checkpoint first,
repository-specific quantization for a custom fine-tune second, and FP/BF16 or
GGUF fallback last.

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

Start with the pre-quantized checkpoint's encoded precision. Do not replace it
with a generic mixed-precision policy. Add context length, group size, vision
size, or LoRA only for a stated need; record choices because they affect
accuracy, size, compilation time, TTFT, and TPS.

Speculative decoding is unsupported by this workflow until issue #128 lands.
Do not pass a draft model or validate speculative output as one
`sima_files/` tree: target and draft compilation produce separate artifact
trees.

A deployable output contains:

```text
<output>/sima_files/
├── devkit/
├── mpk/
└── npy_files/  # optional LoRA material
```

Keep `onnx_files/` and compiler intermediates on the host. Deploy with
`llima-deploy`; do not copy the entire compiler workspace as a runtime model.
