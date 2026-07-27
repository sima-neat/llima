# LLiMa Model Inputs

## Select the Source Format

LLiMa supports these GenAI source families:

- Hugging Face safetensors for supported LLMs and VLMs;
- supported compressed-tensor GPTQ, AWQ, or AutoRound-style safetensor LLMs
  and VLMs; and
- GGUF for supported LLMs.

GGUF is not the default VLM source path. Check the current supported-model and
limitations tables before proceeding.

Do not use this skill for a generic ONNX computer-vision model.

## Compare Input-Format Tradeoffs

Treat these as typical characteristics, not performance or accuracy
guarantees:

| Model input | Typical result | Main tradeoff |
|----|----|----|
| Hugging Face BF16/FP safetensors, quantized by LLiMa | Fast Modalix runtime using LLiMa's native INT8 or INT4 weight quantization | Quantization, especially INT4, can degrade model accuracy; validate representative prompts and images |
| Prequantized GPTQ, AWQ, or AutoRound model in LLiMa's supported compressed-tensors format with block/group size 128 or 256 | Usually combines fast runtime with high accuracy because the weights were quantized with an accuracy-aware method before compilation | Availability and compatibility are model-specific; the published layout and quantization settings must be supported |
| GGUF | Convenient when a supported GGUF is already available and avoids quantizing the original weights again | Runtime is normally slower than the native LLiMa or supported prequantized path because common Q4_0/Q8_0 GGUF weights use block quantization with a block size of 32 |

Prefer a compatible prequantized GPTQ, AWQ, or AutoRound model when one is
available and has passed model-quality validation. Otherwise, start from the
original Hugging Face safetensors and choose LLiMa INT8 or INT4 based on the
required accuracy, memory, and performance. Use GGUF when availability or
convenience matters more than peak runtime performance.

Do not infer compatibility from `GPTQ`, `AWQ`, or `AutoRound` in the repository
name. For the current direct prequantized path, require the Hugging Face
`quantization_config` to use the `compressed-tensors` format with supported
symmetric 4-bit or 8-bit weights. Expect the fast runtime path only for
block/group size 128 or 256; measure other supported configurations rather
than assuming equivalent performance. Reject unsupported layouts or
asymmetric zero points instead of silently dequantizing or requantizing them.

GGUF quantization varies by file. The block-size statement applies directly to
the commonly used Q4_0 and Q8_0 formats; other GGUF formats may be converted by
LLiMa or fall back to dequantization. Inspect the exact GGUF quantization type
rather than relying only on the filename.

## Check Support Before Download

Read the installed or source-controlled `docs/index.md` and
`docs/compilation_genai.md` for:

- supported architectures and model sizes;
- required tokenizer, processor, configuration, and weight files;
- source-format limitations;
- vision input requirements;
- supported quantization options; and
- architecture-specific compiler flags.

Fail early when the architecture or source format is unsupported. Do not
silently choose a similarly named model.

## Resolve Hugging Face Access

Prefer:

1. an existing complete local model directory;
2. an approved organization cache at an immutable revision; or
3. a direct Hugging Face download when authorization permits it.

For gated models, the account or service token must have access to that exact
repository. Never print a token or embed it in a URL. Pass authentication
through the standard Hugging Face environment or credential store.

When a direct download is required, pin a known revision:

```bash
hf download <organization/model> \
  --revision <immutable-revision> \
  --local-dir <model-directory>
```

Record the public model ID and immutable revision when available. Do not copy
private weights, customer data, or token values into reports.

## Choose Compilation Options

Start with the documented default configuration. Add options only for a stated
need, such as:

- context length;
- prefill group size;
- BF16, INT8, or INT4 layer policy;
- VLM image dimensions;
- embedding quantization;
- LoRA; or
- speculative decoding.

Use a reviewed configuration file for per-layer policy. Record the selected
policy because it affects accuracy, memory, compilation time, TTFT, and TPS.

## Expected Compiler Output

A deployable compiler output must include:

```text
<output>/
└── sima_files/
    ├── devkit/
    ├── mpk/
    └── npy_files/  # optional, for supported LoRA workflows
```

Generated `onnx_files/` and other compiler intermediates remain host-side.
Deploy the runtime material through `llima-deploy`; do not manually treat the
whole compiler working directory as the Modalix runtime model.
