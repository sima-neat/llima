# Pre-Quantized Checkpoints and Custom Fine-Tunes

## Collection Role

Use the
[SiMa.ai Pre-Quantized Models collection](https://huggingface.co/collections/simaai/pre-quantized-models)
as the mandatory first lookup. The collection indexes independent,
model-specific repositories. It does not provide one generic quantization
script.

Require an exact match for:

- architecture and generation;
- parameter size;
- base versus instruction variant; and
- LLM versus VLM, including the vision encoder/projector contract.

If no exact repository exists, report the gap. Do not select the nearest
model's script. Use FP/BF16 or GGUF only for an explicit requirement or with
user agreement.

## Published Pre-Quantized Checkpoint

Pin and download the matching compressed-tensors checkpoint, then pass that
directory directly to `llima-compile`:

```bash
hf download <simaai/model-repository> \
  --revision <immutable-revision> \
  --local-dir <prequantized-model-directory>

llima-compile <prequantized-model-directory> -o <compiled-output>
```

Treat the checkpoint's `quantization_config`, model card, `recipe.yaml`, and
`versions.txt` as authoritative. Do not add a generic precision configuration
that requantizes or contradicts the checkpoint.

## Custom Fine-Tune

Use the exact matching pre-quantized model repository as a recipe carrier:

1. Pin the repository revision.
2. Read its model card, `quantize.py`, `recipe.yaml`, and `versions.txt`.
3. Use a full, locally loadable Hugging Face fine-tuned checkpoint with the
   same architecture, size, variant, and modality.
4. Create a GPU environment matching `versions.txt`.
5. Run that repository's unmodified `quantize.py` using only flags exposed by
   the script.
6. Validate the compressed output, then compile it with LLiMa.

Download the model-specific recipe files without downloading its weights when
the installed `hf download` supports positional filenames:

```bash
hf download <simaai/model-repository> \
  quantize.py recipe.yaml versions.txt \
  --revision <immutable-revision> \
  --local-dir <recipe-directory>

python <recipe-directory>/quantize.py \
  --model-path <custom-finetuned-model-directory> \
  --output-dir <quantized-output-directory>

llima-compile <quantized-output-directory> -o <compiled-output>
```

Use `hf download --help` and `python quantize.py --help` as the command
contracts. Do not infer additional flags.

The script owns the quantization method, target module names, exclusions,
weight precision, group sizes, calibration dataset, sample count, sequence
length, and validation. Do not translate it into a shared GPTQ/AutoRound
recipe or modify it to resemble another model.

Treat an NVIDIA CUDA GPU as required unless that exact model repository
explicitly documents another supported execution environment. Confirm enough
GPU memory, host RAM, and storage before starting. Quantization may be lengthy
and must remain separate from the Python 3.12 Model Compiler environment when
the repository's `versions.txt` specifies different dependencies.

## Quantized Output Checks

Before `llima-compile`, require:

- safetensor weights and the model configuration;
- tokenizer files and processor files for a VLM;
- a supported compressed-tensors `quantization_config`;
- the emitted `quantize.py`, `recipe.yaml`, and `versions.txt` when the script
  preserves them;
- successful finite-scale or other validation performed by the script; and
- a load/generation smoke test when documented by the repository.

Record both source revisions: the custom fine-tune revision and the matching
SiMa.ai recipe repository revision. Validate quality on representative
domain-specific prompts and images after quantization and again on Modalix.
