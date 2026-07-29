# LLiMa Environments

Use CLI help and docs matching the installed release:

- `llima-compile --help`
- `llima-deploy --help`
- `llima run --help`
- `docs/setup.md`, `docs/compilation_genai.md`, `docs/deployment.md`,
  `docs/runtime.md`

## Custom Fine-Tune Quantization Host

Skip this environment when compiling a published pre-quantized checkpoint.

For a custom fine-tune, use the exact matching SiMa.ai model repository's
`quantize.py` in a separate environment matching that repository's
`versions.txt`. Do not assume the Model Compiler Python environment has
compatible PyTorch, Transformers, llm-compressor, AutoRound, or
compressed-tensors versions.

Treat an NVIDIA CUDA GPU as required unless the exact repository explicitly
documents another supported environment. Confirm the GPU and Python stack
before starting:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python <recipe-directory>/quantize.py --help
```

Ensure enough GPU memory, host RAM, and storage for source weights,
calibration, compressed output, and temporary files. Never expose Hugging Face
tokens or private fine-tuned weights.

## Compilation Host

After selecting or producing the pre-quantized checkpoint, activate the Python
3.12 environment installed by Model Compiler, commonly:

1. `/sdk-extensions/model-compiler`
2. `/sdk-add-on/model-compiler`
3. `$HOME/sdk-extensions/model-compiler`

```bash
source <model-compiler-venv>/bin/activate
python --version
llima-compile --help
```

Install an approved LLiMa artifact there if needed; do not create another
environment with an incompatible Model Compiler. Ensure space for source
weights, ONNX, intermediates, and compiled output.

## Modalix

Confirm compatible device software and LLiMa runtime, mounted NVMe with enough
space, SSH reachability when deploying remotely, and available MLA resources.
Install an approved runtime if required:

```bash
sima-cli neat install llima@<version-or-ref>
llima run --help
```

Models default to `/media/nvme/llima/models`; an explicit path or
`LLIMA_MODELS_PATH` may override it. Compilation success is not runtime
validation; run on Modalix or report device access as unavailable.
