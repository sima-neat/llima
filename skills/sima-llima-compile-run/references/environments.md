# LLiMa Environments

Use CLI help and docs matching the installed release:

- `llima-compile --help`
- `llima-deploy --help`
- `llima run --help`
- `docs/setup.md`, `docs/compilation_genai.md`, `docs/deployment.md`,
  `docs/runtime.md`

## Compilation Host

Activate the Python 3.12 environment installed by Model Compiler, commonly:

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
