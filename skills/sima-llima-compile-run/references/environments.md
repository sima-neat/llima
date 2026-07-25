# LLiMa Environments

## Source of Truth

Prefer the installed LLiMa CLI help and documentation that matches the current
environment:

- `llima-compile --help`
- `llima-deploy --help`
- `llima run --help`
- `docs/setup.md`
- `docs/compilation_genai.md`
- `docs/deployment.md`
- `docs/runtime.md`

When a LLiMa source checkout is available, read its current documentation
before using examples from another release.

## Compilation Environment

Compile in the Python 3.12 virtual environment installed by Model Compiler.
Common locations are:

1. `/sdk-extensions/model-compiler`
2. `/sdk-add-on/model-compiler`
3. `$HOME/sdk-extensions/model-compiler`

Activate the environment that exists on the system:

```bash
source <model-compiler-venv>/bin/activate
python --version
llima-compile --help
```

If LLiMa is not installed in that environment, install an approved published
artifact or follow the current LLiMa setup documentation. Do not install a
second incompatible copy of Model Compiler into another virtual environment.

Before compiling, confirm adequate local space for the source model, generated
ONNX, compiler intermediates, and compiled output.

## Runtime Environment

Run compiled models on a compatible Modalix DevKit. Confirm:

- the LLiMa runtime is installed;
- the device software is compatible with the Model Compiler version;
- NVMe storage is mounted and has enough free space;
- the device is reachable over SSH when deploying remotely; and
- another process is not unexpectedly holding required MLA resources.

If the runtime is missing, install an approved compatible artifact on Modalix:

```bash
sima-cli neat install llima@<version-or-ref>
llima run --help
```

The default managed-model location is `/media/nvme/llima/models`. An explicit
model path is also accepted. `LLIMA_MODELS_PATH` can select another managed
model root.

Compilation success on the host is not runtime validation. Complete a Modalix
smoke run unless device access is unavailable, and report that limitation
explicitly.
