# LLiMa test suite

## Overview

LLiMa has two CI test paths with different artifacts and execution
environments. Vulcan CI orchestrates both paths: compiler tests are delegated
to a reusable workflow, while runtime tests run in a dedicated DevKit job.

| Area | CI entry point | Runner | Tested artifact | Hardware |
|------|----------------|--------|-----------------|----------|
| Model compiler | `.github/workflows/model-compiler-tests.yml`, called by `test-model-compiler` in `vulcan-ci.yml` | Host test runner | Exact compiler wheel for the candidate commit | No DevKit |
| Runtime | `test-devkit` in `.github/workflows/vulcan-ci.yml` | ARM64 Modalix runner | Exact LLiMa and Internals DEBs plus the runtime-test extras archive | Real DevKit and MLASHM dispatcher |

Compiler tests live under `tests/compilation/`. Runtime tests live under
`tests/runtime/`. Runtime coverage never falls back to host-only execution or
mock hardware services.

## Model compiler CI

### Purpose

The compiler workflow validates the exact LLiMa compiler wheel produced for
the candidate commit. It covers configuration, model ingestion, generated
ONNX, quantization, and a bounded full compilation pipeline.

Model inputs are downloaded from the internal Vulcan cache. After preparation,
Hugging Face and Transformers run in offline mode so test groups cannot
silently fetch additional inputs.

### Test inputs and model cache

The workflow downloads every active source from the Hugging Face,
configuration-only, and GGUF manifests into a runner-local model directory.

The preparation step:

- Validates cache manifests and selection fingerprints.
- Downloads files concurrently.
- Verifies file sizes and SHA-256 digests.
- Produces model-input provenance.
- Exports `LLIMA_HF_MODELS_PATH`.
- Enables `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

All model-backed compiler groups use these prepared files. For local runs,
point `LLIMA_HF_MODELS_PATH` at a directory with the same layout:

```bash
export LLIMA_HF_MODELS_PATH=/path/to/llima-model-inputs
```

### Test groups

#### Fast compiler unit tests

- Location: `tests/compilation/unit/`
- Marker: `compiler_unit`
- Expected cases: 23

Fast, hermetic tests that run before model inputs are downloaded:

- Configuration parsing and validation.
- Quantization configuration and precision selection.
- Weight-name mapping.
- Small pure-Python compiler checks.

The fixtures remove model-cache environment variables, enable Hugging Face
offline mode, and reject network connections.

#### Configuration contract regression

- Location: `tests/compilation/configuration/`
- Marker: `compiler_config`
- Expected cases: 25

Generates `VlmConfig` objects from cached Hugging Face and GGUF sources and
compares them with the checked-in JSON contracts under
`tests/compilation/configuration/references/`.

This protects:

- Model and architecture detection.
- Language, vision, and multimodal configuration.
- Tensor dimensions and attention configuration.
- Tokenizer and context configuration.
- Hugging Face and GGUF configuration compatibility.

#### Model-source ingestion

- Location: `tests/compilation/source_ingestion/`
- Marker: `compiler_source`
- Expected cases: 15

This group validates:

- GGUF parser detection for Q8_0 and Q4_0 inputs.
- Dequantization for Q8_0, Q4_0, Q6_K, Q5_K, Q4_K, and Q3_K.
- Numerical comparison with the BF16 GGUF reference and GGUF library.
- Resolution of Hugging Face weight names to GGUF weights.
- Shape agreement between Hugging Face and GGUF weights.

The comparisons are exhaustive across model tensors. Reference weights are
shared across quantization variants where possible.

#### ONNX generation and validation

- Location: `tests/compilation/onnx_regression/`
- Marker: `compiler_onnx_regression`
- Expected cases: 32

Every case:

1. Generates candidate ONNX.
2. Runs `onnx.checker`.
3. Creates deterministic inputs.
4. Executes the graph with ONNX Runtime.
5. Validates output count, dtype, rank, and fixed dimensions.

Feature branches, except `release*`, also generate the same components with
the latest published `develop` compiler wheel. `develop`, `main`, `release*`,
and tag builds validate the candidate only because those refs do not have a
separate baseline artifact.

Each case declares one regression mode in `tests/compilation/cases.py`:

- `required`: candidate and baseline generation must succeed; interface or
  numerical differences fail.
- `informative`: candidate generation and execution must succeed. Missing
  baseline support or differences emit warnings.
- `disabled`: neither revision generates or executes the case.

Use `informative` for new model support that is not yet available from the
published `develop` compiler. Change it to `required` once baseline support is
available.

Generated ONNX and NumPy payloads are deleted after the workflow. They are not
stored in the repository or artifact cache.

#### Generated-graph and quantization integration

- Location: `tests/compilation/graph_integration/`
- Marker: `compiler_graph_integration`
- Expected cases: 22 standard and 4 high-memory

This group validates:

- Embedding quantization and dequantization wiring.
- Staged source-to-ONNX-to-quant generation versus direct generation.
- GGUF-generated quantized graphs versus Hugging Face or BF16 source graphs.
- Speculative pre, cache, post, and draft-FC graph generation.

The speculative-decoding cases are serial and high-memory. CI runs the 22
standard cases first and the 4 high-memory cases separately.

#### Selected-model full compilation E2E

- Location: `tests/compilation/e2e/`
- Marker: `compiler_e2e`
- Expected cases: 1

This bounded end-to-end case:

1. Selects an eligible model deterministically from the candidate commit SHA.
2. Resolves and validates cached source provenance.
3. Generates ONNX.
4. Quantizes the selected model component.
5. Invokes Model Compiler.
6. Validates the resulting MPK archive and MLA ELF.

The test selects layer 0 and INT4 quantization to cover the complete pipeline
without compiling an entire generative model. It is serial and high-memory.

### Compiler test definitions and helpers

- `tests/compilation/cases.py` contains the model and component matrices.
- `tests/compilation/conftest.py` resolves the prepared model-input directory.
- `tests/compilation/helpers/` contains model-loading, path-validation, and
  output-comparison helpers.
- The root `pytest.ini` declares compiler markers and makes the compiler
  premerge suite the safe repository default. Its `testpaths` excludes the
  DevKit runtime suite.

The workflow audits expected case counts and rejects unexpected skips. Update
both the centralized matrix and the expected count in
`.github/workflows/model-compiler-tests.yml` when adding or removing cases.

### Running compiler tests locally

Activate a Model Compiler environment and install the LLiMa compiler wheel
with its test dependencies.

Fast hermetic tests:

```bash
python -P -m pytest \
  -c pytest.ini \
  tests/compilation/unit \
  -m compiler_unit \
  --strict-markers \
  -vv -ra
```

Cached-model groups:

```bash
export LLIMA_HF_MODELS_PATH=/path/to/llima-model-inputs

python -P -m pytest \
  -c pytest.ini \
  tests/compilation/configuration \
  -m compiler_config \
  --strict-markers \
  -vv -ra

python -P -m pytest \
  -c pytest.ini \
  tests/compilation/source_ingestion \
  -m compiler_source \
  --strict-markers \
  -vv -ra

python -P -m pytest \
  -c pytest.ini \
  tests/compilation/graph_integration \
  -m "compiler_graph_integration and not high_memory" \
  --strict-markers \
  -vv -ra
```

ONNX regression additionally requires separately generated candidate and
baseline roots and manifests. The authoritative invocation is maintained in
`.github/workflows/model-compiler-tests.yml`.

### Compiler reference-artifact policy

- Checked-in JSON configuration contracts are allowed.
- Reference ONNX and NumPy output files are not checked in or stored
  externally.
- Numerical regression compares candidate and baseline artifacts generated
  during the same workflow run.
- Generated ONNX, quantization, and compilation payloads are temporary.

## DevKit runtime CI

### Purpose and requirements

Runtime CI validates installed artifacts through the same public entry points
used on a Modalix DevKit. It requires:

- ARM64 Modalix hardware.
- An active `simaai-appcomplex.service`.
- The MLASHM dispatcher and MLA hardware.
- Exact LLiMa and Internals packages from the candidate build.

Missing models, packages, services, or the wrong architecture are failures,
not skips or host-side fallbacks.

### Runtime artifacts

`./build.sh --all` cross-compiles the C++ tests and creates:

```text
dist/sima-lmm-<version>-Linux-extras.tar.gz
```

The extras archive contains:

- Relocatable test executables and CTest metadata under
  `lib/sima-lmm/tests/`.
- Pytest sources and runtime helpers under
  `share/sima-lmm/tests/runtime/`.

The archive does not duplicate runtime libraries or media. Tests link against
the installed runtime and use the image and audio assets installed by
`sima-lmm-core`.

### GenAI model preparation

The dedicated `Prepare GenAI models` step keeps the runtime fixtures aligned
with Core:

- `Qwen2.5-0.5B-Instruct-GPTQ-a16w4`
- `LFM2.5-VL-450M-a16w4`
- `whisper-small-a16w8`

The shared environment contract is:

- `LLIMA_MODELS_PATH`
- `SIMA_TEST_LLIMA_TEXT_MODEL`
- `SIMA_TEST_LLIMA_VLM_MODEL`
- `SIMA_TEST_LLIMA_ASR_MODEL`

Models are retained on the runner between CI runs.

### C++ runtime tests

CTest executes serially with a dispatcher resource lock:

| Test | Coverage |
|------|----------|
| `runtime.dispatcher_lifecycle` | Connect and disconnect through the installed dispatcher stack |
| `runtime.text_generation` | Qwen text generation on MLA |
| `runtime.vision_generation` | LFM2 image-conditioned generation using the installed sample image |
| `runtime.asr_transcription` | Whisper transcription using the installed sample audio |

The executables link directly against the in-tree runtime while building, then
use install RPATHs to load the installed runtime and dispatcher libraries on
the DevKit.

### Python and black-box tests

Pytest validates the installed package and external interfaces:

| Test | Coverage |
|------|----------|
| Installed Python lifecycle | Imports the installed extension and connects to and disconnects from the dispatcher |
| CLI black box | Starts `llima`, submits a real Qwen query, validates the answer, sends `quit`, and verifies teardown |
| OpenAI-compatible HTTP | Non-streaming response, SSE reconstruction, `/stop`, malformed-input recovery, and a successful request after interruption |
| ZMQ black box | CURVE-secured MessagePack request, generated tensor response, and remote server shutdown |

`tests/runtime/pytest.ini` is packaged with these tests and prevents compiler
marker defaults from affecting runtime test selection.

Speculative-decoding runtime coverage is intentionally deferred until a
compiled target/draft model pair is published.

### Lifecycle and teardown assertions

The CLI lifecycle test establishes a steady-state baseline and checks that
subsequent execution:

- Keeps the `mlashmcomplex` daemon PID stable.
- Does not grow daemon thread or file-descriptor counts.
- Reaps the `llima` process.
- Returns CMA memory within the configured tolerance.

### Cancellation and test isolation

CTest and pytest run in dedicated process groups. Cancellation sends only
`SIGINT` to active inference and waits for graceful shutdown; it never
escalates to `SIGTERM` or `SIGKILL`.

mla-rt versions before 2.1.3 can retain MLA buffers between processes. Until
mla-rt 2.1.3 is available, every runtime case restarts
`simaai-appcomplex.service` before loading its model. Remove this workaround
when the fixed runtime is adopted.

### Cleanup and retained state

An `if: always()` workflow step removes:

- `_work/llima-install`
- `_work/runtime-tests`
- `_work/runtime-test-venv`

Runtime test dependency installation uses `pip --no-cache-dir`.

The runner deliberately retains:

- Installed LLiMa and Internals Debian packages, which the next run
  overwrites.
- Downloaded Qwen, LFM2, and Whisper models.
- The running `simaai-appcomplex.service`.

Runtime tests do not upload reports or generated outputs after execution.

### Running runtime tests on a DevKit

Build and install the candidate LLiMa and Internals packages, then extract the
extras archive:

```bash
mkdir -p _work/runtime-tests
tar -C _work/runtime-tests \
  -xzf dist/sima-lmm-<version>-Linux-extras.tar.gz
```

Prepare models:

```bash
bash _work/runtime-tests/share/sima-lmm/tests/runtime/prepare_genai_models.sh
```

Run C++ tests serially:

```bash
_work/runtime-tests/share/sima-lmm/tests/runtime/run_with_cancellation_cleanup.sh \
  ctest \
    --test-dir _work/runtime-tests/lib/sima-lmm/tests \
    --output-on-failure \
    --no-tests=error \
    --timeout 900 \
    -j 1 \
    -L devkit
```

Run Python tests from a system-site-packages venv:

```bash
python3 -m venv --system-site-packages _work/runtime-test-venv
_work/runtime-test-venv/bin/python -m pip install --no-cache-dir \
  'msgpack>=1.1,<2' \
  'pyzmq>=27.1,<28' \
  'pytest>=8,<9'

(
  cd _work/runtime-tests
  share/sima-lmm/tests/runtime/run_with_cancellation_cleanup.sh \
    ../runtime-test-venv/bin/python \
      -m pytest \
        -c share/sima-lmm/tests/runtime/pytest.ini \
        -q \
        share/sima-lmm/tests/runtime
)
```

### Adding a runtime test

When adding runtime coverage:

1. Use a real DevKit, installed packages, and the dispatcher path.
2. Add C++ targets and extras installation rules in
   `tests/runtime/CMakeLists.txt`, or package a pytest under
   `tests/runtime/`.
3. Keep model names and environment variables aligned with Core.
4. Add a bounded timeout and dispatcher serialization.
5. Assert observable behavior and teardown, not only process startup.
6. Update the extras archive validation in `vulcan-ci.yml`.
7. Do not add post-test uploads.

## Shared CI rules

### Failure and skip policy

- Required dependencies and fixtures fail loudly when missing.
- Runtime tests do not skip because a DevKit, service, or model is absent.
- Compiler case counts are audited and unexpected skips fail the workflow.
- New compiler support may use informative ONNX comparison until it exists in
  the published `develop` baseline.

### Provenance

- Compiler tests install the exact candidate wheel.
- Runtime tests install the exact candidate LLiMa packages and resolved
  Internals packages.
- Build metadata and manifests tie packages and test extras to the candidate
  commit.

### Execution policy

- Runtime tests execute serially against the dispatcher.
- High-memory compiler cases execute separately from the standard matrix.
- Generated compiler payloads and extracted runtime-test workspaces are
  temporary.
