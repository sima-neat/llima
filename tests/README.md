# LLiMa test suite

The test suite is organized around compiler tests and DevKit runtime tests.
Compilation tests live under `tests/compilation/`. Runtime tests live under
`tests/runtime/`, are cross-compiled with the C++ runtime, and require a real
DevKit runtime environment. They do not fall back to host-only execution or
mock the hardware services.

The compilation workflow installs and tests the exact LLiMa compiler wheel for
the candidate commit. Tests run against model inputs downloaded from the
internal Vulcan cache; Hugging Face and Transformers are then placed in offline
mode.

## DevKit runtime tests

`./build.sh --all` cross-compiles the C++ runtime tests and packages them as
`dist/sima-lmm-<version>-Linux-extras.tar.gz`. The archive contains relocatable
CTest metadata under `lib/sima-lmm/tests/` and pytest sources under
`share/sima-lmm/tests/runtime/`.

The pytest coverage uses the installed package and command-line entry points.
It includes a real `llima` start/query/quit session rather than limiting CLI
validation to `llima --help`.

Runtime CI installs the exact LLiMa and Internals packages from the same build,
downloads the extras archive, prepares the same GenAI models used by Core, and
runs the tests serially against the real MLASHM dispatcher:

- `Qwen2.5-0.5B-Instruct-GPTQ-a16w4`
- `LFM2.5-VL-450M-a16w4`
- `whisper-small-a16w8`

The model root and names use the same environment contract as Core:
`LLIMA_MODELS_PATH`, `SIMA_TEST_LLIMA_TEXT_MODEL`,
`SIMA_TEST_LLIMA_VLM_MODEL`, and `SIMA_TEST_LLIMA_ASR_MODEL`.

These tests require an ARM64 Modalix DevKit and an active
`simaai-appcomplex.service`. Missing models, runtime packages, hardware
services, or the wrong architecture are failures rather than host-side skips.
CTest and pytest run in dedicated process groups; cancellation first sends
`SIGINT` to active inference and waits for its graceful shutdown.

Until mla-rt 2.1.3 fixes unreleased MLA buffers, every runtime case restarts
`simaai-appcomplex.service` before loading its model. This keeps cases isolated
without weakening the per-test daemon and CMA teardown assertions.

## CI test groups

### Run fast compiler unit tests

Location: `tests/compilation/unit/`  
Marker: `compiler_unit`  
Expected cases: 23

Fast, hermetic tests that run before downloading model inputs:

- Configuration parsing and validation
- Quantization configuration and precision selection
- Weight-name mapping
- Small pure-Python compiler checks

The unit-test fixtures remove model-cache environment variables, enable
Hugging Face offline mode, and reject network connections. These tests should
remain fast and independent of external model files.

### Download and verify cached models

This is workflow preparation rather than a pytest group. It downloads every
active source from the Hugging Face, configuration-only, and GGUF manifests
into a runner-local model directory.

The preparation step:

- Validates cache manifests and selection fingerprints
- Downloads files concurrently
- Verifies file sizes and SHA-256 digests
- Produces model-input provenance
- Exports `LLIMA_HF_MODELS_PATH`
- Enables `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`

All subsequent model-backed test groups use these prepared files. They must not
download models directly from Hugging Face.

For local runs, point `LLIMA_HF_MODELS_PATH` at a directory with the same
prepared layout:

```bash
export LLIMA_HF_MODELS_PATH=/path/to/llima-model-inputs
```

### Run configuration contract regression

Location: `tests/compilation/configuration/`  
Marker: `compiler_config`  
Expected cases: 25

Generates `VlmConfig` objects from cached Hugging Face and GGUF sources and
compares them with the checked-in JSON contracts under
`tests/compilation/configuration/references/`.

This group protects:

- Model and architecture detection
- Language, vision, and multimodal configuration
- Tensor dimensions and attention configuration
- Tokenizer and context configuration
- HF and GGUF configuration compatibility

JSON configuration contracts are intentionally stored in the repository.
Generated ONNX files and NumPy output fixtures are not.

### Run model-source ingestion tests

Location: `tests/compilation/source_ingestion/`  
Marker: `compiler_source`  
Expected cases: 15

Validates full-model GGUF source ingestion:

- GGUF parser detection for Q8_0 and Q4_0 inputs
- Dequantization of every tensor for Q8_0, Q4_0, Q6_K, Q5_K, Q4_K, and Q3_K
- Numerical comparison with both the BF16 GGUF reference and the GGUF library
- Resolution of every Hugging Face weight name to a GGUF weight
- Shape agreement between Hugging Face and GGUF weights

The comparisons are exhaustive across the model tensors. Reference weights are
shared across quantization variants where possible to avoid redundant loading;
the validation coverage remains unchanged.

### Run ONNX generation and validation

Location: `tests/compilation/onnx_regression/`  
Marker: `compiler_onnx_regression`  
Expected cases: 32

Generates representative ONNX model components with the candidate LLiMa
compiler wheel. Feature branches also generate the same components with the
latest published `develop` compiler wheel and compare the results.

The validation mode is selected as follows:

- Feature branch push, except `release*`: compare with the latest published
  `develop` compiler wheel
- `develop`, `main`, `release*`, and tag pushes: candidate-only validation with
  no baseline artifact

The matrix covers representative pre, cache, post, per-layer, convolution,
vision, and speculative-decoding components. Every case:

1. Generates candidate ONNX.
2. Runs `onnx.checker`.
3. Creates deterministic inputs.
4. Executes the candidate graph with ONNX Runtime.
5. Validates runtime output count, dtype, rank, and fixed dimensions against
   the graph interface.

In compare mode, the test additionally generates and executes baseline ONNX,
compares the runtime input/output interface, and compares numerical outputs.

Each case has a regression mode in `tests/compilation/cases.py`:

- `required`: candidate and baseline generation must succeed, and any interface
  or numerical difference fails the test.
- `informative`: candidate generation, checking, and execution must succeed. If
  the baseline compiler does not support the model or component, the baseline
  manifest records it as unavailable and the test emits a warning. When both
  graphs are available, interface or numerical differences also emit warnings.
- `disabled`: neither revision generates or executes the case.

This allows a feature branch to validate newly added model support before that
support exists in the published `develop` baseline. The case should change from
`informative` to `required` once baseline support is available.

ONNX payloads are generated during the workflow and deleted afterward. No
reference ONNX files are stored in the repository or artifact cache.

### Run generated-graph and quantization integration

Location: `tests/compilation/graph_integration/`  
Marker: `compiler_graph_integration`  
Expected cases: 22 standard and 4 high-memory

Validates generated SiMa Model SDK graphs and quantization behavior:

- `test_embedding_quantization.py`: embedding quantization and dequantization
  wiring for LLM, VLM, and per-layer inputs
- `test_path_equivalence.py`: staged
  source-to-ONNX-to-quant generation versus direct source-to-quant generation
- `test_gguf_integration.py`: generated GGUF-based quantized graphs versus
  corresponding Hugging Face or BF16 source graphs
- `test_speculative_decoding.py`: staged versus direct generation for
  speculative pre, cache, post, and draft-FC graphs

The speculative-decoding cases are marked `serial` and `high_memory`. The
workflow runs the 22 standard cases first and the 4 high-memory cases
separately.

### Run selected-model full compilation E2E

Location: `tests/compilation/e2e/`  
Marker: `compiler_e2e`  
Expected cases: 1

Performs one bounded full compiler pipeline:

1. Selects an eligible model deterministically from the candidate commit SHA.
2. Resolves and validates its cached source provenance.
3. Generates ONNX.
4. Quantizes the selected model component.
5. Invokes Model Compiler.
6. Validates the resulting MPK archive and MLA ELF.

The compile configuration selects layer 0 and INT4 quantization so the test
exercises the complete pipeline without compiling an entire generative model.
The case is marked `serial` and `high_memory`.

## Test definitions and shared helpers

- `tests/compilation/cases.py` contains the centralized model and component
  matrices.
- `tests/compilation/conftest.py` resolves the prepared model-input directory.
- `tests/compilation/helpers/` contains shared model-loading, path-validation,
  and output-comparison utilities.
- `pytest.ini` declares the test markers used by the workflow.

The workflow audits the expected case count and rejects unexpected skips. When
adding or removing a case, update both the centralized matrix and the
corresponding expected count in
`.github/workflows/model-compiler-tests.yml`.

## Running groups locally

Activate a Model Compiler environment and install the LLiMa compiler wheel with
its test dependencies before running these commands.

Fast hermetic tests:

```bash
python -P -m pytest \
  -c pytest.ini \
  tests/compilation/unit \
  -m compiler_unit \
  --strict-markers \
  -vv -ra
```

Cached-model test groups:

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

ONNX regression additionally requires separately generated baseline and
candidate ONNX roots and manifests. The authoritative invocation is maintained
in `.github/workflows/model-compiler-tests.yml`.

## Reference-artifact policy

- Checked-in JSON configuration contracts are allowed.
- Checked-in or externally stored reference ONNX and NumPy output files are
  not used.
- Numerical regression compares artifacts generated from the candidate branch
  with artifacts generated from the appropriate baseline branch during the
  same workflow run.
- Generated ONNX, quantization, and compilation payloads are temporary.
