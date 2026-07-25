# LLiMa Compiler CI/CD Plan

Status: implemented; pending Vulcan CI validation and team review

## Scope

This document defines the intended test structure for the LLiMa compilation
toolchain. It covers model configuration, source-model ingestion, ONNX
generation, graph generation, quantization, and compilation.

Runtime validation on a SiMa DevKit is a separate future workflow and is not
part of this plan.

The test suite must be organized around the failure class each test detects.
Model architecture, source format, generated component, precision, and
features such as speculative decoding are dimensions of the test matrix; they
are not independent top-level test groups.

## Reference and baseline policy

Compiler CI must not persist generated model outputs as regression references.
In particular, the following must not be stored in Git, GitHub Actions
artifacts, Vulcan, or a shared filesystem:

- ONNX reference graphs;
- NumPy `.npy` or `.npz` reference outputs;
- quantized intermediate-output baselines;
- compiled-model reference artifacts; or
- mutable binary reference directories.

The following persistent inputs are allowed:

- reviewed JSON reference configurations;
- source Hugging Face and GGUF models cached at immutable revisions;
- source-controlled test cases, tolerances, seeds, and policies; and
- JUnit, timing, coverage, and human-readable comparison reports.

Generated ONNX, quantized graphs, compiler outputs, and test tensors must live
in runner-local temporary directories and be discarded after the job.

For branch-relative regression, every push generates the baseline during the
same CI run from an immutable wheel resolved through Vulcan:

- a feature-branch push compares against the latest successfully promoted
  `develop` wheel;
- a `develop` push compares against the latest successfully promoted `main`
  wheel;
- a `main` push compares against the previous `main` SHA from the push event;
  and
- a tag push compares against the latest successfully promoted `main` wheel.

The base and candidate must use the same cached source models, deterministic
inputs, seeds, toolchain version, and comparison policy. Each revision must be
installed and executed in an isolated environment.

## Test matrix

The supported cases must be defined centrally rather than repeated across
individual test files. A case can declare:

| Dimension | Examples |
|---|---|
| Architecture | Llama, Gemma, Qwen, Mistral, Phi, LFM |
| Source format | Hugging Face safetensors, GGUF |
| Component | pre, cache, post, vision, per-layer, convolution, draft FC |
| Feature | standard, multimodal, embedding quantization, speculative decoding |
| Precision | BF16, INT8, supported GGUF quantization types |
| Resource class | normal, serial/high-memory, full E2E |
| Regression mode | required, informative, disabled |

Every test group selects the relevant subset of this matrix. This avoids
duplicated model setup and generation code and makes coverage gaps visible.

Disabled regression cases require a reason, an owner, and a condition for
re-enabling them.

## Test directory structure

The test tree must separate host-side compilation tests from future
DevKit-side runtime tests. No test module should remain directly under
`tests/`.

The target structure is:

```text
tests/
├── __init__.py
├── compilation/
│   ├── __init__.py
│   ├── conftest.py
│   ├── cases.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_quantization.py
│   │   └── test_weight_mapping.py
│   ├── configuration/
│   │   ├── __init__.py
│   │   ├── test_config_regression.py
│   │   └── references/
│   │       └── *.json
│   ├── source_ingestion/
│   │   ├── __init__.py
│   │   └── gguf/
│   │       ├── __init__.py
│   │       ├── test_parser.py
│   │       ├── test_weight_loading.py
│   │       └── test_dequantization.py
│   ├── onnx_regression/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── generate.py
│   │   └── test_branch_regression.py
│   ├── graph_integration/
│   │   ├── __init__.py
│   │   ├── test_path_equivalence.py
│   │   ├── test_embedding_quantization.py
│   │   ├── test_gguf_integration.py
│   │   └── test_speculative_decoding.py
│   ├── e2e/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   └── test_selected_model_compile.py
│   └── helpers/
│       ├── __init__.py
│       ├── model_factory.py
│       ├── output_comparison.py
│       └── paths.py
└── runtime/
    └── __init__.py
```

The root `tests/__init__.py` is only a package marker that prevents an
installed third-party `tests` package from shadowing this test tree. It is not
a test module; all test implementations and helpers live under
`tests/compilation/` or `tests/runtime/`.

The six test-group directories map one-to-one to the six logical groups:

| Test group | Directory |
|---|---|
| Fast compiler unit tests | `tests/compilation/unit/` |
| Configuration contract regression | `tests/compilation/configuration/` |
| Model-source ingestion | `tests/compilation/source_ingestion/` |
| ONNX generation and branch regression | `tests/compilation/onnx_regression/` |
| Generated-graph and quantization integration | `tests/compilation/graph_integration/` |
| Selected-model full compilation E2E | `tests/compilation/e2e/` |

`helpers/`, `cases.py`, and `conftest.py` are shared compilation-test
infrastructure and are not additional test groups.

`tests/runtime/` is created now as an empty package boundary. Future tests
that execute LLiMa on a DevKit, validate generated text or media, exercise the
C++ runtime, or test CLI/web/ZMQ runtime behavior belong there. Runtime tests
must not be added to `tests/compilation/` merely because the runtime suite is
not implemented yet.

For the initial reorganization, every existing test and test helper belongs
under `tests/compilation/`. Tests that are later identified as runtime
behavior, such as the current `llama.cpp` text-inference check, should be
removed from compiler CI and reintroduced under `tests/runtime/` when the
runtime workflow exists.

### Placement rules

- `compilation/conftest.py` owns compiler-specific pytest options and fixtures.
- `compilation/cases.py` is the single source of truth for model, architecture,
  component, feature, precision, resource, and regression-mode cases.
- Test modules must not import functions from `conftest.py`; reusable functions
  belong under `compilation/helpers/`.
- JSON configuration references live beside the configuration-regression
  tests under `compilation/configuration/references/`.
- A test file is named for the behavior it validates, not for the historical
  implementation file or a single model.
- Standard and speculative cases should share helpers and parameterized test
  infrastructure.
- Compilation tests must not import runtime-test helpers, and future runtime
  tests must not depend on compiler internals.
- Generated ONNX, graph, quantization, and compiler outputs must never be
  written into the source-controlled test tree.

### Migration from the current tree

| Current location | Target |
|---|---|
| `tests/conftest.py` | Compiler fixtures to `tests/compilation/conftest.py`; reusable path helpers to `tests/compilation/helpers/` |
| `tests/test_whisper_config.py` | Hermetic cases to `tests/compilation/unit/test_config.py`; model-backed cases to configuration regression |
| `tests/test_vlm_config.py` | Hermetic cases to `tests/compilation/unit/test_config.py`; model-backed golden cases to `tests/compilation/configuration/test_config_regression.py` |
| `tests/reference_configs/` | `tests/compilation/configuration/references/` |
| `tests/test_gguf.py` | Split between `unit/`, `source_ingestion/gguf/`, `configuration/`, and `graph_integration/test_gguf_integration.py` according to the invariant tested |
| `tests/model/test_pre_model.py`, `test_cache_model.py`, `test_post_model.py` | Replace with parameterized cases in `tests/compilation/graph_integration/test_path_equivalence.py` |
| `tests/model/test_embedding_quantization.py` | Pure logic to `unit/`; model-backed graph checks to `graph_integration/test_embedding_quantization.py` |
| `tests/model/test_onnx_regression.py` | `tests/compilation/onnx_regression/test_branch_regression.py` |
| `tests/model/test_speculative_decoding_*` | Consolidate graph cases into `graph_integration/test_speculative_decoding.py` and ONNX cases into the shared branch-regression matrix |
| `tests/model/model_setup.py` and speculative setup | Consolidate into `tests/compilation/helpers/model_factory.py` |
| `tests/test_e2e_compile.py` | `tests/compilation/e2e/test_selected_model_compile.py` |
| `tests/internal/*.py` | Move initially to `tests/compilation/helpers/`; retain only helpers used by compiler tests |

The migration should not be a directory-only rename. Existing mixed-purpose
files must be split according to the six logical test groups while preserving
their valuable assertions.

## Test groups

### 1. Fast compiler unit tests

#### Purpose

Validate pure compiler and configuration logic before downloading or loading
large models.

#### Requirements

- no Hugging Face model loading;
- no GGUF, ONNX, or NumPy fixture loading;
- no Model Compiler invocation;
- no network access; and
- run against the exact candidate wheel installed by the workflow.

#### Coverage

- configuration validation and serialization logic;
- layer, component, and precision selection;
- weight-name mapping;
- quantization and dequantization algorithms using small generated arrays;
- embedding-scale calculations that do not require a real model;
- speculative-decoding configuration validation;
- invalid configuration combinations;
- unsupported source formats or precisions; and
- small deterministic error-path tests.

Existing hermetic Whisper and VLM configuration tests and the GGUF
attention-bias weight-map test belong here.

These tests should complete in seconds and run before model-cache preparation.

### 2. Configuration contract regression

#### Purpose

Ensure every supported source model produces the expected normalized LLiMa
configuration.

#### Method

For every enabled model case:

1. load the cached source model;
2. generate its normalized LLiMa configuration;
3. compare it with a committed JSON reference configuration; and
4. report a readable field-level difference on failure.

#### Coverage

- Hugging Face safetensors models;
- GGUF models where source-format-specific configuration matters;
- LLM and VLM architectures;
- speculative target and draft configuration;
- tokenizer and attention settings;
- vision and multimodal settings;
- grouping and cache settings; and
- quantization-related configuration.

Committed JSON is the only golden model output permitted by this plan. Any
intentional reference change must be visible and reviewed in the pull request.

### 3. Model-source ingestion

#### Purpose

Verify that supported source-model formats are interpreted correctly before
graph generation begins.

This group initially focuses on GGUF because Hugging Face loading is already
exercised throughout the remaining groups.

#### Coverage

- file metadata and quantization-type detection;
- Hugging Face-to-GGUF weight-name mapping;
- weight existence, shapes, and dtypes;
- dequantization against the GGUF library implementation;
- numerical similarity against corresponding BF16 or Hugging Face weights;
- supported Q8, Q4, Q3_K, Q4_K, Q5_K, Q6_K, and eventually Q2_K formats; and
- clear errors for malformed files or missing and incompatible tensors.

#### Migration of existing tests

- consolidate `test_parser`, `test_unpack`, and `test_load_weights` into a
  coherent parameterized ingestion suite;
- replace `test_unpack_q2_k` and its stored NumPy fixtures with a generated
  input or real cached GGUF comparison;
- move GGUF configuration golden coverage to configuration regression; and
- move `llama.cpp` text inference to the future runtime or tool-integration
  workflow.

This group validates source data. It must not also be responsible for testing
all graph-generation paths.

### 4. ONNX generation and branch regression

#### Purpose

Prove that every supported architecture and component can generate valid ONNX
and detect behavioral changes relative to the exact push baseline.

Architecture export validation is part of this group so each ONNX graph is
generated only once per revision.

#### Method

For every named architecture/component case:

1. generate candidate ONNX once;
2. validate it with `onnx.checker`;
3. validate expected input and output names, shapes, and dtypes;
4. generate the corresponding ONNX once from the resolved immutable baseline
   wheel;
5. construct identical deterministic inputs;
6. execute both graphs with ONNX Runtime;
7. compare output count, shapes, dtypes, and values using the declared
   tolerance; and
8. discard both generated graphs after reporting the result.

#### Coverage

- language pre, cache, and post components;
- vision components;
- per-layer components;
- convolution components;
- speculative draft pre, cache, post, and FC components; and
- one or more named models for every supported architecture.

Each case has one regression mode:

- `required`: any structural or numerical difference fails CI;
- `informative`: execute and report differences without blocking the pull
  request; or
- `disabled`: do not execute, with a documented reason and owner.

There are no stored ONNX references. Existing ONNX regression tests that read
shared reference graphs must be replaced by same-run base-versus-candidate
generation.

### 5. Generated-graph and quantization integration

#### Purpose

Ensure that LLiMa's alternative graph-generation and quantization paths
produce equivalent behavior and that feature-specific transformations are
wired correctly.

These tests evaluate the generated SDK or AFE graph. They are not a substitute
for the final Model Compiler E2E test.

#### Standard path equivalence

For representative cases, compare:

```text
source -> ONNX -> quantized SDK graph
                     versus
source -> directly quantized SDK graph
```

Both paths must run with identical deterministic inputs. Their output count,
shapes, dtypes, and numerical values must be compared using explicit
tolerances.

The representative matrix should cover:

- pre, cache, and post components;
- early and late layers when their behavior differs;
- BF16 and representative quantized precisions; and
- models needed for architecture-specific generation behavior.

This replaces the duplicated setup currently spread across
`test_pre_model.py`, `test_cache_model.py`, and `test_post_model.py`.

#### Feature-specific integration

This group also covers transformations that require a loaded model or
generated graph:

- embedding tensor quantization;
- embedding-scale and dequantization wiring;
- per-layer embedding inputs;
- multimodal versus language-only graph behavior; and
- other feature-specific SDK graph transformations.

Tests that only exercise quantization math or configuration logic should be
moved to the fast unit group.

#### Speculative decoding

Speculative decoding uses the same path-equivalence infrastructure, with cases
for:

- draft pre;
- draft cache;
- draft post; and
- draft FC.

The cases must validate their target/draft configuration and graph contracts,
not merely duplicate four test implementations. They run as a separate serial,
high-memory subsection because loading both target and draft models can exhaust
runner memory.

Speculative ONNX behavior is additionally covered in the ONNX generation and
branch regression group.

#### GGUF graph integration

Representative GGUF cases compare:

```text
HF or BF16 generated graph output
                  versus
quantized GGUF generated graph output
```

This validates the integration between GGUF ingestion and graph generation.
Low-level GGUF parsing and dequantization remain in the model-source ingestion
group.

### 6. Selected-model full compilation E2E

#### Purpose

Validate the complete production compilation path through the installed Model
Compiler.

#### Selection

Select one eligible model reproducibly from the candidate commit SHA. The
selection should appear random across changes while remaining reproducible for
a given commit.

The eligible set must exclude models that are known not to support full
compilation and must be defined in source control.

#### Method

```text
source model
  -> configuration
  -> ONNX
  -> quantization
  -> Model Compiler
  -> final compiled artifact
  -> basic artifact validation
```

The test must report:

- selected model and selection seed;
- source revision;
- stages completed;
- final artifact metadata; and
- stage timings.

This group runs last because it is the most expensive. It must not use a stored
compiled reference artifact.

## Workflow order

The compiler workflow should execute in this order:

1. check out and stage the LLiMa workspace;
2. install the exact candidate LLiMa wheel and Model Compiler toolchain;
3. verify package and toolchain provenance;
4. run fast compiler unit tests;
5. download and verify all model inputs required by the enabled matrix;
6. enable offline model access for the remaining tests;
7. run configuration contract regression;
8. run model-source ingestion tests;
9. run ONNX generation and branch regression;
10. run standard generated-graph and quantization integration;
11. run speculative-decoding integration serially;
12. run the selected-model full compilation E2E test; and
13. publish test reports and audit collection and skip results.

The workflow should reuse the prepared model cache and runner-local workspace
without persisting generated model outputs.

## Test execution policy

### Determinism

- all random inputs use explicit seeds;
- candidate and baseline executions use identical inputs;
- numerical tolerances are declared per case or feature;
- model selection is derived from the candidate SHA; and
- tests must not depend on mutable model revisions.

### Missing inputs

An enabled required test must fail when its declared cached input is missing.
It must not silently skip. Optional or disabled cases must be represented
explicitly in the central policy.

### Markers and resources

Use one primary marker per logical group, for example:

- `compiler_unit`;
- `compiler_config`;
- `compiler_source`;
- `compiler_onnx_regression`;
- `compiler_graph_integration`; and
- `compiler_e2e`.

Orthogonal resource markers such as `serial` and `high_memory` may additionally
control scheduling.

### Provenance

Tests must import the exact candidate wheel, not the source checkout
accidentally. CI should record:

- installed LLiMa version and package path;
- candidate SHA and baseline artifact specification/package version;
- Model Compiler/toolchain version;
- source model revisions; and
- selected E2E model and seed.

### Audit

At the end of the workflow, CI must verify:

- expected test groups were collected;
- required cases were executed;
- no unexpected tests were skipped;
- no generated model references were uploaded; and
- reports clearly identify failures by architecture, component, format, and
  generation path.

## Completion criteria

The compiler CI redesign is complete when:

- the six logical groups are implemented and independently selectable;
- the central test matrix describes supported coverage;
- fast unit tests run before model download;
- configuration regression uses only reviewed JSON references;
- GGUF tests no longer depend on stored NumPy outputs;
- ONNX regression generates base and candidate graphs in the same run;
- standard and speculative generation tests share parameterized
  infrastructure;
- one reproducibly selected model completes full Model Compiler E2E;
- missing required inputs cannot become silent skips; and
- runtime/DevKit testing remains a clearly separate workflow.
