# Contributing to LLiMa

LLiMa contains a host-side GenAI compiler and a C++ runtime for Modalix. The
runtime is operated through packaged CLI/HTTP/ZMQ entry points; Python is CLI
orchestration, not a separate public runtime API. Keep compiler and runtime
environments and dependencies separate. In a repository checkout,
`CONTRIBUTING.md` provides the quick start and `AGENTS.md` defines
agent-specific rules; this guide is the detailed contributor policy.

## Coding-Agent Skills

Install both LLiMa contributor skills as part of standard contributor setup.
They are intentionally not installed by the default Neat SDK playbook index:

```bash
sima-cli playbooks install \
  gh:sima-neat/llima/skills/sima-contribute-to-llima
sima-cli playbooks install \
  gh:sima-neat/llima/skills/sima-add-llima-model-support
```

The general contributor skill covers repository-wide compiler, runtime,
packaging, test, documentation, and skill changes. The model-support skill adds
the compatibility and implementation workflow for LLM and VLM architectures,
checkpoints, tensor layouts, tokenizers, and prompt contracts. Keep both
installed so the appropriate guidance is available when a contribution crosses
those boundaries.


## Repository Map

| Area | Paths | Responsibility |
| --- | --- | --- |
| Configuration | `sima_lmm/config/` | LLM, VLM, and ASR configuration contracts |
| Ingestion | `sima_lmm/hf/`, `sima_lmm/gguf/` | Hugging Face and GGUF loading and conversion |
| Compilation | `sima_lmm/model/`, `sima_lmm/preproc/` | Model parts, quantization, graphs, and preprocessing |
| Host tools | `sima_lmm/host/` | Compile, deploy, LoRA, and benchmark entry points |
| Evaluation | `sima_lmm/mole/` | MoLE workflows |
| Runtime CLI | `sima_lmm/devkit/` | Python CLI orchestration and model management |
| C++ runtime | `sima_lmm/devkit/cpp/` | Models, tokenizer, MLA, CLI/HTTP/ZMQ implementations, and the internal CLI binding |
| Tests | `tests/` | Compiler and Modalix runtime tests |
| Packaging | `CMakeLists.txt`, `cmake/`, `build*.sh`, `tools/install_*.sh` | Debian, wheel, and artifact assembly |
| CI/caches | `.github/workflows/`, `tools/ci/`, `tools/hf-safetensors/` | Builds, tests, and model caches |
| Docs/skills | `README.md`, `docs/`, `skills/` | User, contributor, and Playbooks guidance |

Compiler-only dependencies must not enter `sima_lmm/devkit/` or the Modalix
runtime packages.

## Development Environments

### Runtime and packaging

Use the Neat SDK as the supported build environment. Build all runtime packages
and packaged tests with:

```bash
./build.sh --all --clean
```

The normal build handles its required setup, including submodules; do not run a
separate dependency-bootstrap step for the standard workflow.

Useful narrower builds:

```bash
./build.sh --clean --core
./build.sh --clean --core --dev
./build.sh --clean --cli
./build.sh --no-dist
```

Outputs are generated under `build-deb/` and staged under `dist/`. Modalix is
required for MLA execution and real `llima run` validation.

### Compiler development

Use the Python 3.12 environment installed by Model Compiler. Search in order:

1. `/sdk-extensions/model-compiler`
2. `/sdk-add-on/model-compiler`
3. `$HOME/sdk-extensions/model-compiler`

```bash
source <model-compiler-venv>/bin/activate
python -m pip install -e '.[sdk_ext,tests]'
llima-compile --help
```

Do not create another environment that shadows the installed compiler
packages. Build publication profiles with:

```bash
./build_compiler_wheel.sh
./build_mole_package.sh
```

They use wheel tooling under `build/` and stage outputs under `dist/compiler/`
and `dist/mole/`.

### Whisper and ASR development

The public `llima-compile` workflow covers LLMs and VLMs. Existing Whisper
compilation instead uses the contributor utility
`scripts/gen_models--openai--whisper.py`:

```bash
python scripts/gen_models--openai--whisper.py \
  --model_path /path/to/openai/whisper-small \
  --output /path/to/whisper-output \
  --part all
```

Run it in the Model Compiler environment with an explicit model path.
`--part` accepts `all`, `encoder`, `language_detect`, `init`, `single_pre`,
`single_post`, and `single_cache`. Add `--enable_log_probe` to compile
log-probe-enabled decoder outputs; use `--part all --enable_log_probe` for a
complete log-probe build.

Whisper model repositories contain one ELF per encoder layer. The runtime does
not support legacy repositories with a monolithic encoder ELF; download a
layered model or recompile the checkpoint with the current LLiMa version.

Compiler changes normally touch `sima_lmm/config/whisper_config.py`,
`sima_lmm/model/whisper_*.py`, and the script; runtime changes touch
`sima_lmm/devkit/cpp/whisper_*`. Validate with the packaged C++ ASR runtime
test documented in `tests/README.md` and representative audio on Modalix. This
is a Whisper-specific path, not a general ASR architecture framework.

### Native graph components

Use `ModelGraph` from `sima_lmm/model/model_graph.py`. It binds the component's
source weights, precision and output path once. A component's implementation can
then focus on its topology:

```python
from sima_lmm.model.model_graph import ModelGraph

def gen_model_sdk_files_directly(self, layer_cfg, log_level, quantizable):
    graph = ModelGraph(self, {"hidden": (1, 1, self.num_tokens, self.cfg.d_model)}, quantizable)
    hidden = graph.layer_norm("model.norm", graph.inputs["hidden"])
    output = graph.mlp("model.mlp", hidden, "gelu", residual=graph.inputs["hidden"])
    graph.save([output])
```

Shapes infer FP32 inputs for `quantizable=True` (a floating graph to quantize
later), or BF16 for `False` (a direct graph using the source weight precision).
Use `TensorType(ScalarType.int8, shape)` for integer caches. Input and output
order follows the supplied specifications; shapes and output types come from
AFE's inference. `constant()` casts floating data to the activation precision;
use `dtype=np.int32`, for example, when integer constants require a specific
width. Node names follow AFE's deterministic creation counter.

`save()` finishes the MLA subnet and creates outer-graph output tuples, preserves integer
outputs, casts BF16 outputs to FP32 on EV, and writes the standard artifact name
(`.fp32` for a floating graph). Use `finish()` instead to obtain the completed
network. Both accept `transform_subnet` for model-specific rewrites before the
outer outputs are extracted. `ModelGraph.from_builder(self, builder)` binds the
same operations inside an existing subnet without creating new inputs.

Common operations include `linear`, `conv`, `layer_norm`, `rms_norm`,
`activation`, `mlp`, `rope`, `split_heads`, `merge_heads`, `quantize`, and
`dequantize`. A gated MLP uses
`projections=("gate_proj", "up_proj", "down_proj")`; the default is
`("fc1", "fc2")`. `rms_norm(None, input, epsilon=...)` infers weightless channels.
RoPE supports full, partial and proportional split-half rotation.

`linear("model.proj", input)` resolves `model.proj.weight` and its optional bias,
converts OI source weights to SiMa's layout, and retains packed weight values,
scales, nonaligned group sizes and relocation metadata. `conv()` infers OIW or
OIHW source layouts. `WeightOptions` documents explicit source-name, layout,
weight/scale/bias transform and relocation overrides. When slicing grouped
weights, supply the corresponding `scale_process_func`; groups must retain the
checkpoint's actual size. LoRA rank and merged-adapter behavior are explicit
arguments to `linear()` and `mlp()`.

For multi-head attention, `project_heads()` preserves the existing MLA channel
padding and packed-versus-separate-head selection; `project_merged_heads()`
merges those outputs before projecting. Scale queries during projection:

```python
queries = graph.project_heads("attn.q_proj", hidden, heads, scale=head_dim ** -0.5)
keys = graph.project_heads("attn.k_proj", hidden, heads)
values = graph.project_heads("attn.v_proj", hidden, heads)
outputs = [graph.attention(q, k, v) for q, k, v in zip(queries, keys, values)]
output = graph.project_merged_heads("attn.out_proj", outputs, heads)
```

For cross-attention, pass the same `query_len` and `kv_len` to all three
projections. `attention()` accepts an additive `mask` and assumes queries are
already scaled.

`graph.einsum(equation, lhs, rhs)` supports the following NHWC contractions,
including renamed labels and whitespace:

| Equation | MLA transpose flags (A, B) |
| --- | --- |
| `nhwc,nhcq->nhwq` | false, false |
| `nhwc,nhqc->nhwq` | false, true |
| `nhcw,nhcq->nhwq` | true, false |
| `nhcw,nhqc->nhwq` | true, true |

Each creates an MLA `BatchMatmulOp`. Unsupported equations, mismatched scalar
or contraction dimensions, and non-singleton head mismatches raise `ValueError`;
there is no alternative backend fallback. Batches must match. Heads may broadcast
from one. Grouped-query head repetition uses the raw batch-matmul operation
because its semantics differ from NumPy `einsum`.

**Raw escape hatch:** use `graph.raw`, the original AFE `SimaBuilder`, for
exceptional topology (for example grouped-query BMM, channel selection, or
model-specific softcapping). Nodes created through either API can be mixed.
Graphs with multiple MLA subnets can retain the raw lifecycle and bind common
operations with `from_builder()`.

Tessellation is inferred centrally in `sima_analysis.get_tessellate_parameters()`:
HWC16 layout, automatic tile sizes and deterministic persistent buffer names.
Ordinary components inherit empty overrides from `BaseModel`. Exceptional
layouts, such as strided KV caches, override the existing
`get_mla_input_tessellate_params()` / `get_mla_output_tessellate_params()` methods;
keys are tensor indices (negative indices count from the end).

Whisper components, all native language graph parts and the standard vision tower
show this API in use; their attention, MLP and rotary helpers share the same
implementations.

## Testing

Choose tests by failure surface. A build does not replace behavioral
validation, and a skipped required case is not a pass.

### Hermetic tests

Keep pure configuration, mapping, serialization, validation, and numerical
logic independent of model downloads:

```bash
pytest -q <targeted-test-path>
```

### Model-backed compiler tests

Compiler tests live under `tests/compilation/`. Select the affected group and
marker described in `tests/README.md`. For example:

```bash
export LLIMA_HF_MODELS_PATH=/path/to/llima-model-inputs
python -P -m pytest \
  -c pytest.ini \
  tests/compilation/configuration \
  -m compiler_config \
  --strict-markers \
  -vv -ra
```

`--model-inputs-path` and `LLIMA_HF_MODELS_PATH` select the prepared Hugging
Face/GGUF input root. CI uses manifests under `tools/hf-safetensors/`.
Configure required inputs instead of accepting fixture skips.

The test matrix, expected counts, and baseline policy live in
`tests/README.md`; CI invocation lives in
`.github/workflows/model-compiler-tests.yml`. Generate ONNX and numerical
comparison artifacts during the run rather than committing binary baselines.

### Runtime validation

Build candidate packages and runtime-test extras:

```bash
./build.sh --all --clean
```

This builds but does not run the tests. Install the matching candidate LLiMa
packages on Modalix, extract the extras archive, and run packaged CTest and
pytest following the DevKit runtime-testing instructions in `tests/README.md`.

Run affected hardware tests when a change reaches model loading, inference,
tokenization, multimodal preprocessing, speculative decoding, CLI/HTTP/ZMQ, or
resource lifecycle. Add a representative smoke test when needed:

```bash
llima run <model_dir> --mode cli
```

For VLM changes, include an image-grounded prompt. Manual smoke testing
complements, but does not replace, affected packaged coverage.

Neat Core consumes LLiMa's installed C++ API and runtime packages. When either
of those surfaces—or behavior exposed through Core's GenAI API—changes, build
Core against the candidate `sima-lmm-core` and `sima-lmm-dev` packages, not a
published or cached LLiMa build. Run the affected Core GenAI C++ tests on
Modalix. This downstream validation is not required for isolated compiler,
documentation, or test-only changes.

### Packaging validation

Build each changed profile:

```bash
./build.sh --all --clean
./build_compiler_wheel.sh
./build_mole_package.sh
```

Verify package names, file ownership, install manifests, dependencies,
checksums, and metadata.

## Coding Standards

### Compatibility and boundaries

Treat installed C++ headers, CLI commands, serialized configuration, package
metadata, and generated artifact layouts as
compatibility surfaces. Prefer additive changes. For a break, document
affected consumers, migration, and release intent; update callers, tests,
examples, and user docs.

Keep compiler, runtime, and MoLE dependencies separate. Runtime state must not
be an input to host compilation. Changes to runtime package boundaries must
preserve the roles of `sima-lmm-core`, `sima-lmm-dev`, and `sima-lmm-cli` and
include appropriate API/ABI validation.

### Implementation quality

- Target C++20 and Python versions declared in `pyproject.toml`.
- Follow surrounding formatting, naming, and include grouping; avoid broad
  mechanical reformatting.
- Keep installed interfaces minimal and implementation details private.
- Add Python type annotations where practical.
- Explain non-obvious contracts, numerical assumptions, and hardware
  constraints; do not narrate the code.
- Reuse nearby helpers before adding abstractions.
- Preserve deterministic model selection, graph structure, serialization, and
  artifact names. Record seeds and avoid filesystem/process ordering.
- Reject unsupported or invalid input with context; preserve original causes
  across layers and never silently select another execution path.
- Bound worker coordination and teardown. Make buffer, handle, thread, and
  temporary-file ownership explicit; clean partial work safely.
- Avoid unnecessary allocation, copies, and synchronization in hot paths.

### Models, artifacts, dependencies, and secrets

Allowed persistent test material:

- reviewed JSON configuration contracts;
- source-controlled cases, seeds, tolerances, and comparison policy; and
- manifests for approved immutable Hugging Face/GGUF revisions.

Do not commit downloaded weights, customer data, or generated ONNX, NumPy,
quantized, MPK, ELF, or runtime model trees. Keep generated outputs in ignored
or temporary directories.

Use `deps/manifest.json` for package/platform versions. Treat `third_party/` as
vendored code; isolate and document intentional submodule updates. Do not add
compiler dependencies to runtime Debian packages.

Never commit or log tokens, SSH credentials, private repositories, signed
URLs, or personal paths. Keep gated-model authorization outside source control
and use public model IDs, immutable revisions, and redacted logs in reports.

## Documentation, Skills, and Pull Requests

Update the closest user guide:

- [System Requirements](setup.md)
- [Model Compilation](compilation_genai.md)
- [Model Deployment](deployment.md)
- [LLiMa CLI](runtime.md)
- [MoLE](mole.md)

Keep root `CONTRIBUTING.md` as the quick start, this file as detailed policy,
and `AGENTS.md` as enforceable agent rules. Skills must contain valid
`SKILL.md`, `playbook.yml`, and agent metadata; keep their main workflow short
and move conditional details into direct references.

Validate all skill payloads from the repository root in the Neat SDK without
changing installed agent state:

```bash
playbooks_validation_dir="$(mktemp -d)"
CODEX_HOME="${playbooks_validation_dir}/codex" \
CLAUDE_HOME="${playbooks_validation_dir}/claude" \
SIMA_CLI_HOME="${playbooks_validation_dir}/sima-cli" \
sima-cli playbooks install ./skills
```

The install summary must report `detected: 3`, `valid: 3`, and `discarded: 0`.
The `sima-cli` version must satisfy `min_cli_version` in each `playbook.yml`.

For pull requests:

- branch from and target current `develop`;
- keep commits focused with imperative subjects;
- use `.github/PULL_REQUEST_TEMPLATE.md`;
- link completed issues with `Fixes #<issue>`;
- report risk, compatibility/migration, docs impact, reproducible commands,
  model/package versions, hardware evidence, skipped checks, and residual risk;
  and
- exclude credentials and private assets.

A contribution is ready when affected tests pass without unintended skips,
required package and Modalix checks complete or are explicitly unavailable,
compatibility and docs are addressed, and the PR contains reproducible
evidence.
