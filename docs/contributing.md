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
source weights, precision and output path once. Graph-node construction lives
in the class; independent array-layout and naming utilities remain module
functions. Model components use the graph methods. A component's implementation can then focus on its topology:

```python
from sima_lmm.model.model_graph import ModelGraph

def generate_graph(self, layer_cfg, quantizable):
    graph = ModelGraph(self, {"hidden": (1, 1, self.num_tokens, self.cfg.d_model)}, quantizable)
    hidden = graph.layer_norm("model.norm", graph.inputs["hidden"])
    output = graph.mlp("model.mlp", hidden, "gelu", residual=graph.inputs["hidden"])
    graph.save([output])
```

Define a standalone component's inputs, top-level topology and `graph.save()`
inside `generate_graph()`. Keep `_build_nodes(graph, inputs)` for shared graph
construction, such as Whisper's pre/cache/post parts used by its combined
decoder graphs.

Logging is scoped by `BaseModel.gen_files()`; graph construction does not need a
separate logging argument.

Shapes infer FP32 inputs for `quantizable=True` (a floating graph to quantize
later), or BF16 for `False` (a direct graph using the source weight precision).
Use `input_dtypes={"cache": np.int8}` for integer inputs such as caches; names must
match the input specifications. Existing explicit AFE tensor specifications remain
supported for low-level callers. Input and output
order follows the supplied specifications; shapes and output types come from
AFE's inference. `constant()` casts floating data to the activation precision;
use `dtype=np.int32`, for example, when integer constants require a specific
width. Node names follow AFE's deterministic creation counter.

Import `Node` from `model_graph.py` for graph-node annotations. Helpers receive
the graph rather than a separate `quantizable` flag. `graph.constant()` selects
floating precision automatically; use the NumPy `graph.dtype` when host-side
array calculations need that precision. The stage flag stays at `generate_graph()`
and graph construction.

`save()` finishes the MLA subnet and creates outer-graph output tuples, preserves integer
outputs, casts BF16 outputs to FP32 on EV, and writes the standard artifact name
(`.fp32` for a floating graph). Use `finish()` instead to obtain the completed
network. Both accept `transform_subnet` for model-specific rewrites before the
outer outputs are extracted. Pass the graph itself to component helpers; inputs,
precision and source weights remain bound to the same object.

Common operations include `add`, `sub`, `mul`, `matmul`, `concat`, `slice`,
`transpose`, `reshape`, `softmax`, `topk`, `sum_channels`, `argmax`, `linear`, `conv`, `layer_norm`, `rms_norm`,
`activation`, `softcap`, `mlp`, `rope`, `rope2d`, `split_heads`, `merge_heads`, `split_concat`,
`clip`, `avgpool2d`, `space_to_depth`, `quant`, and `dequant`. A gated MLP uses
`projections=("gate_proj", "up_proj", "down_proj")`; the default is
`("fc1", "fc2")`. `rms_norm("model.norm", input)` uses the language configuration's
epsilon and weight offset. An explicit `epsilon` keeps zero weight offset unless
`weight_offset` is also supplied, for vision and GDN norms.
`rms_norm(None, input, epsilon=...)` infers weightless channels.
RoPE supports full, partial and proportional split-half rotation.
`rope2d(input, cos_x, sin_x, cos_y, sin_y)` rotates channel quarters in
`[x-real, x-imag, y-real, y-imag]` order. `split_heads(input, heads, repeat=...)`
repeats each head for grouped-query patterns. `split_concat()` exposes AFE's
regrouping operation for spatial/token layouts that need more than a reshape.
`space_to_depth(input, blocksize)` merges complete spatial blocks into channels.
`quant(input)` returns `(int8_values, scale)`; `dequant(int8_values, scale)`
restores the graph's activation precision. `argmax(input)` returns INT32 channel indices.
`slice(input, begin, end, stride, axis)` automatically uses selector convolutions
for unaligned, contiguous, single-axis channel slices of rank-four FP32/BF16 tensors; other
slices retain AFE's native behavior.

`linear("model.proj", input)` resolves `model.proj.weight` and its optional bias,
converts OI source weights to SiMa's layout, and retains packed weight values,
scales, nonaligned group sizes and relocation metadata. `conv()` infers OIW or
OIHW source layouts; an explicit transform can convert other source layouts,
such as Qwen's five-dimensional patch weights. `WeightOptions` documents source-name, layout,
weight/scale/bias transform and relocation overrides. When slicing grouped
weights, supply the corresponding `scale_process_func`; groups must retain the
checkpoint's actual size. LoRA rank and merged-adapter behavior are explicit
arguments to `linear()` and `mlp()`.

Build attention from projections and head layout operations. Fold query scaling
into `linear()` to retain the existing projection rounding:

```python
queries = graph.split_heads(graph.linear("attn.q_proj", hidden, scale=head_dim ** -0.5), heads)
keys = graph.split_heads(graph.linear("attn.k_proj", hidden), heads)
values = graph.split_heads(graph.linear("attn.v_proj", hidden), heads)
context = graph.attention(queries, keys, values)
output = graph.linear("attn.out_proj", graph.merge_heads(context))
```

`split_heads()` handles unaligned head channels through selector convolutions.
Standard vision attention pads projection weights for unaligned heads to avoid these
convolutions; grouped output weights retain their original layout.
`attention()` uses query/key lengths to select separate head branches for large
attention tensors, including cross-attention. It accepts an additive `mask` and
assumes queries are already scaled unless `score_scale` is supplied. That scale is applied after
Q×K and before the mask, preserving models such as Qwen vision's BF16 order.
Masks must be rank-four, vectors or scalars and broadcast to `[N,H,T_query,T_key]`;
unsupported shapes raise `ValueError`.

`graph.matmul(lhs, rhs)` creates an MLA batch matmul. Both transpose flags default
to `False`; use `transpose_a=True` for Kᵀ×V or `transpose_b=True` for Q×Kᵀ.
Inputs must be rank-four FP32/BF16 tensors with matching batches and contraction
dimensions. Divisible head counts use implicit repetition for grouped-query
attention; invalid shapes or types raise `ValueError`.

`graph.softmax(x)` defaults to the last axis; an explicit `axis` is supported.
Use `graph.slice(x, start=0, stop=128, axis=-1)` for a contiguous single-axis slice,
or the existing `begin`/`end`/`stride`/`axis` lists for multi-axis slicing. The
single-axis form defaults to start zero, requires an explicit axis and in-range
nonempty bounds, and retains unaligned-channel handling. Do not mix the two forms.

`ModelGraph` extends AFE's `SimaBuilder`, so native operations are available
on the same object as the common helpers:

```python
projected = graph.linear("model.proj", graph.inputs["hidden"])
output = graph.add(projected, graph.inputs["hidden"])
graph.save([output])
```

Concise primitive operations are direct aliases of AFE methods, retaining their
signatures, type inference and deterministic naming. For uncommon operations,
inherited `create_*` methods remain available on the same graph. Graphs requiring
a custom multi-subnet lifecycle can still use AFE's `SimaBuilder` directly with
the low-level operation helpers.

Tessellation is inferred centrally in `sima_analysis.get_tessellate_parameters()`:
HWC16 layout, automatic tile sizes and deterministic persistent buffer names.
Ordinary components inherit empty overrides from `BaseModel`. Exceptional
layouts, such as strided KV caches, override the existing
`get_mla_input_tessellate_params()` / `get_mla_output_tessellate_params()` methods;
keys are tensor indices (negative indices count from the end).

Whisper components, all native language graph parts and the standard vision tower
show this API in use; their attention, MLP and rotary helpers share the same
implementations. DFlash context fusion, K/V projection and state resolver components
also use `generate_graph()` and `ModelGraph`. Its final draft graph keeps the paired
target's packed output-head weights and scales; verification retains the S1 and
solved-intermediate outputs expected by the runtime resolver.

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
`.github/workflows/model-compiler-tests.yml`. Generate native SDK graphs and
numerical comparison artifacts during the run rather than committing binary
baselines.

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
