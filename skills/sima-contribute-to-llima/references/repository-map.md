# LLiMa Repository Map

## Compilation

- `sima_lmm/config/`: LLM, VLM, and ASR configuration contracts.
- `sima_lmm/hf/`: Hugging Face model loading and transformation.
- `sima_lmm/gguf/`: GGUF parsing, mapping, and dequantization.
- `sima_lmm/model/`: generated model components, ONNX construction,
  quantization, and compiler graph assembly.
- `sima_lmm/preproc/`: host-side image and audio preprocessing.
- `sima_lmm/host/`: `llima-compile`, deploy, benchmark, and LoRA entry points.

Compiler work runs in the Python 3.12 Model Compiler environment and may need
large cached model inputs.

## Runtime

- `sima_lmm/devkit/`: Python runtime, model manager, and CLI dispatch.
- `sima_lmm/devkit/cpp/`: C++ runtime, tokenizer, LLM/VLM/ASR orchestration,
  MLA integration, CLI, web, ZMQ, and Python binding.
- `sima_lmm/assets/`: small runtime warm-up/sample assets.

Runtime changes must preserve the lean device dependency set. Real MLA
execution requires compatible Modalix hardware.

## Evaluation

- `sima_lmm/mole/`: MoLE accuracy and performance evaluation.

Keep evaluation dependencies out of the runtime Debian package.

## Packaging and Dependencies

- `deps/manifest.json`: package and platform version source of truth.
- `CMakeLists.txt`, `cmake/`, `build.sh`: runtime CMake and Debian packaging.
- `pyproject.toml`, `pyproject_metadata.py`: wheel metadata and Python extras.
- `build_compiler_wheel.sh`, `build_mole_package.sh`: Python publication
  profiles.
- `tools/install_*.sh`: published artifact installers.
- `dist/`: generated publication layouts; never hand-edit as source.

## Tests and CI

- `tests/compilation/`: compiler unit, configuration, source-ingestion, ONNX,
  graph-integration, and bounded end-to-end coverage.
- `tests/runtime/`: Modalix C++ and Python runtime validation.
- `tests/README.md`, `pytest.ini`: test matrix, local commands, markers, and
  compiler premerge policy.
- `.github/workflows/vulcan-ci.yml`: build, publication, compiler smoke, and
  DevKit validation orchestration.
- `.github/workflows/model-compiler-tests.yml`: Model Compiler test workflow.
- `tools/ci/prepare_model_inputs.py`: approved cached model preparation.
- `tools/hf-safetensors/`: source-model manifests and cache publisher logic.

Read `tests/README.md` for local commands and
`tests/compilation/conftest.py` for the current model-input override. Do not
accept unintended fixture skips as proof.

## Documentation and Skills

- `README.md`: repository-level install, package, build, and runtime overview.
- `docs/`: official LLiMa user and contributor documentation.
- `CONTRIBUTING.md`: concise GitHub contribution entry point.
- `AGENTS.md`: enforceable repository guardrails.
- `skills/`: Playbooks-compatible LLiMa workflows.
- `.github/PULL_REQUEST_TEMPLATE.md`: required PR evidence and impact
  assessment.

Update the closest source of truth rather than repeating the same policy in
several files.
