# LLiMa Repository Map

| Surface | Paths |
| --- | --- |
| Configuration | `sima_lmm/config/` |
| Hugging Face/GGUF ingestion | `sima_lmm/hf/`, `sima_lmm/gguf/` |
| Graphs, quantization, model parts | `sima_lmm/model/` |
| Host preprocessing | `sima_lmm/preproc/` |
| Compile/deploy/benchmark/LoRA tools | `sima_lmm/host/` |
| Python CLI orchestration/model management | `sima_lmm/devkit/` |
| C++ runtime, tokenizer, MLA, CLI/HTTP/ZMQ, internal CLI binding | `sima_lmm/devkit/cpp/` |
| MoLE evaluation | `sima_lmm/mole/` |
| Runtime/Debian packaging | `CMakeLists.txt`, `cmake/`, `build.sh` |
| Wheel profiles | `pyproject.toml`, `pyproject_metadata.py`, `build_*package.sh`, `build_compiler_wheel.sh` |
| Dependencies/installers | `deps/manifest.json`, `tools/install_*.sh` |
| Compiler/runtime tests | `tests/compilation/`, `tests/runtime/` |
| CI/model caches | `.github/workflows/`, `tools/ci/`, `tools/hf-safetensors/` |
| User/contributor docs | `README.md`, `docs/`, `CONTRIBUTING.md`, `AGENTS.md` |

Whisper compilation is a repository-local exception:
`scripts/gen_models--openai--whisper.py`,
`sima_lmm/config/whisper_config.py`, and `sima_lmm/model/whisper_*.py`.
Its runtime starts at `sima_lmm/devkit/cpp/whisper_*`; the script is not a
public `llima-compile` entry point.

Use:

- `tests/README.md` for test groups, commands, and model-path overrides;
- `.github/workflows/vulcan-ci.yml` for build/publication/DevKit orchestration;
- `.github/workflows/model-compiler-tests.yml` for compiler tests;
- `tools/ci/prepare_model_inputs.py` and `tools/hf-safetensors/` for cache
  preparation; and
- `.github/PULL_REQUEST_TEMPLATE.md` for required PR evidence.

Treat `dist/` as generated output. Update the closest source of truth.

For checkpoint compatibility or new model support, use
`sima-add-llima-model-support` and read its
`references/compatibility-audit.md`; that audit owns the detailed boundary
entry points and resolver routing.
