---
name: sima-contribute-to-llima
description: Modify or review the LLiMa repository. For new LLM/VLM checkpoint compatibility or architecture support, use sima-add-llima-model-support.
---

# Contribute to LLiMa

Work in a LLiMa checkout and follow its `AGENTS.md`. Skill installation includes
the references below, not the repository's contributor guide. Repository paths
such as `docs/contributing.md` are relative to that checkout.

## Route the Work

Load guidance for the affected surface, rather than all repository documents:

| Task | Guidance |
| --- | --- |
| Locate code or package ownership | [Repository map](references/repository-map.md) |
| Build or compare compiler graphs | ModelGraph, `graph.run()` reference execution, and tensor layouts in `CONTRIBUTING.md` and `docs/contributing.md` |
| Change runtime, installed APIs, bindings, or runtime packages | [Runtime contracts](references/runtime-changes.md) |
| Select tests or build commands | Relevant sections of `tests/README.md` and `docs/contributing.md` |
| Change dependencies or entry points | `deps/manifest.json` or `pyproject.toml`; verify the actual installed versions too |

Use the model-support skill when the question is new LLM/VLM compatibility.
Compiling an already supported model belongs to `sima-llima-compile-run`.
Existing Whisper maintenance belongs here; the repository map identifies its
separate compiler and runtime paths.

## Constraints and Validation

Preserve the compiler/runtime/package boundaries and existing API, CLI,
configuration, and artifact contracts. Reuse existing resolvers and graph
helpers; reject unsupported input instead of silently changing the source,
revision, precision, or execution path. Repository artifact, credential, and
vendor rules remain authoritative.

Prefer existing test coverage; add or extend tests only for uncovered behavior
or meaningful regression risk, not tests that merely mirror the implementation.
Select validation for the changed surface: hermetic tests for pure logic,
affected model-backed cases for graphs, package builds for packaging, and
packaged Modalix checks for hardware-dependent runtime behavior. Skills need
structural checks and the isolated install command in `docs/contributing.md`.
Do not download models or run hardware/package suites for unrelated changes.
Unintended skips and unavailable checks are not passes.

## Completion

A review or diagnosis ends with evidence-backed findings. An implementation
includes the requested changes, affected validation, and user-visible docs.
Report compatibility impact, validation results, unavailable checks, and
remaining limitations; distinguish independent downstream blockers.
