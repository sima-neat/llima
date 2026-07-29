---
name: sima-contribute-to-llima
description: Make safe changes in the sima-neat/llima repository across GenAI compilation, Hugging Face or GGUF ingestion, generated graphs, quantization, Modalix runtime, C++ or Python APIs, packaging, CI, tests, documentation, and skills. Use for implementation, diagnosis, refactoring, testing, review fixes, and contributor documentation. Do not use for compiling a customer's model or building a Neat application.
---

# Contribute to LLiMa

## Orient

1. Read `AGENTS.md` and `docs/contributing.md`.
2. Read `references/repository-map.md`, then classify the change.
3. For runtime implementation, installed headers, bindings, CLI/HTTP/ZMQ,
   lifecycle, or runtime packages, read `references/runtime-changes.md`.
4. For a new architecture, source layout, tokenizer, or prompt contract, also
   use `sima-add-llima-model-support`.
5. Search for nearby implementations, tests, CLI definitions, and docs.
   Preserve unrelated work and vendored code.

## Implement

- Make the smallest change that preserves compiler/runtime/package boundaries.
- Treat installed APIs, CLI contracts, serialized configuration, package
  metadata, and artifact layouts as compatibility surfaces.
- Use `deps/manifest.json` for dependency versions.
- Reject unsupported input explicitly; do not silently change model, revision,
  precision, format, or execution path.
- Add focused tests and update the closest official guide with user-visible
  behavior.

Apply the artifact, model-input, secret, vendor, and coding rules from
`AGENTS.md`; do not duplicate them in task-specific changes.

## Validate

Use exact commands from `docs/contributing.md` and `tests/README.md`:

- hermetic tests for pure logic;
- configured model-backed tests without unintended skips;
- affected Debian/wheel builds for packaging;
- packaged Modalix tests for MLA/runtime behavior; and
- `quick_validate.py` plus isolated Playbooks installation for skills.

Run targeted checks before broader tiers. Report unavailable model, compiler,
or hardware checks as limitations, not passes.

## Finish

Report changed surfaces, compatibility/docs impact, tests run, skipped checks,
and residual risk.

## References

- `references/repository-map.md`
- `references/runtime-changes.md`
