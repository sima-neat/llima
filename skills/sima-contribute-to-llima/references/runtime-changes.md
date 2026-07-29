# LLiMa Runtime Changes

Use for C++ runtime, installed API, binding, CLI/HTTP/ZMQ, lifecycle, or
runtime-package changes.

## Package Ownership

| Package | Owns |
| --- | --- |
| `sima-lmm-core` | `libsima_lmm_runtime.so` and shared runtime assets |
| `sima-lmm-dev` | Installed `sima_lmm` and required bundled headers; `SimaLMM` CMake package |
| `sima-lmm-cli` | Python CLI modules, internal nanobind bridge, and `llima` launcher |

`dev` and `cli` require the exact matching `core`. Keep compiler-only
dependencies out of all three.

## Public Headers

Install a header only when an external package includes it or another installed
header needs it. Headers used only by LLiMa sources, bindings, or executables
stay private.

Every `SIMA_LMM_PUBLIC_HEADERS` entry is a compatibility surface:

- package its transitive includes;
- hide storage, ownership, and synchronization details where practical;
- audit downstream consumers before changing or removing it; and
- update the CMake package and `sima-lmm-dev` dependencies together.

Do not publish a header solely for an in-tree test.

## Cross-Surface Changes

Implement shared semantics in the common C++ layer, then inspect:

- installed declarations and implementation;
- internal `binding.cpp` bridge and Python CLI orchestration;
- CLI, OpenAI-compatible HTTP, and ZMQ behavior;
- errors, cancellation, and subsequent requests;
- buffer, model, thread, process, and descriptor lifetime; and
- CMake components and Debian dependencies.

Expose a feature only where its contract is defined; document intentional
surface differences.

## API, ABI, and Package Validation

For private implementation changes, run focused unit/CTest coverage and
packaged Modalix tests when MLA, dispatcher, or lifecycle behavior is involved.

For installed C++ API changes:

1. Install candidate `core` and `dev` packages into the test sysroot overlay.
2. Compile/link a consumer with `find_package(SimaLMM CONFIG REQUIRED)` and
   `SimaLMM::sima_lmm_runtime`, without source-tree includes.
3. Exercise the API, rebuild known consumers, and update docs/migration notes.

For an ABI-compatible claim, compare against the last compatible published
library: exported symbols, affected layouts/virtual interfaces, calling
conventions, ownership, and exception boundaries. Run a consumer binary built
with the prior `dev` package against candidate `core`. If compatibility is not
demonstrated, mark the change breaking and coordinate versioning, dependency
constraints, downstream rebuilds, and migration.

For bindings or entry points, test installed `sima-lmm-cli`, not an editable
checkout. Build `./build.sh --all --clean`, inspect package files/dependencies,
install matching packages on Modalix, and run affected packaged tests from
`tests/README.md`.

## Example: Cancellation

Verify:

- common C++ cancellation stops inference within a bound;
- CLI uses graceful `SIGINT`, never `SIGTERM` or `SIGKILL`;
- HTTP `/stop` ends the stream and a later request succeeds;
- CLI/ZMQ remain valid or gain coverage when cancellation is exposed;
- the client exits and releases buffers, threads, and descriptors; and
- appcomplex PID, thread/descriptor counts, and CMA memory remain near
  baseline.

Until mla-rt 2.1.3, restart appcomplex before each case to isolate retained
buffers. Still assert dispatcher stability within the case.
