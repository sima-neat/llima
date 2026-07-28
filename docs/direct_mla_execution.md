# Direct MLA execution

This document explains how the LLiMa runtime executes compiled QMLA models
through the MLA kernel driver. It is primarily a reviewer and maintainer guide:
applications should continue to use the normal LLiMa CLI, C++, or Python
interfaces.

The direct path has one intentionally short ownership chain:

```text
LLiMa model
  -> MlaExecutionSession
  -> MlaKernelBackend
  -> /dev/mla
  -> one terminal completion per JOB_EXEC
```

There is no Dispatcher or MLA-RT execution dependency in this chain. LLiMa
still understands the same compiler-produced model package and preserves the
useful queue-ahead behavior of the earlier MLA-RT multi-model integration, but
model submission, completion, and failure now have one owner.

The behavior comparison point is
`mla_rt@596a9d8d2624fb5869a798098742296783656305` on
`SWMLA-9487-v2-userspace`. The direct path preserves a useful execution
property from that revision; it does not copy its public API or add an MLA-RT
link dependency.

## Design goals

The direct path is designed around five rules:

1. **One session owns one execution context.** Models, imported buffers,
   adapter selection, ordered execution, and shutdown all belong to the same
   `MlaExecutionSession`.
2. **Compiler metadata is authoritative.** LLiMa binds the physical ports and
   byte extents described by the QMLA package. It does not infer a different
   port order or pass physical addresses from userspace.
3. **Validate once, submit an immutable handle.** A `BoundExecution` captures
   one checked model-and-buffer binding. `JOB_EXEC` never receives a mutable
   binding vector.
4. **One completion owns the result.** The terminal CQE reports success,
   cancellation, timeout, or hardware failure. There is no second fence/reap
   lifecycle.
5. **Keep scheduling policy private.** LLiMa keeps a small internal window of
   ordinary jobs ready. The kernel remains the only scheduler and may
   arbitrate between contexts at every compiled-job boundary.

## What changed from the earlier runtime

The earlier path passed model work through MLA-RT and Dispatcher-managed
objects. It also relied on process-global runtime state for device submission.
The direct path keeps the compiler package semantics but replaces the transport
and ownership model:

| Earlier path | Direct path |
| --- | --- |
| MLA-RT/Dispatcher owns submission | `MlaKernelBackend` owns submission |
| Process-global execution handle | Explicit `MlaExecutionSession` |
| Mutable model/buffer arguments at execution | Immutable `BoundExecution` |
| Separate submission and completion objects | One retained `JOB_EXEC` and terminal CQE |
| Userspace runtime participates in scheduling | Kernel schedules context FIFOs |
| Runtime-specific physical-address bindings | Checked dma-buf `BufferView` bindings |

`connect_mla()` remains as a source-compatibility wrapper. It creates the
process-default direct session. Historical MLA-RT command-line arguments are
accepted so existing callers still build, but they are ignored and cannot
select another transport. Internal model constructors retain the session
explicitly instead of looking up a global device handle while executing.

## Session ownership

Each `MlaExecutionSession` owns:

- one Backend and one `/dev/mla` context;
- the context's configure-once workload priority;
- every model package published into that context;
- the cache of imported dma-buf registrations;
- immutable model/buffer bindings;
- one execution lease that prevents dependent model segments from
  interleaving;
- the private rolling job window; and
- the poisoned/healthy state used for fail-stop recovery.

The default LLiMa session opens with `Background` intent. That lets a separate
latency-sensitive Neat context run between LLiMa's compiled jobs. Priority does
not interrupt an MLA command that is already running, and it does not change
within a session.

The boundary is finer than token production: a higher-priority context may run
between any two compiled layer jobs in a token. Jobs already accepted into
LLiMa's small window remain ordered within their context, but they do not
reserve the accelerator ahead of a higher-priority runnable context.

An `MlaExecutionSegment` acquires the session execution lease. The lease spans
all of the segment's submitted jobs and may intentionally span a CPU
observation or mutation between subsegments. This is necessary for language
models: two producers can be memory-safe yet still corrupt KV or speculative
state if their dependent layers interleave.

## Object and buffer lifetimes

The direct path uses RAII at every layer. The important relationships are:

| Object | Owns or retains | May be released when |
| --- | --- | --- |
| `MLABuffer` | One coherent userspace allocation | No live LLiMa model state needs the allocation |
| Backend `Buffer` | One imported dma-buf registration | No view, bound execution, or accepted job retains it |
| `BufferView` | A checked offset and byte extent into a `Buffer` | No binding or accepted job retains it |
| `ModelPackage` | A transactionally published ordered model set | No wrapper, binding, plan, or job retains its models |
| `BoundExecution` | One model plus all input/output views | No plan, segment snapshot, or job retains it |
| Backend `Job` | An accepted `JOB_EXEC` and its terminal result | Its continuation has returned and all holders release it |
| `MlaExecutionPlan` | Prebound jobs plus package, adapter, and allocation generations | Before package/buffer teardown or after it becomes stale |

Destroying a top-level wrapper does not invalidate storage already retained by
an accepted job. Conversely, retaining a userspace pointer is never used as a
lifetime mechanism.

### Checked views, not raw addresses

`MLABufferSlice` describes a logical slice in LLiMa. Before submission, the
session converts it to a Backend `BufferView` with a checked byte offset and
the compiler-required access extent. The Backend rejects overflow, out-of-range
access, cross-context objects, missing hidden inputs, and mismatched physical
port counts before a job reaches the kernel.

Each allocation has a generation cookie. Prebound plans record the cookies of
their parent buffers and the session's package/adapter publication generation.
A reallocated buffer or republished adapter therefore makes an old plan stale
before its first new submission.

### CPU and MLA ownership

LLiMa allocations used by direct `JOB_EXEC` are coherent and exportable as
dma-bufs. A `CpuAccessGuard` is still required when CPU code changes storage
that may also be referenced by accepted jobs:

```text
acquire CPU guard
  -> read or write the buffer
  -> end guard and publish CPU writes
  -> submit the immutable binding
```

Beginning CPU access fails while an accepted job owns the buffer. Submitting a
job similarly fails while a CPU guard is live. This rule protects embedding,
KV, checkpoint, adapter, and output storage without adding a hidden staging
copy.

## Transactional model loading

Related QMLAs are loaded as one ordered `ModelPackage`:

1. LLiMa resolves the model-family paths.
2. The Backend parses and validates every QMLA.
3. All models are defined in the kernel.
4. The package is published only if every model succeeds.
5. LLiMa eagerly builds exact default bindings for models whose ports are
   already known.

Failure leaves the previous package state unchanged and RAII releases all
objects created by the unsuccessful attempt. The ordered package identity is
important because the ordinary decode plan stores stable model ordinals.

Package and buffer teardown happens in the reverse order:

```text
execution plan
  -> model/binding wrappers
  -> package and imported buffers
  -> underlying LLiMa allocations
  -> Backend context
```

## Two execution styles

All model families use the same Backend and `JOB_EXEC` contract. LLiMa chooses
between two private orchestration styles.

### Transactional segments

Prefill, vision, Whisper, speculative decoding, and uncommon dynamic binding
paths build an `MlaExecutionSegment`:

```text
collect binding intent
  -> validate and snapshot the complete segment
  -> submit a bounded set of independent JOB_EXEC commands
  -> consume every terminal CQE
  -> return to the model
```

No job is submitted until the entire segment has validated. Abandoning a
segment before `commit()` is therefore a complete rollback, not a cancellation
operation. A committed segment uses the same bounded rolling executor for its
physical jobs, then joins it before returning. Ordinary decode extends that
executor across token positions instead of returning to the producer at each
position.

### Persistent ordinary decode

Ordinary, non-speculative language decode has a stable sequence of physical
models for every token position. Initialization flattens those recipes into
one private `MlaExecutionPlan`. Each position points to a span of prevalidated
`BoundExecution` objects.

The rolling executor keeps at most three physical `JOB_EXEC` commands
accepted. This is a window of compiled jobs, **not** a batch of tokens and not
a public queue-depth control. When one job completes, the Backend's sole CQ
consumer submits the next job into the released slot.

```text
submit jobs A, B, C

CQE A -> release A's buffer ownership -> submit D
CQE B -> release B's buffer ownership -> submit E
CQE C -> release C's buffer ownership -> submit F
...
```

The window depth is compiled into the runtime because it is an implementation
detail selected by qualification. Exposing it through LLiMa, Neat, or the
kernel would create a compatibility knob without changing model semantics.

When the last physical job for a token completes, the CQ owner:

1. reads the coherent scalar token output;
2. decides whether decode should continue;
3. writes the next embedding row under CPU ownership;
4. starts the next position's first `JOB_EXEC`; and
5. only then publishes the completed token to the producer thread.

Starting the successor before waking the producer avoids a completion-thread
round trip on the serial decode boundary. It does not create another
scheduler: each job still has its own terminal CQE, and the kernel still
arbitrates at every compiled-job boundary.

The implementation preserves the useful property observed in the pinned
MLA-RT multi-model integration: the thread that consumes completion can launch
the next ready work. It does so through the generic Backend continuation rather
than depending on MLA-RT or adding a model-sequence command to the kernel ABI.

## Completion and error handling

The Backend has one CQ owner and exactly one optional continuation per job.
The continuation is a `noexcept` function pointer so the hot path does not own
an allocator-backed callback and exceptions cannot escape through the
completion pump.

A completion continuation may submit replacement work. It must not:

- wait for this or another job;
- stop the Backend;
- perform synchronous model or buffer control operations; or
- outlive its callback context.

`drain_and_join()` is the callback-context lifetime join. It waits until every
accepted job is terminal and every registered continuation has returned.

Any submit or terminal failure in a dependent segment poisons the complete
session. The runtime drains authoritative completions, stops further
admission, and reports the error on the caller thread. It does not try to
continue from possibly mutated KV state or silently fall back to another
transport. Reconstructing the session is the recovery boundary.

Session destruction is the single no-throw shutdown path. It releases plans
and models, stops outstanding Backend work, releases the kernel context, and
reports cleanup failures through logging without throwing from a destructor.

## KV, checkpoints, and LoRA

### KV and state buffers

KV caches, convolution state, frequency tables, embeddings, and scalar outputs
remain ordinary LLiMa-owned `MLABuffer` allocations. The session imports them
once and reuses their checked views. The execution lease serializes CPU
checkpoint operations with dependent device work.

Coherent mappings remove the need for legacy cache-maintenance operations on
the qualified direct path, but they do not remove ordering. Terminal CQE
acquire ordering makes MLA writes visible to the CPU; ending a CPU write guard
publishes changes before the next submission.

### LoRA/RELOC adapters

Adapter changes are context-wide transactions:

1. wait until the session execution lease is available;
2. pack the candidate adapter tensors into the inactive dma-buf arena;
3. validate every compiler-named hidden port and checked view;
4. publish the complete new binding generation atomically; and
5. rebuild the ordinary decode plan for that generation.

Accepted jobs retain their old immutable views, so a new adapter cannot mutate
queued descriptor bindings. If the plan optimization cannot be rebuilt after
a successful adapter publication, the adapter remains valid and LLiMa falls
back to the checked generic recipe until a later cold rebuild. Unsetting an
adapter publishes base bindings before inactive adapter storage is scrubbed.

## Model-family behavior

The transport is shared, but each family retains its existing model semantics:

- **Text LLMs:** grouped prefill and exceptional paths use transactional
  segments; ordinary non-speculative decode uses the persistent plan when it
  is valid.
- **VLMs:** vision and projection work use transactional segments, followed by
  the same language-model execution rules.
- **Speculative decoding:** target/draft and verification boundaries remain
  explicit transactions. They do not use the ordinary persistent chain where
  CPU decisions or dynamic shapes break the stable recipe.
- **Whisper:** encoder, decoder initialization, language detection when
  present, and iterative decode use direct transactional segments.

The direct transport does not reinterpret graph order, tensor contents,
stopping rules, sampling, or model configuration.

## Whisper 2.0.0 compatibility

One published Whisper 2.0.0 decoder-init layer-0 QMLA exposes three physical
inputs:

1. token embeddings;
2. position embeddings; and
3. encoder features.

Current packages expose two physical inputs. During model construction, LLiMa
reads the compiler-authoritative public-port count and binds the matching
layout. It does not guess from equal byte extents, a package filename, or a
runtime version.

This compatibility branch can be deleted only after:

- the published package has been regenerated with the current two-port
  contract;
- all supported package catalogs no longer contain the three-port artifact;
  and
- direct Whisper qualification passes with the regenerated package.

Until then, both layouts are intentional product inputs.

## Designs intentionally not used

The direct path avoids several larger mechanisms because they duplicate an
existing owner or reduce scheduling flexibility:

- **No kernel DAG or model-sequence command.** LLiMa already knows the model
  graph. Independent `JOB_EXEC` commands keep priority arbitration at every
  compiled-job boundary.
- **No all-at-once token submission.** The next token is not known until the
  scalar output is complete and stopping rules run.
- **No one-enter full-chain publication.** The qualification prototype
  prepared a complete ordinary-token chain before one SQ publication. That
  moved preparation onto the serial token boundary instead of overlapping it
  with MLA execution, so the experiment was slower and was removed.
- **No aggregate sequence command.** A private prototype replaced the
  per-layer jobs with one flat transaction and one terminal result. Atomic
  construction again delayed first `GO`, reduced priority boundaries, and
  performed worse than the rolling `JOB_EXEC` path.
- **No public queue-depth setting.** The small rolling window is a measured
  executor detail, not application policy.
- **No per-job priority.** Priority belongs to the kernel context; changing it
  inside a dependent model sequence would complicate ordering and admission.
- **No second completion fence or reap call.** The retained io_uring command
  and its terminal CQE already own completion and telemetry.
- **No runtime binding fast-path fork.** `BoundExecution` is the only submit
  operand. Dynamic bindings use the same validation and lifecycle.
- **No MLA-RT fallback.** Failing to open or use the direct Backend is reported
  as an error rather than selecting a second execution engine.

## Qualification map

The direct path has focused DVT tests under `tests/dvt/`:

| Test | Main contract |
| --- | --- |
| `llima_direct_session_smoke` | Session creation, checked slices, buffer generations, and failure behavior |
| `llima_direct_resnet_depth2` | Multiple accepted jobs, context isolation, and transactional segment rollback |
| `llima_direct_text_model` | Text model output and ordinary direct execution |
| `llima_direct_vlm_benchmark` | Fixed-length VLM output, timing evidence, and persistent decode |
| `llima_direct_whisper` | Direct Whisper model execution and output |
| `analyze_llima_direct_vlm_benchmark.py` | Reproducible benchmark-record validation |

The test programs are target binaries. Inspect them with `file` and run
AArch64 artifacts on a DevKit rather than executing them on an x86 build host.

## Known release gaps

The architecture is intentionally frozen, but release qualification still has
package- and product-level work:

- complete the required long-duration token campaign for every advertised
  text, vision-language, and speech product member;
- report long-context performance as a position curve, because cache-attention
  QMLA cost grows with the compiled context and is not a kernel handoff cost;
- retain the Whisper compatibility branch until the published package is
  regenerated and qualified; and
- re-run the relevant DVT, failure, priority, and sanitizer suites whenever a
  semantic lifetime or ordering change is made.

These are qualification or compiled-package concerns. They are not reasons to
add another runtime scheduler or completion API.

## Source map for reviewers

Start with these files:

- `sima_lmm/devkit/cpp/mla_model.hpp` — public session and segment contracts.
- `sima_lmm/devkit/cpp/mla_model.cpp` — package loading, binding, rolling
  execution, poisoning, and shutdown.
- `sima_lmm/devkit/cpp/mla_buffer.*` — coherent allocation and checked logical
  slices.
- `sima_lmm/devkit/cpp/mla_execution_plan.hpp` — private ordinary-decode plan.
- `sima_lmm/devkit/cpp/language_model.cpp` — persistent decode, KV/checkpoint
  ordering, and adapter publication.
- `sima_lmm/devkit/cpp/whisper_model.cpp` — the temporary two-/three-port
  decoder-init compatibility boundary.
