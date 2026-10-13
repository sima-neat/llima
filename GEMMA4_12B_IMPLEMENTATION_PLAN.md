# Gemma 4 12B Unified: implementation plan

## Proposed approach

Target Gemma 4 12B Unified directly: add its configuration and decoder
differences, then its image embedder, image-window scheduling and audio frontend.
MTP remains outside this implementation.

Keep the existing aligned n128 grid and use two execution cases:

- **Outside image windows:** normal full forward pass for one n128 chunk.
- **Inside a multi-chunk image window:** process its chunks together, layer by layer.
  Run all PRE chunks for that layer before their attention/POST chunks.

Only the current image window needs extra hidden/Q staging. Reuse that memory
for subsequent windows and prompts. Do not stage the whole uncached suffix.
Use existing PRE/POST components and dynamic-mask global/sliding cache parts;
no cross-attention graph, arbitrary chunk starts, physical KV guard rows or
new checkpoint family is needed. Generated text retains n1 decoding.

Worktree: `llima-gemma4-12b`; branch: `feature/249-gemma4-12b-unified`.
Related issue: [#249](https://github.com/sima-neat/llima/issues/249).
Initial configuration: context C=4096, group G=128, window W=1024.
Initial image budget M=280; use each processed image's actual token count.

## 1. Reference contract

Pin the actual reference execution path. Our BF16 GPU runs used Transformers
5.16.1 with SDPA and dynamic caches. Gemma 4 Unified's active `forward()` calls
generic `create_masks_for_generate()` with vision block IDs, allowing
bidirectional image attention in **both sliding and global layers**. Its
separate `create_masks_for_vision_model()` helper instead keeps global layers
causal. The earlier plan mistakenly treated that helper as the active behavior.

For valid query/key positions q/k, `same_image(q,k)` means both are soft image
tokens in the same contiguous block. Text, delimiters and audio have no image
block ID. The installed active forward uses:

```python
global_allowed = k <= q or same_image(q, k)
sliding_allowed = (k > q-W and k <= q) or same_image(q, k)
```

Intersect with valid keys. Initially require M<=W, including the 280-token
budget; larger budgets need a separately validated KV-view bound. Do not add
an upper-window bound on future same-image keys.

Preserve start/end-image delimiters and derive block IDs from the final
expanded prompt. One image cannot attend to a later image; later positions
can attend causally to earlier images. This is decoder self-attention.
Implement the pinned active-forward policy. Selecting the alternative helper
would change model semantics and requires an explicit decision.

Reference checkpoint:
`/project/mlasw/share/huggingface/models--google--gemma-4-12B-it`, revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
Use its [configuration](https://huggingface.co/google/gemma-4-12B-it/blob/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7/config.json)
and the installed Transformers source, recording hashes in comparisons.
The [pinned upstream source](https://github.com/huggingface/transformers/blob/536ecc007387a50e77603bb5d92100e9b07514cc/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py)
is supplementary, not byte-identical to the installed reference.

## 2. Select the smallest complete image windows

Record actual image spans `[begin,end)` after placeholder expansion. A
256-token image can touch three n128 chunks; a 280-token image can touch four.
The token count comes from Unified preprocessing, not a fixed encoder output.

Round each span outward to G and merge **overlapping** rounded intervals:

```python
windows = []
for begin, end in image_spans:  # sorted by prompt position
    begin = (begin // G) * G
    end = round_up(end, G)
    if windows and begin < windows[-1][1]:
        windows[-1] = (windows[-1][0], max(windows[-1][1], end))
    else:
        windows.append((begin, end))
```

Do not merge intervals that merely touch. They can run separately and reuse
the same buffers. Overlap must be merged: finishing a shared n128 chunk would
otherwise also process the beginning of another image before its future keys
exist. This small helper is sufficient; no general scheduler abstraction.

Examples, using half-open positions:

| Image spans | Execution windows |
| --- | --- |
| `[576,704)` | `[512,768)` — two chunks |
| `[577,833)` | `[512,896)` — three chunks |
| `[1663,1943)` | `[1536,2048)` — four chunks |
| `[577,833)`, `[834,1090)` | `[512,1152)` — five chunks, merged |
| `[512,768)`, `[896,1152)` | Two separate two-chunk windows |

Reuse matching text only before the first image initially. Round that count
down to G. Require C to be a multiple of G and prompt length <= C; separately
check any needed generation room. All padded PRE writes stay inside the
existing C-row KV/RoPE allocation. Only final prompt padding is invalid.

## 3. Execute and reuse buffers

Walk the uncached prompt in order:

```text
while cursor < padded_prompt_end:
    if a multi-chunk image window starts at cursor:
        run_image_window(cursor, window_end)
        cursor = window_end
    else:
        run_model_once(n128, cursor)  # normal full forward pass
        cursor += 128
```

Normal text chunks keep their existing chunk-major queueing. An image entirely
inside one chunk can also use that full-forward path, with its image mask:
PRE already supplies all of that image's keys. For regenerated Gemma artifacts,
cache binding/mask helpers use the capacities below, causal masks for text,
and same-image exceptions where required. Legacy artifacts keep their old path.

For an image window, copy just its prepared embeddings into reusable hidden
staging, zero final prompt padding if present, and execute:

```text
for each decoder layer:
    for each n128 chunk in this window:
        PRE(layer, hidden[chunk], absolute RoPE position)
        retain Q[chunk]; write K/V into the persistent cache at that position
    execute PRE phase and wait

    for each n128 chunk in this window:
        attention(Q[chunk], K/V view, mask[chunk])
        POST(hidden[chunk], attention_output) -> hidden[chunk]
    execute attention/POST phase and wait
```

One hidden buffer suffices: all PRE reads finish before POST, and POST only
needs its own original residual rows. Existing middle-layer POST already
aliases input/output. Retain Q until attention consumes it; one Q slot would
be overwritten by the next PRE. Rebind PRE hidden/Q, cache Q/K/V/mask, and
POST residual/output explicitly, reusing existing wrapper/binding helpers.

At each layer, earlier prompt K/V are already cached and every image in the
current window has complete current-layer K/V before attention. Masks exclude
future text and other future images. Complete the window through all layers,
then reuse its hidden/Q buffers for the next chunk or window.

Keep final-layer optimization: PRE writes K/V for every valid row, but only
the final prompt row needs final attention/POST/head. Reuse its n1 POST/head
and row bindings. Do not generate n128 vocabulary logits.

Advance the processed prefix only after a full chunk/window finishes. If
interrupted inside a window, retain only the prefix before it. Its partial KV
writes are uncommitted, masked and overwritten on restart. Cross-request reuse
still stops before the first image; identical placeholders do not identify
identical media. No media hashing or new checkpoint families initially.

## 4. Static cache shapes, masks and memory

Extend `LanguageCacheModel` with explicit dynamic-mask selection and optional
context capacity, preserving old defaults. Replace the static causal mask;
adding a dynamic mask afterward cannot unmask future image keys. Reuse existing
ModelGraph matmul/softmax, GQA handling and BMM2 segmentation.

Use one dynamic n128 global/sliding executable per geometry. For M<=W:

```python
K_global = C
K_slide = min(C, round_up(W + G + M - 2, G))
kv_begin = min(max(0, chunk_start - W + 1), C - K_slide)  # global: 0
```

The sliding capacity covers W-1 past keys plus future keys of an image that
starts at the chunk's last row. It is 1408 for M=256, 1536 for M=280, or 1792
for M=560. Starting the view at `group_end-W` drops keys needed by earlier
queries. Keep physical KV head stride C in PRE/cache descriptors and use
signed position arithmetic.

Gemma 4 sliding shapes: Q `[1,16,128,256]`, K/V `[1,8,1536,256]`, mask
`[1,1,128,1536]`, output `[1,1,128,4096]`. Global: Q `[1,16,128,512]`, K/V
`[1,1,4096,512]`, mask `[1,1,128,4096]`, output `[1,1,128,8192]`.
Ordinary chunks use causal masks on the same shapes. No third image-cache part
is required. Keep existing n1 causal selection with updated head geometry.

Generate masks when a window is reached and retain only that window's masks.
Reuse them across layers and overwrite after the window finishes. Normal
chunks use slot zero of each mask pool and reuse it after their queued full
forward finishes, matching develop's buffer lifetime. Separate global/sliding
pools keep normal chunk queueing unchanged. Mask uncommitted/padded keys;
give discarded padding query rows one valid dummy key to avoid all-masked
softmax. Validate the existing finite additive-mask convention in BF16.

`add_to_queue()` snapshots bindings, not buffer contents. Each chunk queued
in the same phase needs its own mask slot. Retaining masks for the small active
window avoids an extra wait/upload per chunk at every layer. There is no need
to retain masks for all prompt chunks.

Before prefill, determine the largest merged window, R rows. Grow hidden/Q
and mask pools once if needed; reuse them for all windows/layers and subsequent
requests. No allocation inside the layer loops. R is runtime scratch sizing,
not another compiled graph size. If there are no images, no image hidden/Q
staging is needed.

Pack each Q chunk as `[query_heads,G,head_dim]` in one flat allocation sized for
the largest head dimension. At a layer with dimension D, the chunk's element
offset is `chunk_index * query_heads * G * D`. Bind flat `MLABufferSlice`
offsets; ELF descriptors define layout. Do not slice `[heads,R,D]` as n128
chunks, because its head stride is different. Validate offsets in compiled
artifacts. Existing attention-output scratch is reused by ordered cache/POST.

For a **single image**, `R <= round_up(M + G - 1,G)`. At M=280, R<=512:

Four chunks are a worst-case alignment bound, not the normal allocation. For
example, an image at `[127,407)` occupies 1/128/128/23 tokens in four chunks.
An aligned 280-token image occupies only three. Allocate using the actual
rounded spans and largest merged window; do not always reserve four chunks.

| Reusable working buffer, Gemma 4 12B | Math | MiB |
| --- | --- | ---: |
| Hidden | `512 * 3840 * 2` | 3.75 |
| Q, maximum head dimension | `512 * 16 * 512 * 2` | 8 |
| Global masks, four slots | `4 * 128 * 4096 * 2` | 4 |
| Sliding masks, four slots | `4 * 128 * 1536 * 2` | 1.5 |
| Total listed buffers | | **17.25** |

The previous 138-MiB whole-suffix allocation is removed. Separated images
reuse these same buffers, independent of how much text surrounds them. Closely
spaced images may share a chunk and require a larger merged window; size by
that actual window, not a fixed four-chunk assumption. A long overlapping
image chain can still grow large. This is an explicit bound of the simple
aligned-chunk design, not a reason to preallocate the entire context.

The table excludes existing n128/attention scratch and persistent assets.
BF16 KV is still 1344 MiB at 4K; BF16 scaled text embeddings are 1920 MiB;
position table about 16.4 MiB. Check full device fit with weights/ELFs. No
per-layer host activation transfers, KV ring, or extra cache types are needed.

## 5. Gemma 4 12B configuration and decoder differences

| Property | Required behavior |
| --- | --- |
| Decoder | 48 dense layers, hidden 3840, MLP 15360 |
| Pattern | Five sliding layers then one global, repeated eight times |
| Q heads | 16 for both types |
| Sliding K/V | 8 heads, dimension 256, window 1024 |
| Global K/V | 1 head, dimension 512 |
| Attention score scale | 1.0 |
| RoPE | Local: 256 channels, theta 10000; global: 128 of 512 channels, theta 1000000 |
| Per-layer embedding branch | Absent, width 0 |
| Layer scalar | Required after every decoder layer, despite the absent branch |
| Head | Tied 262144-row embedding matrix; final logit softcap 30 |

In `config/vlm_config.py`, recognize `gemma4_unified`/`gemma4_unified_text` and
normalize raw fields plus Transformers `per_layer_config`. Add per-type KV
head-count access alongside existing head-dimension access. Validate consistent
geometry within a type and Q-head divisibility. Update Python and C++ shape,
KV allocation, tessellation and component-selection callers together. Retain
old values when overrides are absent. Add optional vision-attention capability
metadata to regenerated Gemma artifacts; legacy artifacts retain their old
runtime path when that field is absent.

Resolve original safetensors names in `hf/hf_transformer.py` and the existing
source resolvers. Decoder keys already use `model.language_model.layers.*`.
Validate weight dimensions and explicitly resolve absent global V weights:

```text
input norm -> k_proj -> weighted K RMSNorm -> RoPE -> cached K
                    -> weightless V RMSNorm       -> cached V
```

Global raw K projection is shared with V; the resulting cached tensors are
**different** and must not share storage. Sliding V retains its own projection.
Reuse existing Gemma PRE/POST math, direct norm weights, tanh-GELU gated MLP,
residuals, final norm and softcap. Extend the existing Gemma 4 architecture
guards in PRE/POST and the final-head helper to include Unified. In
`language_post_model.py`, apply layer scalar outside the optional per-layer
embedding branch, exactly once. Actual scalar
values span 0.0045471–0.886719; omitting them is not a small perturbation.

Do not generate zero-width per-layer embedding assets. In
`LanguageModel.get_embeddings_tensor()`, reproduce the unified BF16 scalar
rounding: sqrt(3840) rounds to 62 **before** multiplying BF16 text embeddings.
Media embeddings and the original tied-head matrix remain unscaled. Handle
out-of-vocabulary media placeholders without indexing the text embedding table.

Mirror the new fields/geometry in `vlm_config.hpp`,
`language_model_definition.cpp` and runtime bindings. Stay within the runtime's
existing position-width limits. Export the exact tokenizer/template/generation
configuration and apply EOS/suppression rules, not just copy their files.

## 6. Unified image embedder and preprocessing

Gemma 4 12B has no SigLIP tower; add one small ModelGraph component:

```text
raw merged RGB patches [G,6912]
  -> affine LayerNorm (eps 1e-5)
  -> Linear 6912->3840, with bias
  -> affine LayerNorm (eps 1e-5)
  -> add gathered factorized position embeddings [G,3840]
  -> affine LayerNorm (eps 1e-5)
  -> weightless RMSNorm (eps 1e-6)
  -> Linear 3840->3840, without bias
```

Raw checkpoint weights use `model.vision_embedder.{patch_ln1,patch_dense,
patch_ln2,pos_embedding,pos_norm}` and
`model.embed_vision.embedding_projection.weight`. Do not assume Transformers'
checkpoint rename has been applied by LLiMa. The factorized position table is
`[1120,2,3840]`; gather/sum its two axes once per image on the host, preserving
reference BF16 rounding. Feed the resulting embeddings as a second graph
input. No MLA dynamic gather is needed.

Match the native processor: RGB rescale to [0,1], aspect-preserving BICUBIC
resize, dimensions aligned to 48, and exact constituent 16x16 patch/3x3 merge
ordering. Validate pixel/channel order, not just patch shapes. Under M=280,
a square becomes 768x768/256 tokens; 480x640 becomes 672x912/266 tokens.
Do not force a 448x448 processed input or mistake the budget for actual count.

Preprocessing returns patches, positions and valid count. Compile the embedder
at G rows and invoke it repeatedly, padding only its final call. Expand the
chat placeholder using the **valid** count; preserve image delimiters; copy
only valid projected rows into the prompt. Associate each span with its media
input and validate placeholder/embedding counts. Extend the existing vision
wrapper for a second input while preserving legacy one-input callers.

Package the position table, embedder ELF and preprocessing parameters through
existing compile/deploy mechanisms. Derive budget from processor configuration;
an optional override must update graph capacity and runtime metadata together.
Initially allow only M<=W and resident BF16 embeddings/KV for this new path.
Do not let embedding offload overwrite staged media or pretend unsupported
quantized-cache/embedding bindings work. Other model paths retain their options.

## 7. Audio frontend

Gemma 4 audio is also encoder-free: mono 16-kHz waveform -> zero-padded 640-sample frames ->
weightless RMSNorm (eps 1e-6) -> Linear 640->3840, without bias. The weight is
`model.embed_audio.embedding_projection.weight`. Valid token count is
`ceil(sample_count/640)`; the zero-padded final frame is still valid. Initial
budget is 750 frames/30 seconds. Audio tokens remain causal: the Transformers
processor marks audio as modality 3; modalities 1/2 are image/video.

Add the audio ModelGraph component, preprocessing, chat/template attachment
handling and runtime embedding insertion together. Normalize actual tokenizer
fields such as `eoa_token_index`; preserve text/image/audio ordering. Start with
mono 16-kHz PCM WAV input and reject unsupported formats or excessive length
clearly. Reuse shared file decoding and resampling from Whisper; keep its mel
spectrogram processing separate. Advertise audio capability
only once its compiled component and end-to-end runtime path are validated.

## 8. Implementation and acceptance order

1. **Config/ingestion and text:** add Unified identification, per-type head
   geometry, global projection sharing, layer scalar and embedding scaling.
   Compare sliding layer 0, global layer 5 and final layer 47 to Transformers.
2. **Image input:** implement the Unified embedder and native preprocessing;
   compare pixels, patch order, positions and projected embeddings for square
   and non-square images. Record actual spans in the expanded prompt.
3. **Graphs and bindings:** add dynamic n128 cache masks/capacities and runtime
   geometry. Register through `config/layer_id.py` / `get_layer_ids()` and the
   existing factory; keep Python/C++ selection and physical strides consistent.
4. **Image scheduling:** add the small aligned-window helper and runtime path.
   Mirror it in `model/language_model.py` for host comparison using identical
   projected image embeddings. Compare masks, activations, K/V and first-token
   logits before evaluating generated prose.
5. **Device bring-up:** compile/deploy Gemma 4 12B using a weight precision that
   fits Modalix. Exercise unaligned image starts, 512 cached text tokens,
   overlapping/touching/separate image windows, changed images with identical
   placeholders, shorter prompts, interruption and the final partial group.
   Validate Q/K/V offsets, queue order, TTFT, TPS and peak DRAM.
6. **Audio:** complete its separate input/component/runtime path and compare
   639/640/641-sample framing plus representative audio prompts.
7. **Regression/release:** run affected required tests and representative Gemma
   E2B/E4B and non-Gemma GQA compilation. Document supported modes and update
   model-support docs/CI manifests after their runtime acceptance gates pass.

Keep all changes capability-gated and reuse existing ModelGraph, component
registration, binding and packaging helpers. Do not add general-purpose
scheduler abstractions or new public cache types for this implementation.

## 9. Design verification

Checked on 2026-10-10. These checks validate the proposal, not an implementation.

- **48 reduced 12-layer comparisons with the Unified mask policy:** mixed normal-chunk/image-window
  execution matches full-sequence outputs and persistent K/V, maximum output
  difference 1.56e-15 in float64. Covers cached text, all toy chunk offsets,
  two images separated by one text token, separate/touching/overlapping
  windows, a long image chain, audio, text-only input and final padding.
- **All 128 single-image offsets:** M=256/280/560 needs at most 3/4/6 n128
  chunks. Actual two-image and chain examples confirm overlapping-window
  growth; the four-chunk bound applies only to a single M=280 image.
- Previous mask checks: 9,437,184 entries against the Unified active-forward
  builder, including a 1120-token block exposing its window composition.
  The separate helper policy was audited independently.
- Previous KV-view checks: 2,286,904 query rows at history/window/context
  boundaries, capacities 1408/1536/1792 fitting the existing C-row allocation.
- Source audit confirms existing in-place POST and queued binding snapshots;
  actual flat Q offsets and strided KV layouts still need compiled validation.
- Checkpoint audit confirmed all 48 geometry/scalar contracts and embedder
  weights. GPU quality comparisons showed the fully causal attention risk;
  they did not execute this proposed runtime schedule.

Window experiment: `/project/scratch/tmpllm/verify_gemma4_image_windows.py`;
results: `/project/scratch/tmpllm/gemma4-image-window-verification.json`.
Earlier mask/capacity experiment:
`/project/scratch/tmpllm/verify_gemma_n128_plan.py`.
GPU reports are in `gemma_vision_attention_ablation/` in the parent workspace.

Still required: actual BF16 checkpoint comparisons, compiler/simulator
acceptance, and Modalix correctness, queue performance and peak DRAM checks.
No runtime implementation or complete model support is claimed yet.

## 10. Implementation validation (2026-10-12)

Implemented on `feature/249-gemma4-12b-unified`: configuration and heterogeneous
attention geometry, encoder-free image projection, aspect-preserving image
preprocessing, dynamic prefill masks, and the C++ image-window schedule. Text
retains ordinary n128 prefill/n1 decode. Audio and MTP remain follow-up work.

Completed checks:

- All **371 compiler unit tests** passed; four cached Gemma/Qwen configuration
  regression cases passed.
- Actual BF16 source weights executed through **ModelGraph.run** for n128
  sliding/global PRE, attention, POST and the image embedder: maximum relative
  L2 error **1.60e-6** against Transformers. Final layer 47, final norm and the
  split LM head matched at **7.80e-7**, with identical argmax.
- Seven reduced decoder scenarios covered text, separate/overlapping image
  windows, boundary crossings, partial final chunks and the context limit.
  A prefill-plus-n1-decode comparison also produced identical first three
  tokens; relative L2 error **2.41e-5**.
- Actual C++ preprocessing produced bit-identical BF16 patches and XY IDs for
  square, landscape, downsampled and extreme-aspect images. Its window/mask
  helpers matched the Python mirror.
- Six components compiled from our packed GPTQ checkpoint with MLA simulator
  checks: n128 PRE layers 0/5, dynamic global/sliding caches, n1 POST layer 47
  including the INT4 LM head, and the INT8 image embedder. This is a component
  validation build, not a complete deployable model.
- All nine changed/new runtime translation units passed AArch64 syntax checks.

Evidence and component artifacts:
`/project/scratch/tmpllm/gemma4-12b-implementation/`.

Still required: a full runtime link/build with the current SDK, complete-model
compilation, and Modalix correctness/DRAM/queue measurements. The available
SDK container reports 2.1.2 and its OpenCV CMake package references a missing
`libopencv_flann.so.4.6.0`; runtime linking could not be validated with it.

## 11. Audio implementation and shared preprocessing (2026-10-12)

Added the standalone n128 waveform RMSNorm/projection graph, source/processor
configuration, audio prompt expansion, runtime embedding insertion and
`add audio <file>` CLI attachments. Image/audio attachment order is preserved;
audio uses ordinary causal prefill. The frontend reuses the image output
scratch buffer, with a separate 160-KiB audio input buffer.

`audio_preprocessing.cpp` and `preproc/audio_preprocessing.py` now provide the
shared mono 16-kHz file decoder. Whisper keeps its 30-second pad/trim and mel
spectrogram. Gemma retains the actual sample count, rejects over-budget input
and pads only its final 640-sample frame. The separate WAV parser was removed.

Validation:

- **374 compiler unit tests**, including 639/640/641-sample framing and audio
  ModelGraph comparisons, passed. Four cached configuration cases passed.
- Actual BF16 checkpoint audio weights, evaluated in FP32 through ModelGraph,
  matched Transformers at **3.85e-7 relative L2**.
- Audio insertion across an n128 boundary and the reduced full decoder matched
  the reference at **7.60e-7**, with identical argmax. Both paths used the same
  BF16 text-embedding asset; audio remained modality 3.
- Native C++/Python BF16 frames were bit-identical for 639, 640, 641 and 16,000
  samples. Padded Whisper input matched its former C++ loader exactly for
  these PCM cases. The 480,001-sample case preserved Whisper truncation and
  rejected excess audio for Gemma.
- Changed audio/Whisper/chat/runtime translation units passed AArch64 syntax
  checks. The audio graph also compiled successfully; Modalix execution remains
  pending.

Full 4K/n128 compilation was initially stopped at the user's request, then
resumed after regenerating DEVKIT assets for audio. All **221 SDK graphs and
221 corresponding MPKs** are present, and compilation finished with exit code
**0**. The output is under
`CompiledModels/Gemma-4-12B-it-GPTQ-a16w4-4k-n128`, backed by
`/project/scratch/tmpllm/gemma4-12b-full-4k-n128-20261012/output`.
MTP remains excluded.

## 12. Audio chat-completion requests (2026-10-12)

`/v1/chat/completions` accepts OpenAI `input_audio` parts with base64 `data`
and `format` (`wav` or `mp3`). These become internal audio attachments, with
text/image/audio ordering preserved. The common decoder reads uploaded bytes
through an anonymous temporary file that is closed on success or failure;
CLI file inputs still use the same decoding and framing path.

Models without audio support and malformed audio fields return request errors.
Worker exceptions now return errors rather than terminating the server, with
SSE errors for streaming requests. The separate Whisper transcription endpoint
is unchanged. Usage is documented in `docs/runtime.md`.

Validation: affected translation units passed AArch64 syntax checks. Focused
native probes verified the actual endpoint conversion and Gemma chat template,
mixed attachment ordering, text/image regressions, matching file/upload frames
for WAV, MP3 and stereo 32-kHz input, invalid inputs and descriptor cleanup.
Packaged HTTP/MLA execution on Modalix remains required.
