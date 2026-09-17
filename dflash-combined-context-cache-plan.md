# DFlash Combined Context Graph and Packed Draft KV Cache

## Objective

Replace the current DFlash context synchronization path with one graph per
token width. The graph will fuse selected target hidden states and produce the
K/V cache for every draft layer. Store the result in a packed, persistent draft
cache.

For the public Qwen3-4B DFlash checkpoint this changes each synchronization
from one fusion graph plus five per-layer context graphs to one graph:

```text
Current:  target hidden states -> fusion -> five K/V graphs -> 20 outputs
Proposed: target hidden states -> fusion -> one packed K/V projection -> 2 outputs
```

This work applies only to a DFlash draft. Target models, ordinary generation,
and EAGLE3 keep their existing cache layout and model graphs.

## Current path

`LanguageDraftFCModel` accepts the target hidden states selected by
`dflash_config.target_layer_ids`, concatenates them, applies the checkpoint's
`fc` projection and `hidden_norm`, and writes `fc_nN_output`.

`LanguageDFlashContextModel` then runs once for every draft layer. Each graph
reads the same fused output and the RoPE tensors, projects K and V, and writes
that layer's cache. With quantized K/V caches, every layer produces four
outputs: K, K scale, V, and V scale.

For Qwen3-4B DFlash, five selected target layers feed a five-layer draft. Each
context synchronization therefore launches six graphs and materializes a
fused tensor that is subsequently read five times.

Context synchronization happens in both places below:

1. After each N128 target prefill chunk, to construct the draft cache for the
   prompt.
2. After every N8 or N16 target verification block, to replace the draft cache
   rows with context derived from the target hidden states. The physical graph
   processes the fixed block width; only committed rows advance the logical
   cache length.

## Packed cache layout

Fold K/V, draft layer, and KV head into one head axis. Let:

- `L` be the number of draft layers;
- `H` be the number of KV heads per draft layer;
- `T` be the maximum context length;
- `D` be the KV head dimension.

Allocate two persistent buffers for a quantized draft cache:

```text
dflash_cache_kv:       [2 * L * H, T, D]  INT8
dflash_cache_kv_scale: [2 * L * H, T, 1]  BF16
```

The head-axis offsets are:

```text
key_offset(layer)   = layer * H
value_offset(layer) = (L + layer) * H
```

For the current Qwen3-4B draft (`L=5`, `H=8`, `T=2048`, `D=128`):

```text
dflash_cache_kv:       [80, 2048, 128]  # approximately 20 MiB
dflash_cache_kv_scale: [80, 2048, 1]    # approximately 320 KiB logical payload
```

The data and scale tensors require separate allocations because their element
types differ. HWC16 row alignment makes the physical scale allocation larger
than its logical payload. Together the two buffers form one logical packed
cache.

## Combined graph

Add a combined DFlash context model with one graph for each required width,
initially N128 and the selected verification width.

For Qwen3-4B, the graph has seven inputs:

```text
target_hidden_0 ... target_hidden_4: [1, 1, N, hidden_size]
freq_real:                            [1, 1, N, rope_dim / 2]
freq_imag:                            [1, 1, N, rope_dim / 2]
```

Its internal calculation is:

1. Concatenate the selected target hidden states.
2. Apply the DFlash `fc` projection.
3. Apply `hidden_norm` once.
4. Concatenate every draft layer's K and V weights along the output-channel
   axis and apply one packed K/V projection.
5. Reshape the projection output into the folded layer/head layout and split
   its K and V regions.
6. Apply any model-specific, layer-specific K normalization, then apply RoPE
   to the packed K region.
7. Dynamically quantize packed K and packed V while retaining one scale per
   head and token.
8. Concatenate quantized K followed by quantized V, and concatenate their
   scales in the same order.

Let `KVD = num_key_value_heads * head_dim`. The packed projection has shape:

```text
hidden_size -> 2 * num_draft_layers * KVD
```

For the two initial checkpoints:

| Draft | Packed K/V projection | Parameters | W8 payload |
| --- | ---: | ---: | ---: |
| Qwen3-4B | `2560 -> 10240` | 26.2M | 25 MiB |
| Llama 3.1 8B | `4096 -> 10240` | 41.9M | 40 MiB |

This projection replaces ten existing K/V projections. It does not add
weights or arithmetic. Together with the existing fusion FC, the combined W8
weight payload is approximately 56.25 MiB for Qwen3-4B and 120 MiB for Llama
3.1 8B.

It has two graph outputs:

```text
packed_kv:       [1, 2 * L * H, N, D]  INT8
packed_kv_scale: [1, 2 * L * H, N, 1]  BF16
```

The packed activation quantization must retain the current per-head,
per-token scale behavior. It must not introduce one scale shared by all layers
or by the full packed tensor.

The compiler's W8 weight quantization is also a correctness boundary. A single
concatenated weight tensor is equivalent when weight scales are per output
channel. If the compiler uses a scale shared by an entire weight tensor, the
builder must preserve the original K/V quantization domains explicitly. In
that case, keep the K/V projections as separate nodes inside the combined
graph rather than changing model accuracy merely to obtain one projection.

If the compiler cannot efficiently concatenate K and V into two outputs, the
bounded fallback is four packed outputs: all K, all K scales, all V, and all V
scales. Do not fall back to twenty external outputs unless required for a
minimal compiler experiment.

## Model compiler changes

### Combined model builder

Replace the DFlash use of `LanguageDraftFCModel` plus
`LanguageDFlashContextModel` with a combined builder, for example
`LanguageDFlashCombinedContextModel`.

The new builder should reuse the existing projection, RMSNorm, RoPE, and
dynamic-quantization helpers. It must load:

- `fc` and `hidden_norm` from the DFlash checkpoint;
- K/V weights from every draft layer and concatenate them in deterministic
  K-layer order followed by V-layer order;
- layer-specific K-normalization weights where the architecture defines them;
- the configured target-layer count rather than assuming five inputs.

Construct one packed K/V Conv/FC node when the selected precision preserves
the current per-output-channel weight scales. Split or reshape its output only
after the projection. Llama can apply RoPE to its complete packed K region.
Qwen must preserve its layer-specific K normalization before the packed RoPE
operation; use separate normalization operations if a folded operation cannot
represent the checkpoint exactly.

Keep `LanguageDraftFCModel` available for EAGLE3. Remove its DFlash-only branch
and the old per-layer DFlash context builder after the combined path passes
numerical and Modalix validation.

### Layer IDs and precision

Update `sima_lmm/config/layer_id.py`, `sima_lmm/config/vlm_config.py`, and
`sima_lmm/model/language_model.py` so a DFlash draft contributes one combined
context layer ID per width instead of one draft-FC ID plus one context ID per
draft layer.

The combined graph contains weights that currently belong to several layer
IDs. Require one common activation/weight precision for the DFlash context
path, or reject inconsistent per-layer precision settings with an actionable
error. Do not silently choose one layer's precision.

Use deterministic artifact names such as:

```text
<draft>_n128_dflash_context_all_layers
<draft>_n8_dflash_context_all_layers
<draft>_n16_dflash_context_all_layers
```

### Output tessellation

Give the combined graph custom output tessellation parameters:

```python
kv_dram_shape = (
    1,
    2 * num_draft_layers * num_key_value_heads,
    max_num_tokens,
    head_dim,
)

scale_dram_shape = (
    1,
    2 * num_draft_layers * num_key_value_heads,
    max_num_tokens,
    1,
)
```

Use the existing HWC16 cache layout. Bind an N-token graph output at the
runtime token offset in the larger `max_num_tokens` allocation.

This is a strided OFM store, which the existing pre-model cache output path
already uses. It should not require the separate compiler change for cropped
strided IFM loads. Confirm this with a small compile before changing compiler
code.

## Runtime changes

### Buffer allocation

In the DFlash draft path of `language_model.cpp`, replace the per-layer K, V,
K-scale, and V-scale allocations with the two packed buffers. Leave the target
and all non-DFlash models unchanged.

Keep the packed layout private to `LanguageModel`; do not add fields to the
installed public interface solely for this optimization.

### Per-layer cache views

Ordinary draft pre/cache/post graphs remain per-layer graphs. Bind their K/V
inputs and outputs to slices of the packed allocations using the head offsets
defined above.

Each `LanguageCacheModel` continues to see its existing logical shapes:

```text
keys or values: [H, T, D]
scales:         [H, T, 1]
```

Its current tessellation parameters and DRAM shapes therefore remain
unchanged. The runtime base offset selects the layer and K/V region; lower-axis
strides are identical to the current per-layer buffers.

Apply the same slice binding to the K/V outputs of the regular draft
`LanguagePreModel`. Those graphs do not need to change.

### Context synchronization

Replace the body of `_append_dflash_context()` that queues one fusion model and
one model per draft layer. The new path should:

1. Select the combined graph by token width.
2. Bind its target hidden-state inputs directly to the captured target buffers.
3. Bind its two RoPE inputs at `token_idx`.
4. Bind its packed outputs at `{head=0, token=token_idx, channel=0}` in the two
   persistent packed buffers.
5. Queue and run one model.
6. Advance the logical draft cache length by `valid_tokens`, preserving the
   current handling of uncommitted physical rows.

Remove `fc_nN_output` and the DFlash context model map after no remaining path
uses them. Preserve the EAGLE3 FC model map if it shares the current storage.

### Cache maintenance

Update DFlash draft cache clearing, invalidation, and flush code to operate on
the two packed allocations. Preserve token-range invalidation: folding layers
into the head axis must not cause the runtime to invalidate the full context
when only an N-token range changed.

Audit every use of `cache_key_l*`, `cache_val_l*`, and their scale buffers. A
DFlash draft must use packed slices consistently for proposal execution,
context synchronization, reset, and interruption/reuse. Target checkpoints and
target rollback state remain unchanged.

## Validation

### Host-side graph validation

- Assert the input count derives from `target_layer_ids`.
- Assert the packed output shapes for N8, N16, and N128.
- Verify the K-then-V layer/head ordering and scale ordering.
- Assert that the packed projection has output width
  `2 * num_draft_layers * num_key_value_heads * head_dim`.
- Verify concatenated weight slices against every original K/V tensor and
  verify that W8 scales retain their original quantization granularity.
- Compare the combined graph numerically with the current fusion plus
  per-layer graphs using identical hidden states and RoPE inputs.
- Check every packed slice against its corresponding old per-layer K/V output.
- Exercise both quantized and BF16 cache construction if both remain supported.

### Compiler validation

- Compile a small combined graph first and confirm the packed HWC16 OFM store.
- Inspect scheduling for spills of the fused hidden tensor.
- Confirm that the compiler emits one packed K/V projection rather than
  expanding it into ten redundant input reads.
- Compare ELF/filter size and graph latency with the six existing graphs.
- Compile Qwen3-4B DFlash N8 and N16 with filter sharing enabled.
- Confirm that N128 artifacts remain reusable between N8 and N16 packages.

Do not add a compiler workaround unless the minimal graph demonstrates a real
unsupported store pattern.

### Modalix runtime validation

- Prefill with full and partial N128 prompt chunks.
- Verify full acceptance, zero draft acceptance, and partial acceptance.
- Run multiple verification rounds and confirm rejected physical rows are
  overwritten before they become logical cache entries.
- Compare generated tokens and accepted-token counts with the existing DFlash
  implementation.
- Cover English, German, Japanese, multi-turn recall, the math prompt, and a
  native tool call.
- Test cancellation followed by another request without restarting the
  runtime.
- Check cache boundaries near the 2K context limit.
- Measure context synchronization latency, total TPS, DRAM traffic, and CMA
  memory before claiming an improvement.

## Expected result

Each prefill chunk or verification round changes from six queued models to one.
The fused target representation no longer crosses an ELF boundary, and the
runtime exposes two packed outputs instead of twenty per-layer outputs. Total
K/V data written is unchanged. The actual speedup depends on whether the
compiler keeps the shared fused representation on-chip and tiles the 25 MiB
Qwen or 40 MiB Llama packed K/V projection efficiently.

## Non-goals

- Variable-width context updates based on accepted-token count.
- Changes to target verification, CPU argmax, or token acceptance.
- Changes to Qwen3.5 GDN state resolution.
- A general packed-cache migration for ordinary generation or EAGLE3.
