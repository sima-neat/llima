# Add a VLM Encoder or Projector

Separate vision-transformer, projector, preprocessing, and prompt-integration
differences before editing.

Use the matching pinned Transformers implementation as the architecture and
numerical reference, or the pinned model repository implementation when
Transformers lacks support. ModelGraph uses NHWC `[B, height, width, C]` for
image tensors, while Transformers typically uses NCHW `[B, C, height, width]`.
Transpose image inputs at the reference boundary. Already patchified inputs
follow the component's token contract. Encoder hidden states use
`[B, 1, T, C]` instead of `[B, T, C]`, and
attention Q/K/V retain `[B, heads, T, D]`, as described in
`llm-architecture.md`.

## Procedure

1. Define image sizes, patch/merge factors, channel layout, normalization, and
   shape constraints.
2. Define encoder input/output and projector output width/token count.
3. Add `VisionArchType` or `VlmArchType` only when existing IDs cannot express
   the model.
4. Extend `StandardVisionLayerModel` when structurally compatible; specialize
   only for different graph semantics.
5. Align host `sima_lmm/preproc/`, device `image_processor.cpp`, and
   `vlm_helper.cpp` image-token handling.
6. Verify language input sequence length, layout, dtype, and insertion points.

## Example: Qwen3-VL from Qwen2.5-VL

- Compare vision config, position encoding, patch merging, projector,
  deep-stack features, and upstream processor output.
- Reuse common Qwen layers; add only Qwen3-specific paths/contracts.
- Extend prompt/image-token handling without changing Qwen2.5 behavior.
- Compare deterministic processor inputs, add config and vision native graph cases,
  exercise direct graph compilation where supported, and run a complete
  image-grounded model on Modalix.
