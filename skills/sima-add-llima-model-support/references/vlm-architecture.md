# Add a VLM Encoder or Projector

Separate vision-transformer, projector, preprocessing, and prompt-integration
differences before editing.

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
- Compare deterministic processor inputs, add config and vision ONNX cases,
  exercise direct graph compilation where supported, and run a complete
  image-grounded model on Modalix.
