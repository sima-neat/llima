# Add a VLM Vision Encoder or Projector

First determine whether the difference is in the vision transformer,
multimodal projector, image preprocessing, language prompt integration, or a
combination.

## Procedure

1. Define accepted image sizes, patch size, merge/downsample factors, channel
   layout, normalization, and dynamic/static shape constraints.
2. Define the vision encoder input and output tensor contract.
3. Define how the projector transforms vision features into language embedding
   width and how many image tokens it produces.
4. Register `VisionArchType` and `VlmArchType` only when existing identifiers
   cannot express the model.
5. Extend `StandardVisionLayerModel` when the graph is structurally compatible;
   introduce a specialized vision model only for genuinely different graph
   semantics.
6. Keep host preprocessing in `sima_lmm/preproc/` aligned with device
   `image_processor.cpp` and prompt/image-token handling in `vlm_helper.cpp`.
7. Verify the language model receives the exact sequence length, layout, dtype,
   and insertion positions expected by the compiled graph.

## Example: Qwen3-VL Relative to Qwen2.5-VL

Request:

> Add Qwen3-VL using Qwen2.5-VL as the closest supported VLM.

Investigation:

- Compare vision configuration, positional encoding, patch merging, projector,
  and any deep-stack feature outputs.
- Confirm which Qwen vision layers can be shared and which model-type
  conditions represent real graph differences.
- Compare upstream processor output for the same image and resolution.

Implementation:

- Add the new VLM and vision identifiers.
- Reuse the Qwen vision implementation for common layers.
- Add only the Qwen3-specific feature paths and output contracts.
- Extend image-token prompt handling without changing Qwen2.5 behavior.

Validation:

- Compare normalized pixel/patch inputs with the upstream processor.
- Add configuration and vision ONNX regression cases at a deterministic image
  size.
- Exercise direct graph compilation if supported.
- Compile the complete VLM and run an image-grounded prompt on Modalix.

