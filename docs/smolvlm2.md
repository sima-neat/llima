# SmolVLM2 integration

The implementation targets these Hugging Face safetensors checkpoints:

| Checkpoint | Revision | Encoder input | Visual tokens per image |
| --- | --- | --- | --- |
| `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` | `067788b187b95ebe7b2e040b3e4299e342e5b8fd` | 512 × 512 | 64 |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` | 512 × 512 | 64 |
| `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | `482adb537c021c86670beed01cd58990d01e72e4` | 384 × 384 | 81 |

Model compilation, packaged C++ tests, and Modalix inference must be validated
before treating this integration as release-qualified support. GGUF input is
not implemented for SmolVLM2.

## Compiler and runtime behavior

Use the ordinary Hugging Face `llima-compile` workflow. The checkpoint supplies
the encoder tile size; `--input_height` and `--input_width` may only repeat that
size. The language model uses standard 1D Llama RoPE: theta 100,000 for 256M/500M
and 130,000 for 2.2B.

The C++ runtime pads each image to a square with the checkpoint's mean color,
then resizes to the encoder input size, preserving the full image and its aspect
ratio. Each image produces one encoder/connector invocation and one global-image
token block. High-resolution crop splitting is deferred. Resizing uses the
existing OpenCV image processor with Lanczos-4, which differs from the checkpoint
processor's PIL Lanczos pixels.

The encoder receives full square tiles, so all patch positions are valid.
Arbitrary masked rectangular encoder inputs and changed encoder resolutions
are not accepted by this implementation. The compiler-side Python processor
provides reference comparisons and asset preparation; inference and image
processing run in C++ on Modalix.

## Sampled video frames

Pass sampled frames as images, interleaved with timestamp text, using the
existing C++ structured-message API. The checkpoint chat template is rendered
by Minja. After loading a compiled model:

```cpp
auto chat = model.create_chat();
chat.set_messages(nlohmann::ordered_json::array({
    {{"role", "user"}, {"content", {
        {{"type", "text"}, {"text", "Frames from a five-second video.\nFrame from 00:00:"}},
        {{"type", "image"}, {"image", "frame0.jpg"}},
        {{"type", "text"}, {"text", "\nFrame from 00:02:"}},
        {{"type", "image"}, {"image", "frame2.jpg"}},
        {{"type", "text"}, {"text", "\nDescribe how the scene changes."}}
    }}}
}));
auto response = model.run_model(chat);
```

Each sampled frame follows the same padding and resizing policy as an image.

Timestamp text and image order provide temporal information. Positions continue
across frames without multimodal RoPE table updates. Budget context for each
image's visual tokens, marker tokens, timestamp, question, and answer; the checkpoints
specify an 8,192-token context.

Frames must already be sampled by the caller. Video-file decoding and new video
endpoints are outside this integration.
