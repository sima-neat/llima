# Model Compilation

## Overview

**Model Compiler** provides the LLiMa command-line tool `llima-compile` to
compile models from Hugging Face safetensors, GGUF files, or pre-quantized
compressed tensor models (GPTQ/AWQ):

``` console
llima-compile [options] <model_path>
```

When you run this command, the tool handles the entire compilation pipeline including calibration, quantization, and code generation. The pipeline consists of several stages that differ slightly depending on the input format:

**For HuggingFace Models:**

1.  **DEVKIT** - Generate runtime orchestration scripts
2.  **ONNX** - Convert model to ONNX intermediate representation
3.  **QUANTIZE** - Quantize model weights and calibrate
4.  **COMPILE** - Compile to Modalix machine code

**For GGUF Models:**

1.  **DEVKIT** - Generate runtime orchestration scripts
2.  **MODEL_COMPILER_DIRECT** - Convert GGUF directly to Model Compiler format (quantization already applied)
3.  **COMPILE** - Compile to Modalix machine code

**For Pre-quantized Compressed Tensor Models (GPTQ/AWQ):**

1.  **DEVKIT** - Generate runtime orchestration scripts
2.  **SOURCE_TO_QUANT** - Convert compressed tensor model directly to Model Compiler format
3.  **COMPILE** - Compile to Modalix machine code

:::note
Compressed tensor models are safetensor models pre-quantized with [llm-compressor](https://github.com/vllm-project/llm-compressor) (e.g. GPTQ or AWQ). Supported for LLMs only. A GPU is recommended for the quantization step.
:::

You can run individual stages using `--onnx`, `--quantize`, `--model_sdk`, `--compile`, or `--devkit` flags if needed.

The compilation process generates the following directory structure in your output directory:

``` text
output_directory/
├── onnx_files/                # ONNX intermediate files (HF models only)
│   └── ...
└── sima_files/                # Compiled model files
    ├── devkit/                # Python runtime orchestration files
    │   ├── tokenizer.json
    │   ├── vlm_config.json
    │   └── ...
    ├── mpk/                   # MPK archives with compiled binaries
    │   ├── layer_0.tar.gz
    │   └── ...
    ├── npy_files/             # LoRA adapter weights (only when compiled with LoRA)
    │   ├── <adapter_name>/
    │   │   └── *.npy
    │   └── ...
    └── ...
```

## Command-Line Arguments

The `llima-compile` tool accepts various arguments to customize the compilation process. The following tables describe the available options:

| Argument | Description |
|----|----|
| `model_path` | Input model path (HuggingFace directory, GGUF file, or pre-quantized compressed tensor directory). |
| `-o, --output` | Output directory for compiled files. Defaults to the model name. |
| `-c, --configuration_file` | Python script to configure precision per layer (e.g., for mixed-precision). |
| `--max_num_tokens` | Max context length. Default: 1024. |
| `--resume` | Resume interrupted builds by skipping existing files. |
| `-j, --jobs` | Number of parallel compilation jobs. Default: Number of physical CPU cores. |
| `--log_level` | Logging level (DEBUG, INFO, WARNING, ERROR). Default: WARNING. |
| `--input_height` | Input image height in pixels. Required for Siglip2 and Qwen-VL based models. |
| `--input_width` | Input image width in pixels. Required for Siglip2 and Qwen-VL based models. |

| Advanced Argument | Description |
|----|----|
| `--language_group_size` | Batch size for parallel token processing during prefill. Larger values (e.g., 256) can improve TTFT for large input prompts, but can decrease TTFT for smaller input prompts. Default: 128. |
| `--future_token_mask_size` | Mask size for reusing compiled models across token positions. Larger values reduce number of compiled binary files, but may reduce TPS. Default: 128. |
| `--enable_filter_sharing` | Reduces DRAM usage by sharing weights between group and single models. Useful for 16GB Modalix boards. Note: only effective when both group and single models use the same precision. **Required when compiling with LoRA.** |
| `--quantize_embeddings` | Reduces memory consumption through embedding quantization. May result in a loss of accuracy. |
| `--return_logits` | Return logits at the last layer output (needed for model evaluator). |
| `--lora_name` | Name for the LoRA adapter being compiled alongside the base model. |
| `--lora_path` | Path to the LoRA adapter directory to compile with the base model. |

## Configuration File

The configuration file allows customizing compilation on a per-layer basis, enabling mixed-precision compilation and selective layer compilation.

LLM inference consists of two distinct phases, and the compiler generates optimized models for each:

- **Prefill (Group models)**: Processes the input prompt in batches using `language_group_size` (e.g., 128 tokens at once). This phase determines TTFT (Time To First Token) and is optimized for throughput.
- **Decode (Single-token models)**: Generates output tokens one at a time autoregressively. This phase determines TPS (Tokens Per Second) and is optimized for low-latency generation.

Because these phases have different performance characteristics, you can apply different quantization strategies to each using the `is_group` flag in the configuration function.

**Input Parameters**

The `get_layer_configuration` function is called for each layer and receives:

- `model_properties`: Dictionary with `{"num_hidden_layers": int}`

- `layer`: Dictionary with:  
  - `"part"`: Layer type - `"PRE"`, `"CACHE"`, `"POST"`, or `"VISION"`
  - `"is_group"`: `True` for batch processing layers, `False` for single-token layers
  - `"index"`: Layer index (0 to num_hidden_layers-1)

**Return Values**

The function returns a dictionary with:

- `"precision"`: Quantization level (required)  
  - `"BF16"`: Full precision - best quality, largest size, slowest
  - `"A_BF16_W_INT8"`: Medium quantization - good quality, moderate size
  - `"A_BF16_W_INT4"`: High quantization - acceptable quality, smallest size, fastest

- `"compile"`: Set to `False` to skip compiling this layer (optional, default: `True`)

- `"lora"`: LoRA mode for this layer (optional, default: `"LORA_DISABLED"`)  
  - `"LORA_DISABLED"`: No LoRA support for this layer. This is the default when no configuration file is provided, resulting in a standard model with no adapter overhead.
  - `"LORA_BRANCH"`: Compiles parallel LoRA branches with zero weights alongside the base model. Adapter weights are loaded from `.npy` files at runtime, enabling dynamic switching between adapters without restarting the model. Use this mode when you need to swap adapters on the fly.
  - `"LORA_MERGED"`: LoRA weights are merged into the base model weights at runtime. The adapter becomes permanently active for the session with no ability to switch or remove it. Use this mode when you always want the adapter applied and do not need dynamic switching.

:::note
**Best Practice:** Use INT8 (`A_BF16_W_INT8`) for group layers to maintain quality during prefill, INT4 (`A_BF16_W_INT4`) for single-token layers for fast generation, and BF16 for vision encoders to preserve image understanding quality. For most models, this configuration provides the optimal balance between model accuracy, throughput, and memory usage.
:::

## Examples

**Example 1: Compiling a Simple LLM**

Compile a Llama model, downloaded from Hugging Face, with default settings:

``` console
sima-user@docker-image-id:/home/docker$ hf download meta-llama/Llama-3.2-3B --local-dir Llama-3.2-3B-Instruct
sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct -o Llama-3.2-3B-Instruct_out
```

This will:

- Use default BF16 precision for all layers
- Set context length to 1024 tokens
- Output to `Llama-3.2-3B-Instruct_out` directory

**Example 2: Compiling with Custom Context Length**

``` console
sima-user@docker-image-id:/home/docker$ hf download meta-llama/Llama-3.2-3B --local-dir Llama-3.2-3B-Instruct
sima-user@docker-image-id:/home/docker$ llima-compile --max_num_tokens 4096 Llama-3.2-3B-Instruct -o Llama-3.2-3B-Instruct_out
```

This will:

- Use default BF16 precision for all layers
- Set context length to 4096 tokens
- Output to `Llama-3.2-3B-Instruct_out` directory

**Example 3: Compiling Gemma 3 VLM with Mixed Precision**

For complex models like Gemma 3 VLM, you may need to specify different precisions for different layers (e.g., keeping the vision encoder in BF16).

1.  **Download the model**:

    ``` console
    sima-user@docker-image-id:/home/docker$ hf download simaai/gemma3-siglip448 --local-dir gemma-3-model
    ```

2.  **Create a configuration file** (e.g., `config.py`):

    ``` python
    def get_layer_configuration(model_properties, layer):
        # Keep vision encoder in full precision
        if layer["part"] == "VISION":
            precision = "BF16"
        # Use INT8 for batch processing layers (better quality)
        elif layer["is_group"]:
            precision = "A_BF16_W_INT8"
        # Use INT4 for single-token layers (smaller size)
        else:
            precision = "A_BF16_W_INT4"
        return {"precision": precision}
    ```

3.  **Run the compiler**:

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile -c config.py --max_num_tokens 2048 gemma-3-model -o gemma-3-model_out
    ```

**Example 4: Advanced Configuration**

Mixed precision with layer-specific control:

``` python
def get_layer_configuration(model_properties, layer):
    # Skip compiling certain cache layers
    if layer["part"] == "CACHE" and layer["index"] > 20:
        return {"compile": False}

    # Higher precision for early layers
    if layer["index"] < 4:
        return {"precision": "BF16"}

    # Standard quantization for middle layers
    return {"precision": "A_BF16_W_INT8"}
```

**Example 5: Compiling an LLM with LoRA**

LoRA (Low-Rank Adaptation) allows a base model to be fine-tuned and the adapter to be dynamically applied or removed at runtime without recompiling the base model. The base model is compiled with parallel LoRA branches (initialized to zero), and the adapter weights are compiled separately into `.npy` files that are loaded on demand.

:::note
`--enable_filter_sharing` is required when compiling with LoRA. LoRA branches are always compiled in INT8 even if INT4 is specified, for better accuracy.
:::

1.  **Download the base model and LoRA adapter**:

    ``` console
    sima-user@docker-image-id:/home/docker$ hf download meta-llama/Llama-3.2-3B-Instruct --local-dir Llama-3.2-3B-Instruct
    sima-user@docker-image-id:/home/docker$ hf download <org>/<lora-adapter> --local-dir my-lora
    ```

2.  **Create a configuration file** (e.g., `lora_config.py`):

    The `lora` key controls LoRA mode per layer. Use `"LORA_BRANCH"` to enable dynamic switching at runtime.

    ``` python
    def get_layer_configuration(model_properties, layer):
        if layer["is_group"]:
            return {"precision": "A_BF16_W_INT8", "compile": True, "lora": "LORA_BRANCH"}
        else:
            return {"precision": "A_BF16_W_INT4", "compile": True, "lora": "LORA_BRANCH"}
    ```

3.  **Compile the base model with the LoRA adapter**:

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct \
        --lora_name my_adapter \
        --lora_path my-lora \
        --enable_filter_sharing \
        -c lora_config.py \
        -o Llama-3.2-3B-lora-out
    ```

    This compiles the base model with one LoRA branch and automatically compiles the adapter weights into `Llama-3.2-3B-lora-out/sima_files/npy_files/my_adapter/`.

    Multiple adapters can be compiled in the same step by repeating `--lora_name` and `--lora_path`:

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct \
        --lora_name my_adapter_A --lora_path my-lora_A \
        --lora_name my_adapter_B --lora_path my-lora_B \
        --enable_filter_sharing \
        -c lora_config.py \
        -o Llama-3.2-3B-lora-out
    ```

4.  **To add more adapters** without recompiling the base model, use `llima-compile-lora` for each additional adapter:

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile-lora Llama-3.2-3B-Instruct ./lora-c \
        -w Llama-3.2-3B-lora-out/sima_files/mpk \
        -o Llama-3.2-3B-lora-out/sima_files/npy_files/adapter_c
    ```

    **llima-compile-lora arguments**

    | Argument | Description |
    |----|----|
    | `base_path` | Path to the original base model directory (HuggingFace format). |
    | `lora_path` | Path to the LoRA adapter directory to compile. |
    | `-w, --weight_map_path` | **Required.** Path to the `mpk/` folder from the base model compilation. Contains the weight maps needed to compile the adapter. |
    | `-o, --output` | Output directory for the compiled adapter `.npy` files. Defaults to the adapter directory name. |
