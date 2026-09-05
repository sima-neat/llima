# Model Compilation

## Overview

**Model Compiler** provides the LLiMa command-line tool `llima-compile` to
compile models from Hugging Face safetensors, GGUF files, or pre-quantized
compressed-tensors models (GPTQ/AutoRound):

``` console
llima-compile [options] <model_path>
```

### Model input formats

LLiMa accepts three model input paths. Choose one based on checkpoint
availability, fidelity requirements, and whether the model is an LLM or VLM.

| Input | Description | When to use |
| --- | --- | --- |
| Original Hugging Face safetensors | An FP/BF16 checkpoint that LLiMa quantizes during compilation. | No exact pre-quantized match exists, or the original weights are required. |
| Pre-quantized Hugging Face safetensors (GPTQ/AutoRound) | A checkpoint whose quantized weights are reused by LLiMa. | Preferred when the collection provides an exact match. |
| GGUF | An existing quantized LLM checkpoint. | A convenient LLM fallback; not supported for VLMs. |

The input format alone does not establish compatibility. The model
architecture, size, tokenizer, and any multimodal components must also be
supported.

### Start with a SiMa.ai pre-quantized model

:::tip Recommended input
Before downloading original Hugging Face or GGUF weights, check the
[SiMa.ai Pre-Quantized Models collection](https://huggingface.co/collections/simaai/pre-quantized-models).
Use an exact match for the requested architecture, parameter size, variant,
and modality when one is available.
:::

Collection checkpoints are model-specific, pre-LLiMa compressed-tensors
artifacts that can be passed directly to `llima-compile`. They avoid the
additional floating-point-to-quantized compiler stage and include the
quantization provenance needed to understand their accuracy and layout. They
are compiler inputs, not compiled Modalix models.

For a custom fine-tune of an existing supported model, the exact matching
collection repository may also provide its model-specific `quantize.py`,
`recipe.yaml`, and `versions.txt`. Read that repository's model card and use
its documented script; do not reuse a recipe from a merely similar model.

``` console
hf download simaai/<model-repository> \
    --revision <immutable-revision> \
    --local-dir <prequantized-model-directory>
llima-compile <prequantized-model-directory> -o <output-directory>
```

### Describe the model. An agent with LLiMa skills compiles it.

SiMa.ai LLiMa supports agentic model compilation out of the box through skills
included with the Neat Development Environment (Neat SDK). These skills give
coding agents the context to assess LLM and VLM compatibility, select an exact
pre-quantized input when available, use the installed LLiMa CLI, and follow the
Modalix deployment and validation workflow.

The recommended agentic path can compile a model, deploy it to a reachable
Modalix DevKit, inspect results and diagnostics, and refine the compilation.
Traditional CLI compilation remains a parallel path for direct control through
the same tools. Both produce standard, inspectable LLiMa artifacts, so you can
review the selected model, commands, options, and output or move between the
two workflows as requirements evolve. See
[Set up the Neat SDK](https://developer.sima.ai/software/getting-started/dev-environment/)
to enable agentic compilation.

Ask for the complete workflow in natural language, for example:

``` text
Compile <model ID or local path> with LLiMa, deploy it to my Modalix at
<user@host>, and smoke-test it. Prefer an exact SiMa.ai pre-quantized
checkpoint when available.
```

The agent records model and recipe provenance, follows the CLI contract of the
installed release, and reports any unsupported model boundary or unavailable
hardware validation instead of silently substituting another model or format.

### Compilation output

The default complete pipeline generates the following directory structure:

``` text
output_directory/
└── sima_files/                # Compiled model files
    ├── devkit/                # Runtime configuration and model data
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

### Layered vision encoders

Supported VLMs compile every executed vision-transformer layer as a separate
compiler unit and ELF artifact. The first artifact includes patch embedding,
the final artifact includes the multimodal projector, and intermediate
artifacts consume and produce the vision hidden state. Qwen3-VL deepstack
features are emitted by the layer that owns each configured deepstack index.

The generated `devkit/vlm_config.json` records the artifacts in execution
order as `<vision-model-name>_layer0`, `<vision-model-name>_layer1`, and so on.
Do not remove, rename, or reorder these artifacts: the runtime validates the
list before allocating MLA buffers. Existing deployed repositories with one
monolithic vision artifact remain loadable, but newly compiled VLMs always use
the layered layout.

Layered compilation reduces peak host memory per compiler job and allows
multiple vision layers to compile in parallel with `--jobs`. At runtime, each
vision layer is submitted as its own dispatcher job, allowing other MLA work
to be scheduled between the dependent layers.

## Command-Line Arguments

The `llima-compile` tool accepts various arguments to customize the compilation process. The following tables describe the available options:

| Argument | Description |
|----|----|
| `model_path` | Input model path (HuggingFace directory, GGUF file, or pre-quantized compressed tensor directory). |
| `-o, --output` | Output directory for compiled files. Defaults to the model name. |
| `-c, --configuration_file` | Python script to configure precision per layer (e.g., for mixed-precision). |
| `--max_num_tokens` | Max context length. Must be a multiple of 1024. Default: 4096. |
| `--resume` | Resume interrupted builds by skipping existing files. |
| `-j, --jobs` | Number of parallel compilation jobs. Default: Number of physical CPU cores. |
| `--log_level` | Logging level (DEBUG, INFO, WARNING, ERROR). Default: WARNING. |
| `--input_height` | Input image height in pixels. Must be provided with `--input_width`. Required for Qwen 2 VL, Qwen 3 VL, and Gemma 4; optional for overriding a SigLIP2 model's configured size. |
| `--input_width` | Input image width in pixels. Must be provided with `--input_height`. Required for Qwen 2 VL, Qwen 3 VL, and Gemma 4; optional for overriding a SigLIP2 model's configured size. |
| `--system_prompt` | System prompt to store for CLI mode and model warm-up. |
| `--system_prompt_file` | Path to a text file containing the system prompt. |
| `--chat_template` | Chat template string to store in the compiled model. Mutually exclusive with the system-prompt and chat-template file options. |
| `--chat_template_file` | Path to a file containing the chat template. Mutually exclusive with the system-prompt options and `--chat_template`. |

:::note
Most models support context lengths up to 8192 tokens. Use `--max_num_tokens 8192` to enable an 8K context length.
:::

| Advanced Argument | Description |
|----|----|
| `--language_group_size` | Batch size for parallel token processing during prefill. Larger values (e.g., 256) can improve TTFT for large input prompts, but can decrease TTFT for smaller input prompts. Default: 128. |
| `--future_token_mask_size` | Mask size for reusing compiled models across token positions. Larger values reduce number of compiled binary files, but may reduce TPS. Default: 128. |
| `--enable_filter_sharing` | Enable filter sharing between group and single models to reduce DRAM usage at a cost of higher TTFT and lower TPS. This is only effective when both model types use the same precision and is required when compiling with LoRA. |
| `--no-quantize_embeddings` | Disable embedding-table quantization, which is enabled by default for supported LLMs and VLMs. |
| `--no-quantize_kv_cache` | Disable KV-cache quantization, which is enabled by default. |
| `--return_logits` | Return logits at the last layer output (needed for model evaluator). |
| `--draft_model_path` | Path to an EAGLE3 draft model for speculative decoding. |
| `--lora_name` | Name for the LoRA adapter being compiled alongside the base model. |
| `--lora_path` | Path to the LoRA adapter directory to compile with the base model. |
| `--compile_lora`, `--no-compile_lora` | Enable or disable adapter-weight compilation when LoRA paths are supplied. Enabled by default. |

## System Prompts

Use `--system_prompt` or `--system_prompt_file` to store a system prompt in the
compiled model configuration. The arguments are mutually exclusive.

``` console
sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct \
    --system_prompt "You are a concise technical assistant." \
    -o Llama-3.2-3B-Instruct_out
```

In CLI mode, this becomes the default system prompt. It can be replaced with
`set system <prompt>` or removed with `clear system` during the session.

In web mode/GenAIServer, the stored prompt is used during model warm-up and can
be cached for the first request. It is not automatically added to
API requests. The client must include the system prompt in the `messages` array
of the first request and every follow-up request.

## Configuration File

The configuration file customizes compilation for each compiler unit, enabling
mixed-precision and selective compilation.

LLM inference consists of two distinct phases, and the compiler generates optimized models for each:

- **Prefill (Group models)**: Processes the input prompt in batches using `language_group_size` (e.g., 128 tokens at once). This phase determines TTFT (Time To First Token) and is optimized for throughput.
- **Decode (Single-token models)**: Generates output tokens one at a time autoregressively. This phase determines TPS (Tokens Per Second) and is optimized for low-latency generation.

Because these phases have different performance characteristics, you can apply different quantization strategies to each using the `is_group` flag in the configuration function.

**Input Parameters**

The `get_layer_configuration` function is called for each compiler unit and
receives:

- `model_properties`: Dictionary with `{"num_hidden_layers": int}`

- `layer`: Dictionary with:
  - `"part"`: Logical component such as `"PRE"`, `"CACHE"`, `"POST"`,
    `"VISION"`, `"DRAFT_FC"`, or `"PER_LAYER"`
  - `"is_group"`: `True` for a multi-token/group variant and `False` otherwise
  - `"index"`: Index of that compiler unit. For `"PRE"` and `"POST"` this
    normally corresponds to a transformer layer. For `"CACHE"` it identifies
    a cache or token-position variant rather than a transformer layer.

**Return Values**

The function returns a dictionary with:

- `"precision"`: Quantization level (optional, default: `"BF16"`)
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
- Set context length to 4096 tokens
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

Mixed precision with transformer-layer-specific control:

``` python
def get_layer_configuration(model_properties, layer):
    # PRE and POST indices normally identify transformer layers.
    if layer["part"] in {"PRE", "POST"} and layer["index"] < 4:
        return {"precision": "BF16"}

    # Keep every required compiler unit and use INT8 elsewhere.
    return {"precision": "A_BF16_W_INT8"}
```

Do not interpret `"CACHE"` indices as transformer-layer indices. Omitting
cache variants can make the compiled output incomplete and unusable at
runtime.

**Example 5: Compiling an LLM with LoRA**

LoRA (Low-Rank Adaptation) allows a base model to be fine-tuned and the adapter to be dynamically applied or removed at runtime without recompiling the base model. The base model is compiled with parallel LoRA branches (initialized to zero), and the adapter weights are compiled separately into `.npy` files that are loaded on demand.

:::note
Filter sharing is required when compiling with LoRA. Enable it with `--enable_filter_sharing`. LoRA branches are always compiled in INT8 even if INT4 is specified, for better accuracy.
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
        --enable_filter_sharing \
        --lora_name my_adapter \
        --lora_path my-lora \
        -c lora_config.py \
        -o Llama-3.2-3B-lora-out
    ```

    This compiles the base model with one LoRA branch and automatically compiles the adapter weights into `Llama-3.2-3B-lora-out/sima_files/npy_files/my_adapter/`.

    Multiple adapters can be compiled in the same step by repeating `--lora_name` and `--lora_path`:

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct \
        --enable_filter_sharing \
        --lora_name my_adapter_A --lora_path my-lora_A \
        --lora_name my_adapter_B --lora_path my-lora_B \
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
