# swml-auto-lmm

## Description
**swml-auto-lmm** is a tool that generates the files needed for running a Vision Language Model
(VLM) or Large Language Model (LLM) on SiMa.ai's gen2 hardware. It is designed to enable users to
easily run models from HuggingFace (HF) repository or GGUF files.

## Contents
* **/scripts** - Contains example scripts for generating files needed to run several VLMs.
* **/sima_lmm** - Contains core functionality.
    * **/sima_lmm/assets** - Contains sample image and audio file for runtime warm up.
    * **/sima_lmm/config** - Contains configuration data structures defining all aspects of VLMs
      and LLMs.
    * **/sima_lmm/devkit** - Contains code responsible for running the VLMs/LLMs on a devkit.
    * **/sima_lmm/gguf** - Contains code responsible for generating the files from a GGUF file.
    * **/sima_lmm/hf** - Contains code responsible for generating the files from an HF repository.
    * **/sima_lmm/model** - Contains code defining the internal representation of a VLM model and
      its modules.
    * **/sima_lmm/mole** - Contains code responsible for benchmarking the VLMs/LLMs.
    * **/sima_lmm/preproc** - Contains definitions for typical VLM preprocessing modules.
    * **/sima_lmm/tokenizer** - Contains definitions for typical VLM tokenizer modules.
* **/tests** - Contains various unit and functional tests. Not included in the package build.

## Dependencies
* [sima_frontend](https://bitbucket.org/sima-ai/awesome-front-end)
* [sima_utils](https://bitbucket.org/sima-ai/sima-utils/)
* **python>=3.11**

## Installation
To get started with this project, follow these installation steps:

1. **Clone the repository**:
    ```bash
    git clone git@bitbucket.org:sima-ai/swml-auto-lmm.git
    cd swml-auto-lmm
    ```
2. **Setup SiMa's Python repository**: Follow the instructions from
[Confluence page](https://sima-ai.atlassian.net/wiki/spaces/STMS/pages/104890369/How+to+setup+SiMa+s+Python+Repository).

3. **Install for model compilation only**:
    ```bash
    pip install -e .[sdk]

    # or the following to install Model SDK separately.
    pip install -e .[sdk_ext]
    ```

4. **Install for model running (build on the devkit)**:
    ```bash
    # Install rust. Need to restart the shell to have environment variables updated.
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

    # Install packages.
    sudo apt install git libfftw3-dev libyaml-cpp-dev cppzmq-dev simaai-memory-lib-dev
    pip install --upgrade pip  # Upgrade pip (>=25.3) to use the -C argument.

    # Populate the third party libraries. These are only used for the the header files.
    # If this step cannot be done on the devkit, clone the repo and update the submodules on the
    # host and then copy/rsync the directory to the devkit.
    git submodule update --init --recursive

    # Option 1: editable install. Good for active python modification.
    pip install -e . -Cbuild-dir="build/{wheel_tag}"

    # Option 2: build install the wheel file.
    pip install build
    python -m build --wheel -Cbuild-dir="build/{wheel_tag}"
    ```

5. **Install for model running (build wheel in elxr palette docker)**:
    1. Download the docker file (choose the desired revision) from [artifactory](https://artifacts.eng.sima.ai:443/artifactory/soc-images/elxr/modalix/): Dockerfile.modalix and elxr-palette-modalix-\*-am64.img.gz.
    2. Modify the Dockerfile.modalix. Instruction is based on the 2.1.0_daily_develop_B889. Actual
        steps may be different.
    ```
    --- Dockerfile.modalix.orig     2026-02-07 11:23:11.872323266 -0800
    +++ Dockerfile.modalix  2026-02-07 13:36:22.951279807 -0800
    @@ -53,14 +53,16 @@
            libssl-dev \
            libgnutls28-dev \
            openssh-client \
    -       simaai-sdk-tools
    +       simaai-sdk-tools \
    +       python3-venv

     RUN export RUSTUP_HOME=/opt/toolchain/rust && export CARGO_HOME=/opt/toolchain/rust && \
         mkdir -p ${RUSTUP_HOME} && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > /tmp/rustup.sh && \
         chmod 755 /tmp/rustup.sh && /tmp/rustup.sh -y && \
         echo "export RUSTUP_HOME=${RUSTUP_HOME}" >> ${CARGO_HOME}/env && \
         echo "export CARGO_HOME=${CARGO_HOME}" >> ${CARGO_HOME}/env && \
    -    . ${CARGO_HOME}/env && rm /tmp/rustup.sh
    +    . ${CARGO_HOME}/env && rm /tmp/rustup.sh && \
    +    rustup target add aarch64-unknown-linux-gnu

     RUN if uname -a | grep -q '#[0-9]*-Ubuntu'; then \
             apt-get update --allow-releaseinfo-change; \
    @@ -75,3 +77,5 @@
            "${SDK_PKG_LIST}"

     RUN echo "source /opt/bin/simaai-init-build-env modalix" >> /root/.bashrc
    +
    +RUN pip install crossenv build --break-system-packages
    ```
    3. Build the docker: ```docker build -t modalix-sdk --file Dockerfile.modalix .```
    4. Build the wheel in the root dir of repo: ```./scripts/build_scripts/build_py_pkg_modalix_docker.sh```
    ```

6. **(Alternative) Download and install the wheel file from Artifactory**
    1. Download the wheel file (choose the desired version) from [artifactory](https://artifacts.eng.sima.ai:443/artifactory/sima-pypi/swml-auto-lmm/).
    2. The runtime uses the system MLASHM dispatcher and requires a running `mlashmcomplex` service.

## Usage
You can compile and run a model using swml-auto-lmm by the following steps.

### Compile
For this example, the model to be compiled is a collection of files in
directory `models/llama`, following the HuggingFace model format.  Use
llima-compile to compile the model:

```bash
llima-compile models/llama
```

This compiles the model and produces files in directory `llama/`
(the same name as the input model).  The files include data that
controls how `llima run` will execute it on the devkit, in addition
to the compiled instructions and data of the model.

Use `llima-compile --help` to see all command line options.

You can pass a configuration file with `-c` followed by a filename.
A configuration file must have a Python function
`get_layer_configuration(model_properties, layer)` that,
when called with a layer, returns the configuration for that layer.
Its parameters are dicts with fields:

*  `model_properties["num_hidden_layers"]` is the number of hidden
   layers in the model.
*  `layer["part"]` is the part of the model that is in the layer
   being examined.  The part can be "PRE", "POST", "CACHE", or "VISION".
*  `layer["is_group"]` is True if the layer is being compiled for
   processing a group of tokens, False if it is being compiled for
   processing a single token.
*  `layer["index"]` is the index of the layer.  For CACHE, this is
   the token index that it processes.  For other parts, this is the
   layer's index in the structure of the model.

The return value of `get_layer_configuration` must be a dict with
optional fields:

*  `_["compile"]` is whether to compile the layer.  The default
   is True.  False can be used to skip layers that have previously
   been compiled.
*  `_["precision"]` is the precision to quantize the layer with.
   The allowed values are `"BF16"`, `"A_BF16_W_INT8"`, and
   `"A_BF16_W_INT4"`.  The default value is `"BF16"`.  It is
   ignored if the model is not being quantized.

The following example configuration file uses different precisions
group layers and single layers.

```python
def get_layer_configuration(model_properties, layer):
    if layer["is_group"]:
        return {"precision": "BF16"}
    else:
        return {"precision": "A_BF16_W_INT8"}
```

LoRA models can be compiled using the `--lora_path` along with the `--lora_name` arguments. Multiple LoRAs can be provided.

Notes:
*  `--enable_filter_sharing` and `--no-split_mlp` needs to be used for LoRA.
*  For better performance, Int4 LoRA nodes are compiled in Int8.

### Deploy compiled files on devkit
Run the following command on the host where the model is compiled. The elf files will be first
extracted from the compiled mpk tar.gz files into elf_files folder. And then rsync is performed to
copy the devkit and elf_files directories to the devkit.

```bash
llima-deploy <src_dir> <dst_dir>
```

* ***src_dir*** - Directory to the model files which contains the 2 (or 3) sub-directories:
    * *devkit* - Directory with the config files, tokenizer file and the embedding table.
    * *mpk* - Directory with the compiled \*_mpk.tar.gz files.
    * *npy_files* - Directory with the LoRA weights if the model was compiled with LoRAs.
*  ***dst_dir*** - Directory to the destination on the devkit to run inference.

### Run inference on devkit
```bash
llima run <model_dir> --mode <mode>
```

* ***model_dir*** - Directory to the model files which contains 2 (or 3) sub-directories:
    * *devkit* - Directory with the config files, tokenizer file and the embedding table.
    * *elf_files* - Directory with the elf files.
    * *npy_files* - Directory with the LoRA weights if the model was compiled with LoRAs.
* ***mode*** - Evaluation mode. Possible values are:
    * *cli* - Run the demo in CLI mode. This starts an interactive session in console.
    * *web* - Run the demo in WEB mode. This starts an HTTP server to receive requests from
        [apps-genai-demo](https://bitbucket.org/sima-ai/apps-genai-demo/src/master/).
* Other options: `llima run --help`
    * ***stt_model_path*** - Directory to the model files for openai/whisper-small.
    * ***parallel_load*** - Load the models in parallel. This may speed up the initialization time.
    * ***report_tps_to_web*** - Web mode specific to report TPS to apps-genai-demo instead of the
        time to generate the next token.
    * ***system_prompt*** - Override the system prompt in string.
    * ***system_prompt_file*** - Override the system prompt from the file.
    * ***log_level*** - Set python logging level.
* Model resolution order for `llima run`:
    * Use the provided local path if it exists.
    * Otherwise check `/media/nvme/llima/models/<model>`.
    * Otherwise check `$LLIMA_MODELS_PATH/<model>` if set.
    * If not found, `llima run` prints a message to use `llima pull` and exits.
* Commands to use in CLI mode:
    * add image <fn> - add an image.
    * clear image - clear all the images.
    * set system <prompt> - set system prompt.
    * clear system - clear system prompt, chat history and images.
    * clear history - clear chat history and images.
    * print history - print chat history.
    * set audio <fn> - set the audio file to be transcribed as query.
    * set language <lang> - set the language string to be used for transcription.
    * set lora <lora_name> - load the lora weights.
    * unset lora - clear the lora weights.
    * list command - print this page.
    * help - print this page.
* Debug model outputs:
```bash
SIMA_LLIMA_RUN_PRINT_INOUTS=1 llima run ...
```
* Profile performance (obtain the latency of running each model):
```bash
SIMA_LLIMA_RUN_PROFILE=1 llima run ...
```
* Disable running models in a queue:
```bash
SIMA_LLIMA_RUN_DISABLE_QUEUE=1 llima run ...
```
* Disable vision support (useful for language only compilation and debug):
```bash
SIMA_LLIMA_RUN_DISABLE_VISION=1 llima run ...
```

### Manage models
```bash
llima search [term]
llima pull <model>
llima list
llima rm <model>
```

* **llima search [term]** - Lists available models from the Sima.ai Hugging Face org. If no term
  is provided, all remote models are listed.
* **llima pull <model>** - Downloads a model by name (without the `simaai/` prefix) from the
  Sima.ai Hugging Face org into `/media/nvme/llima/models` or `LLIMA_MODELS_PATH`.
* **llima list** - Lists top-level local model folders found in `/media/nvme/llima/models` and/or
  `LLIMA_MODELS_PATH` (no recursive scanning).
* **llima rm <model>** - Removes a local model folder by name or path.

`LLIMA_MODELS_PATH` is an optional environment variable you should set if your models are stored
outside `/media/nvme/llima/models`.

### Low-level API to load a VLM model
Although `llima-compile` is sufficient for most purposes,
Python functions can be called to load and manipulate a model's
data.

```python
from sima_lmm.model.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_hf_cache(
  model_name, model_path, onnx_path, sima_path, max_num_tokens
)
```

* ***model_path*** - a path to HF repository or GGUF file.
* ***onnx_path*** - target path for generating the ONNX files.
* ***sima_path*** - target path for generating the sima files.

### Low-level API to generate files
```python
model.gen_files(gen_mode, precision)
```

* ***gen_mode*** - Flag indicating what files will be generated. Possible values:
    * *FileGenMode.ALL* - [default] Generate ONNX files. Use ONNX files to generate quantized
        ModelSDK files. Use quantized ModelSDK files to generate MPK .tar.gz files.
    * *FileGenMode.ONNX* - Generate ONNX files.
    * *FileGenMode.MODEL_SDK_QUANTIZE* - Generate quantized ModelSDK files from ONNX files. Requires
        ONNX files to be generated.
    * *FileGenMode.MODEL_SDK_DIRECT* - Generate quantized ModelSDK files directly based on VLM
        config. This is the only possibility for generating the quantized ModelSDK files for GGUF
        based models, as ONNX framework does not support GGUF types.
    * *FileGenMode.MODEL_SDK_COMPILE* - Generate MPK tar.gz files. Requires the quantized ModelSDK
        files to be generated.
    * *FileGenMode.DEVKIT* - Generate the script to run on the board.
* ***precision*** - The precision to be used for Model SDK quantization mode.

### Run evaluation inference in software
This runs inference in software without using a devkit.
```python
model.eval(eval_mode, chat)
```

* ***eval_mode*** - Evaluation mode. Possible values are:
    * *EvalMode.HF* - Execute VLM using transformers library.
    * *EvalMode.ONNX* - Execute VLM running the inference of generated ONNX files using onnxruntime
        library.
    * *EvalMode.SDK* - Execute VLM running the inference of generated quantized ModelSDK files.
* ***chat*** - Chat data structure containing possible prompt, query and images inference inputs.


### Run benchmark tool (MoLE)
Currently this only works for LLM.
Install benchmark dependencies on the host:
```bash
pip install -e .[sdk]

# or the following to install Model SDK separately.
pip install -e .[sdk_ext]
```
Or, if you are installing from a built wheel:
```bash
pip3 install './sima_lmm-<version>-py3-none-any.whl[sdk]'
pip3 install './sima_lmm-<version>-py3-none-any.whl[sdk_ext]'
```

CLI format:
```bash
llima-benchmark <command> <model_id> [options]
```

Supported commands:
* `accuracy` - Quality benchmark using lm-eval tasks.
* `perf` - Performance benchmark (TTFT/TPS) on Modalix backend.

Accuracy tasks, see lm_eval's [page](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/tasks/README.md) for full list.

HF accuracy on host:
```bash
llima-benchmark accuracy <model_id> -b hf -o <output_dir>
```
Notes:
* Runs on host only (no board server).
* Uses `cuda:0` if available, otherwise CPU.

Modalix accuracy from host:
> **IMPORTANT:** `accuracy` with `-b modalix` requires a board model compiled with
> `return_logits=True`. If `return_logits` is disabled, benchmark accuracy will fail at runtime.

```bash
llima-benchmark accuracy <model_id> -b modalix -o <output_dir> --max_num_tokens <max_num_tokens> \
    --board_ip <board_ip> --board_port <board_port> \
    --board_model <model_on_board> --board_start_server \
    --board_venv_path <venv_on_board>
```
Notes:
* `--max_num_tokens` should be set to match the maximum number of tokens used in model compilation.
* For communicating between SCD and lab, use an unused port between `8000-8009`.
* `--board_model` must already exist on the devkit and contain both `devkit/` and `elf_files/`.
* To use an already-running board server, pass `--no-board_start_server`.
* `--board_venv_path` is optional. If omitted, remote start uses `llima` from the board's default
    `PATH` (typically global/system install).

Performance benchmark on Modalix (run on host):
```bash
llima-benchmark perf <model_id> -o <output_dir> \
    --board_ip <board_ip> --board_port <board_port> \
    --board_model <model_on_board> --board_start_server \
    --board_venv_path <venv_on_board> \
    --max_num_tokens 1024 --max_new_tokens 256
```
Notes:
* `perf` reports TTFT and TPS statistics (mean/median/p5/p95) by token-length buckets.
* Sample data is generated from `cimec/lambada` and is for performance benchmarking only.
* This command is intended to run on the host. Running it directly on the board is possible but a special-case workflow.
* `perf` does not require `return_logits=True`.

Start benchmark server manually on devkit:
```bash
llima benchmark-server <model> --port <port>
```

Model resolution order on devkit (`<model>` argument):
* Use provided local path if it exists.
* Otherwise check `/media/nvme/llima/models/<model>`.
* Otherwise check `$LLIMA_MODELS_PATH/<model>` if set.
* Supports both `<name>` and `simaai/<name>`.

Check help pages:
```bash
llima-benchmark --help
llima-benchmark accuracy --help
llima-benchmark perf --help
```


### Pre-downloaded HuggingFace models
HuggingFace models that are tested are pre-downloaded to /project/mlasw/share/huggingface
