"""Central model-case definitions for LLiMa compiler tests."""

from dataclasses import dataclass
from typing import Literal


SourceFormat = Literal["hf", "gguf"]


@dataclass(frozen=True)
class ConfigurationCase:
    model_folder: str
    reference_config: str
    source_format: SourceFormat
    image_resolution: tuple[int, int] | None = None

    @property
    def id(self) -> str:
        return f"{self.source_format}:{self.model_folder.removeprefix('models--')}"


CONFIGURATION_CASES = (
    # Hugging Face LLMs
    ConfigurationCase(
        "models--meta-llama--Llama-2-7b-chat-hf",
        "llama2_vlm_config.json",
        "hf",
    ),
    ConfigurationCase(
        "models--meta-llama--Llama-3.1-8B-Instruct",
        "llama3.1_vlm_config.json",
        "hf",
    ),
    ConfigurationCase(
        "models--meta-llama--Llama-3.2-1B-Instruct",
        "llama3.2_vlm_config.json",
        "hf",
    ),
    ConfigurationCase(
        "models--google--gemma-2-2b-it",
        "gemma2_vlm_config.json",
        "hf",
    ),
    ConfigurationCase(
        "models--google--gemma-3-1b-it",
        "gemma3_vlm_config.json",
        "hf",
    ),
    ConfigurationCase(
        "models--google--gemma-4-E2B-it",
        "gemma4_e2b_it_vlm_config.json",
        "hf",
        (480, 480),
    ),
    ConfigurationCase(
        "models--mistralai--Mistral-7B-Instruct-v0.3",
        "mistral_vlm_config.json",
        "hf",
    ),
    ConfigurationCase(
        "models--microsoft--Phi-3.5-mini-instruct",
        "phi3.5_vlm_config.json",
        "hf",
    ),
    ConfigurationCase(
        "models--microsoft--Phi-4-mini-instruct",
        "phi4_vlm_config.json",
        "hf",
    ),
    ConfigurationCase(
        "models--Qwen--Qwen2.5-0.5B-Instruct",
        "qwen2.5_vlm_config.json",
        "hf",
    ),
    ConfigurationCase(
        "models--Qwen--Qwen3-0.6B",
        "qwen3_vlm_config.json",
        "hf",
    ),
    ConfigurationCase(
        "models--LiquidAI--LFM2-350M",
        "lfm2_vlm_config.json",
        "hf",
    ),
    # Hugging Face VLMs
    ConfigurationCase(
        "models--stribomon--gemma3-siglip448",
        "gemma3_siglip448_vlm_config.json",
        "hf",
    ),
    ConfigurationCase(
        "models--Qwen--Qwen2.5-VL-3B-Instruct",
        "qwen2.5_vl_vlm_config.json",
        "hf",
        (448, 448),
    ),
    ConfigurationCase(
        "models--Qwen--Qwen3-VL-2B-Instruct",
        "qwen3_vl_vlm_config.json",
        "hf",
        (448, 448),
    ),
    ConfigurationCase(
        "models--LiquidAI--LFM2-VL-450M",
        "lfm2_vl_vlm_config.json",
        "hf",
    ),
    ConfigurationCase(
        "models--LiquidAI--LFM2.5-VL-450M",
        "lfm2.5_vl_450m_vlm_config.json",
        "hf",
    ),
    # GGUF LLMs
    ConfigurationCase(
        "models--unsloth--Llama-3.1-8B-Instruct-GGUF",
        "llama3.1_vlm_config.json",
        "gguf",
    ),
    ConfigurationCase(
        "models--unsloth--Llama-3.2-1B-Instruct-GGUF",
        "llama3.2_vlm_config.json",
        "gguf",
    ),
    ConfigurationCase(
        "models--unsloth--gemma-4-E2B-it-GGUF",
        "gemma4_e2b_it_vlm_config.json",
        "gguf",
    ),
    ConfigurationCase(
        "models--bartowski--Mistral-7B-Instruct-v0.3-GGUF",
        "mistral_vlm_config.json",
        "gguf",
    ),
    ConfigurationCase(
        "models--bartowski--Phi-3.5-mini-instruct-GGUF",
        "phi3.5_vlm_config.json",
        "gguf",
    ),
    ConfigurationCase(
        "models--Qwen--Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5_vlm_config.json",
        "gguf",
    ),
    ConfigurationCase(
        "models--Qwen--Qwen3-0.6B-GGUF",
        "qwen3_vlm_config.json",
        "gguf",
    ),
    ConfigurationCase(
        "models--LiquidAI--LFM2-350M-GGUF",
        "lfm2_vlm_config.json",
        "gguf",
    ),
)


@dataclass(frozen=True)
class GgufFileCase:
    quantization: str
    model_folder: str
    filename: str
    reference_tolerance: tuple[float, float] | None = None
    library_tolerance: tuple[float, float] | None = None

    @property
    def id(self) -> str:
        return self.quantization

    @property
    def relative_path(self) -> str:
        return f"{self.model_folder}/{self.filename}"


GGUF_HF_REFERENCE_MODEL = "models--google--gemma-3-1b-it"

GGUF_FILE_CASES = (
    GgufFileCase(
        "BF16",
        "models--unsloth--gemma-3-1b-it-GGUF",
        "gemma-3-1b-it-BF16.gguf",
    ),
    GgufFileCase(
        "Q8_0",
        "models--unsloth--gemma-3-1b-it-GGUF",
        "gemma-3-1b-it-Q8_0.gguf",
        (0.01, 0.05),
        (0.01, 0.02),
    ),
    GgufFileCase(
        "Q4_0",
        "models--unsloth--gemma-3-1b-it-GGUF",
        "gemma-3-1b-it-Q4_0.gguf",
        (0.05, 0.18),
        (0.02, 0.05),
    ),
    GgufFileCase(
        "Q6_K",
        "models--unsloth--gemma-3-1b-it-GGUF",
        "gemma-3-1b-it-Q6_K.gguf",
        (0.04, 0.06),
        (0.02, 0.03),
    ),
    GgufFileCase(
        "Q5_K",
        "models--unsloth--gemma-3-1b-it-GGUF",
        "gemma-3-1b-it-Q5_K_S.gguf",
        (0.04, 0.06),
        (0.03, 0.04),
    ),
    GgufFileCase(
        "Q4_K",
        "models--unsloth--gemma-3-1b-it-GGUF",
        "gemma-3-1b-it-Q4_K_S.gguf",
        (0.06, 0.10),
        (0.05, 0.05),
    ),
    GgufFileCase(
        "Q3_K",
        "models--unsloth--gemma-3-1b-it-GGUF",
        "gemma-3-1b-it-Q3_K_S.gguf",
        (0.08, 0.14),
        (0.08, 0.05),
    ),
)

GGUF_PARSER_CASES = (
    GgufFileCase(
        "Q8_0",
        "models--unsloth--gemma-3-1b-it-GGUF",
        "gemma-3-1b-it-Q8_0.gguf",
    ),
    GgufFileCase(
        "Q4_0",
        "models--google--gemma-3-1b-it-qat-q4_0-gguf",
        "gemma-3-1b-it-q4_0.gguf",
    ),
)


RegressionMode = Literal["required", "informative", "disabled"]


@dataclass(frozen=True)
class OnnxRegressionCase:
    model_folder: str
    component: str
    image_resolution: tuple[int, int] | None = None
    target_model_folder: str | None = None
    mode: RegressionMode = "required"
    atol: float = 0.0
    rtol: float = 0.0

    @property
    def id(self) -> str:
        model = self.model_folder.removeprefix("models--")
        prefix = "speculative" if self.target_model_folder is not None else "standard"
        return f"{prefix}:{model}:{self.component}"


ONNX_REGRESSION_CASES = (
    OnnxRegressionCase("models--meta-llama--Llama-3.2-1B-Instruct", "pre"),
    OnnxRegressionCase("models--meta-llama--Llama-3.2-1B-Instruct", "cache"),
    OnnxRegressionCase("models--meta-llama--Llama-3.2-1B-Instruct", "post"),
    OnnxRegressionCase("models--google--gemma-3-1b-it", "pre"),
    OnnxRegressionCase("models--google--gemma-3-1b-it", "cache"),
    OnnxRegressionCase("models--google--gemma-3-1b-it", "post"),
    OnnxRegressionCase(
        "models--google--gemma-4-E2B-it", "pre", (240, 240)
    ),
    OnnxRegressionCase(
        "models--google--gemma-4-E2B-it", "cache", (240, 240)
    ),
    OnnxRegressionCase(
        "models--google--gemma-4-E2B-it", "post", (240, 240)
    ),
    OnnxRegressionCase(
        "models--google--gemma-4-E2B-it", "per_layer", (240, 240)
    ),
    OnnxRegressionCase(
        "models--mistralai--Mistral-7B-Instruct-v0.3", "pre"
    ),
    OnnxRegressionCase(
        "models--mistralai--Mistral-7B-Instruct-v0.3", "cache"
    ),
    OnnxRegressionCase(
        "models--mistralai--Mistral-7B-Instruct-v0.3", "post"
    ),
    OnnxRegressionCase("models--microsoft--Phi-3.5-mini-instruct", "pre"),
    OnnxRegressionCase("models--microsoft--Phi-3.5-mini-instruct", "cache"),
    OnnxRegressionCase("models--microsoft--Phi-3.5-mini-instruct", "post"),
    OnnxRegressionCase("models--Qwen--Qwen2.5-0.5B-Instruct", "pre"),
    OnnxRegressionCase("models--Qwen--Qwen2.5-0.5B-Instruct", "cache"),
    OnnxRegressionCase("models--Qwen--Qwen2.5-0.5B-Instruct", "post"),
    OnnxRegressionCase("models--Qwen--Qwen3-0.6B", "pre"),
    OnnxRegressionCase("models--Qwen--Qwen3-0.6B", "cache"),
    OnnxRegressionCase("models--Qwen--Qwen3-0.6B", "post"),
    OnnxRegressionCase("models--LiquidAI--LFM2-350M", "conv"),
    OnnxRegressionCase("models--stribomon--gemma3-siglip448", "vision"),
    OnnxRegressionCase(
        "models--Qwen--Qwen2.5-VL-3B-Instruct", "vision", (224, 224)
    ),
    OnnxRegressionCase(
        "models--Qwen--Qwen3-VL-2B-Instruct", "vision", (224, 224)
    ),
    OnnxRegressionCase("models--LiquidAI--LFM2-VL-450M", "vision"),
    OnnxRegressionCase(
        "models--google--gemma-4-E2B-it", "vision", (240, 240)
    ),
    OnnxRegressionCase(
        "models--lmsys--SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge",
        "pre",
        target_model_folder="models--meta-llama--Llama-3.1-8B-Instruct",
    ),
    OnnxRegressionCase(
        "models--lmsys--SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge",
        "cache",
        target_model_folder="models--meta-llama--Llama-3.1-8B-Instruct",
    ),
    OnnxRegressionCase(
        "models--lmsys--SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge",
        "post",
        target_model_folder="models--meta-llama--Llama-3.1-8B-Instruct",
    ),
    OnnxRegressionCase(
        "models--lmsys--SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge",
        "draft_fc",
        target_model_folder="models--meta-llama--Llama-3.1-8B-Instruct",
    ),
)


@dataclass(frozen=True)
class GraphPathCase:
    component: Literal["pre", "cache", "post"]
    layer_index: int
    input_shapes: tuple[tuple[int, ...], ...]
    tolerance: float = 0.01
    concatenate_outputs: bool = False

    @property
    def id(self) -> str:
        return f"{self.component}:layer-{self.layer_index}"


STANDARD_GRAPH_MODEL = "models--google--gemma-3-1b-it"

STANDARD_GRAPH_CASES = (
    GraphPathCase(
        "pre",
        2,
        ((1, 1, 1, 1152), (1, 1, 1, 128), (1, 1, 1, 128)),
    ),
    GraphPathCase(
        "cache",
        2,
        ((1, 4, 1, 256), (1, 1, 3, 256), (1, 1, 3, 256)),
    ),
    GraphPathCase(
        "post",
        2,
        ((1, 1, 1, 1152), (1, 1, 1, 1024)),
        concatenate_outputs=True,
    ),
    GraphPathCase(
        "post",
        25,
        ((1, 1, 1, 1152), (1, 1, 1, 1024)),
        concatenate_outputs=True,
    ),
)


@dataclass(frozen=True)
class GgufGraphCase:
    component: Literal["pre", "cache", "post"]
    quantization: str
    gguf_relative_path: str
    layer_index: int
    output_tolerances: tuple[float, ...]
    reference_relative_path: str = STANDARD_GRAPH_MODEL

    @property
    def id(self) -> str:
        return f"{self.component}:{self.quantization}:layer-{self.layer_index}"


_GEMMA3_GGUF_FOLDER = "models--unsloth--gemma-3-1b-it-GGUF"

GGUF_GRAPH_CASES = (
    GgufGraphCase(
        "pre",
        "Q8_0",
        f"{_GEMMA3_GGUF_FOLDER}/gemma-3-1b-it-Q8_0.gguf",
        2,
        (0.02, 0.01, 0.02),
    ),
    GgufGraphCase(
        "pre",
        "Q4_0",
        f"{_GEMMA3_GGUF_FOLDER}/gemma-3-1b-it-Q4_0.gguf",
        2,
        (0.09, 0.06, 0.12),
    ),
    GgufGraphCase(
        "pre",
        "Q4_1",
        f"{_GEMMA3_GGUF_FOLDER}/gemma-3-1b-it-Q4_1.gguf",
        2,
        (0.12, 0.06, 0.12),
    ),
    GgufGraphCase(
        "cache",
        "Q8_0",
        f"{_GEMMA3_GGUF_FOLDER}/gemma-3-1b-it-Q8_0.gguf",
        2,
        (0.0,),
    ),
    GgufGraphCase(
        "post",
        "Q8_0",
        f"{_GEMMA3_GGUF_FOLDER}/gemma-3-1b-it-Q8_0.gguf",
        2,
        (0.02,),
        f"{_GEMMA3_GGUF_FOLDER}/gemma-3-1b-it-BF16.gguf",
    ),
    GgufGraphCase(
        "post",
        "Q4_0",
        f"{_GEMMA3_GGUF_FOLDER}/gemma-3-1b-it-Q4_0.gguf",
        2,
        (0.06,),
        f"{_GEMMA3_GGUF_FOLDER}/gemma-3-1b-it-BF16.gguf",
    ),
    GgufGraphCase(
        "post",
        "Q4_1",
        f"{_GEMMA3_GGUF_FOLDER}/gemma-3-1b-it-Q4_1.gguf",
        2,
        (0.04,),
        f"{_GEMMA3_GGUF_FOLDER}/gemma-3-1b-it-BF16.gguf",
    ),
    GgufGraphCase(
        "post",
        "Q8_0",
        f"{_GEMMA3_GGUF_FOLDER}/gemma-3-1b-it-Q8_0.gguf",
        25,
        (0.09,),
        f"{_GEMMA3_GGUF_FOLDER}/gemma-3-1b-it-BF16.gguf",
    ),
    GgufGraphCase(
        "post",
        "Q4_0",
        f"{_GEMMA3_GGUF_FOLDER}/gemma-3-1b-it-Q4_0.gguf",
        25,
        (0.18,),
        f"{_GEMMA3_GGUF_FOLDER}/gemma-3-1b-it-BF16.gguf",
    ),
    GgufGraphCase(
        "post",
        "Q4_1",
        f"{_GEMMA3_GGUF_FOLDER}/gemma-3-1b-it-Q4_1.gguf",
        25,
        (0.15,),
        f"{_GEMMA3_GGUF_FOLDER}/gemma-3-1b-it-BF16.gguf",
    ),
)


@dataclass(frozen=True)
class SpeculativeGraphCase:
    component: Literal["pre", "cache", "post", "draft_fc"]

    @property
    def id(self) -> str:
        return self.component


SPECULATIVE_TARGET_MODEL = "models--meta-llama--Llama-3.1-8B-Instruct"
SPECULATIVE_DRAFT_MODEL = (
    "models--lmsys--SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge"
)
SPECULATIVE_GRAPH_CASES = tuple(
    SpeculativeGraphCase(component)
    for component in ("pre", "cache", "post", "draft_fc")
)


E2E_ELIGIBLE_MODELS = (
    "models--meta-llama--Llama-3.2-1B-Instruct",
    "models--google--gemma-3-1b-it",
    "models--mistralai--Mistral-7B-Instruct-v0.3",
    "models--microsoft--Phi-3.5-mini-instruct",
    "models--Qwen--Qwen2.5-0.5B-Instruct",
    "models--Qwen--Qwen3-0.6B",
    "models--LiquidAI--LFM2-350M",
)
