import json
import math
import re
from dataclasses import dataclass, asdict, field, InitVar
from typing import Any
import numpy as np
from pathlib import Path

from sima_lmm.config.layer_id import LayerID
from sima_lmm.utils import ceil_div, round_up_to
from sima_utils.logging.sima_logger import sima_log_warning

LONG_CONTEXT_MIN_TOKENS = 2048
LONG_CONTEXT_FUTURE_TOKEN_MASK_SIZE = 1024


class ExtensibleEnum:
    """
    Enum-like base class that exposes class attributes as extensible constants
    with name/value lookup helpers.
    """
    @classmethod
    def _constants(cls) -> dict[str, Any]:
        """Return all non-private, non-callable attributes."""
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }

    @classmethod
    def values(cls) -> list[Any]:
        return list(cls._constants().values())

    @classmethod
    def names(cls) -> list[str]:
        return list(cls._constants().keys())

    @classmethod
    def name_from_value(cls, value: Any) -> str:
        for name, val in cls._constants().items():
            if val == value:
                return name
        raise ValueError(f"{value!r} is not a valid value in {cls.__name__}")

    @classmethod
    def value_from_name(cls, name: str) -> Any:
        try:
            return cls._constants()[name]
        except KeyError:
            raise ValueError(f"{name!r} is not a valid name in {cls.__name__}")


class ModelFormat(str, ExtensibleEnum):
    """
    Model file format.
    """
    FORMAT_HF = "hf"  # HuggingFace safetensors format
    FORMAT_GGUF = "gguf"  # GGUF format


class VlmArchType(str, ExtensibleEnum):
    """VLM/LLM architecture type.
    """
    LLM_GEMMA = "llm-gemma"
    LLM_GEMMA2 = "llm-gemma2"
    LLM_GEMMA3 = "llm-gemma3"
    LLM_GPT_OSS = "llm-gpt_oss"
    LLM_LFM2 = "llm-lfm2"
    LLM_LLAMA = "llm-llama"
    LLM_MISTRAL = "llm-mistral"
    LLM_PHI3 = "llm-phi3"
    LLM_QWEN2 = "llm-qwen2"
    LLM_QWEN3 = "llm-qwen3"
    VLM_CUSTOM = "vlm-custom"
    VLM_GEMMA3 = "vlm-gemma3"
    VLM_GEMMA4 = "vlm-gemma4"
    VLM_LFM2_VL = "vlm-lfm2_vl"
    VLM_LLAVA = "vlm-llava"
    VLM_PALIGEMMA = "vlm-paligemma"
    VLM_QWEN2_5_VL = "vlm-qwen2_5_vl"
    VLM_QWEN3_VL = "vlm-qwen3_vl"

class VisionArchType(str, ExtensibleEnum):
    """Vision architecture type.
    """
    CLIP = "clip"
    SIGLIP = "siglip"
    SIGLIP2 = "siglip2"
    QWEN2_VISION_ENCODER = "qwen2_5_vl"
    QWEN3_VISION_ENCODER = "qwen3_vl"
    GEMMA4_VISION_ENCODER = "gemma4_vision"


class LlmArchType(str, ExtensibleEnum):
    """LLM architecture type.
    """
    LLAMA = "llama"
    LFM = "lfm"
    GEMMA = "gemma"
    GPT_OSS = "gpt_oss"
    PHI = "phi"
    QWEN = "qwen"
    MISTRAL = "mistral"
    OLMOE = "olmoe"


class LlmDataType(str, ExtensibleEnum):
    """ LLM data type.
    """
    F32 = "float32"
    F16 = "float16"
    BF16 = "bfloat16"
    Q4_0 = "mostly-Q4_0"
    Q4_1 = "mostly-Q4_1"
    Q8_0 = "mostly-Q8_0"
    Q5_0 = "mostly-Q5_0"
    Q5_1 = "mostly-Q5_1"
    Q2_K = "mostly-Q2_K"
    Q3_K_S = "mostly-Q3_K_S"
    Q3_K_M = "mostly-Q3_K_M"
    Q3_K_L = "mostly-Q3_K_L"
    Q4_K_S = "mostly-Q4_K_S"
    Q4_K_M = "mostly-Q4_K_M"
    Q5_K_S = "mostly-Q5_K_S"
    Q5_K_M = "mostly-Q5_K_M"
    Q6_K = "mostly-Q6_K"
    Q2_K_S = "mostly_Q2_K_S"


class GgufFileType(int, ExtensibleEnum):
    """ The type of the majority of the tensors in GGUF file.
    """
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q8_0 = 7
    Q5_0 = 8
    Q5_1 = 9
    Q2_K = 10
    Q3_K_S = 11
    Q3_K_M = 12
    Q3_K_L = 13
    Q4_K_S = 14
    Q4_K_M = 15
    Q5_K_S = 16
    Q5_K_M = 17
    Q6_K = 18
    Q2_K_S = 21
    BF16 = 32


MLA_CONSTRAINTS: dict = {
    "max_position_embeddings": 2048,  # k-v cache size
}

# Number of tokens processed in parallel per speculative-decoding step.
# "draft": the number of candidate tokens the draft model proposes per step (topk).
# "target": the number of tokens the target model verifies in parallel per step.
SPECULATIVE_BUDGET: dict = {
    "draft": 5,
    "target": 16,
}


class BaseConfig(object):
    """Base configuration with a set_config method.
    """
    def set_config(self, cfg):
        if isinstance(cfg, dict):
            for key, value in cfg.items():
                if hasattr(self, key) and not isinstance(value, dict):
                    attr = getattr(type(self), key, None)
                    if isinstance(attr, property) and attr.fset is None:
                        # Skip read-only properties (lfm2 has num_patches in config.json but it is something else)
                        continue
                    setattr(self, key, value)


@dataclass
class VisionModelConfig(BaseConfig):
    """Configuration of Vision Encoder.

    Attributes:
        model_type: The type of the model.
        image_size: The resolution of input images.
        patch_size: The patch size to divide an image.
        hidden_size: The dimension of embedding.
        intermediate_size: The dimension of MLP layer.
        num_attention_heads: The numbder of attention heads.
        num_hidden_layers: The number of transformer blocks.
        hidden_act: The type of activation in MLP.
        layer_norm_eps: The small value to prevent division by zero.
        spatial_merge_size: Vision token merge/pool factor per dimension.
        temporal_patch_size: Qwen-VL: Number of frames in the temporal dimension per patch.
        window_size: Qwen2.5-VL: Window size for windowed attention blocks.
        num_position_embeddings: Qwen3-VL: Number of learnable position embeddings.
        fullatt_block_indexes: Qwen2.5-VL: Layer indices using full attention instead of windowed.
        deepstack_visual_indexes: Qwen3-VL: Layer indices for intermediate deepstack merger outputs.
        rope_theta: Vision RoPE theta.
    """
    model_type: str = ""
    arch: VisionArchType = VisionArchType.CLIP
    image_size: int | list[int] = 0
    patch_size: int = 0
    cls_embed: bool = False
    hidden_size: int = 0
    intermediate_size: int = 0
    num_attention_heads: int = 0
    num_hidden_layers: int = 0
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    spatial_merge_size: int = 0
    temporal_patch_size: int = 0
    window_size: int = 0
    num_position_embeddings: int = 0
    fullatt_block_indexes: list[int] = field(default_factory=list)
    deepstack_visual_indexes: list[int] = field(default_factory=list)
    rope_theta: float = 10000.0

    def set_config(self, arch: VisionArchType, model_cfg: dict, vision_cfg: dict):
        if "num_attention_heads" not in vision_cfg and "num_heads" in vision_cfg:
            vision_cfg["num_attention_heads"] = vision_cfg["num_heads"]
        if "num_hidden_layers" not in vision_cfg and "depth" in vision_cfg:
            vision_cfg["num_hidden_layers"] = vision_cfg["depth"]
        super().set_config(vision_cfg)

        self.arch = arch
        self.image_size = vision_cfg.get("image_size") or model_cfg.get("tile_size", 0)
        self.cls_embed = arch == VisionArchType.CLIP
        if arch == VisionArchType.GEMMA4_VISION_ENCODER:
            self.spatial_merge_size = vision_cfg.get("pooling_kernel_size", 1)
        if "rope_parameters" in vision_cfg:
            rope_params = vision_cfg["rope_parameters"]
            self.rope_theta = rope_params.get("rope_theta", self.rope_theta)

    @property
    def num_patches(self) -> int | list[int]:
        if isinstance(self.image_size, int):
            return self.image_size // self.patch_size
        elif isinstance(self.image_size, list):
            return [l // self.patch_size for l in self.image_size]
        else:
            raise ValueError("image_size must be int or list of int")


    @property
    def seq_len(self) -> int:
        if isinstance(self.num_patches, int):
            return self.num_patches ** 2 + self.cls_embed
        elif isinstance(self.num_patches, list):
            return int(np.prod(self.num_patches)) + self.cls_embed
        else:
            raise ValueError("image_size must be int or list of int")


@dataclass
class MMConnectionConfig(BaseConfig):
    """Configuration of Multi-Modal Connection.

    The MM connection consists of 1 or 2 linear layers.

    Attributes:
        num_layers: The number of linear layers in the connection.
        hidden_act: The type of activation if num_layers is 2.
        mm_tokens_per_image: The number of tokens projected for each image.
            If mm_tokens_per_image is less than num_patches, AvgPool is inserted.
        proj_dim: The number of projected tokens for an image.
        downsample_factor: Reduces vision tokens by factor² via PixelUnshuffle.
        projector_use_layernorm: Whether to apply LayerNorm, which is absent in LFM2.5-VL projector.
    """
    num_layers: int = 2
    hidden_act: str = "gelu"
    mm_tokens_per_image: int = 0
    proj_dim: int = 0
    downsample_factor: int = 1
    projector_use_layernorm: bool = True

    def set_config(self, vm_arch: VisionArchType, lm_hidden_size: int, model_cfg: dict):
        super().set_config(model_cfg)
        self.hidden_act = None if vm_arch == VisionArchType.SIGLIP else "gelu"
        self.num_layers = 2 if self.hidden_act == "gelu" else 1
        self.proj_dim = lm_hidden_size
        if "mm_tokens_per_image" in model_cfg:
            self.mm_tokens_per_image = model_cfg["mm_tokens_per_image"]
        elif "num_image_tokens" in model_cfg.get("vision_config", {}):
            self.mm_tokens_per_image = model_cfg["vision_config"]["num_image_tokens"]
        elif "num_image_tokens" in model_cfg.get("text_config", {}):
            self.mm_tokens_per_image = model_cfg["text_config"]["num_image_tokens"]


@dataclass
class TokenEmbedConfig(BaseConfig):
    """Configuration of tokenizer and embedding.

    Attributes:
        vocab_size: The size of vocabulary of the tokenizer.
    """
    vocab_size: int = 0

    # Keep as InitVar for backward compatibility.
    tokenizer_type: InitVar[str] = ""
    tokenizer_path: InitVar[str] = ""
    special_tokens: InitVar[dict] = dict()


@dataclass
class RopeScalingConfig(BaseConfig):
    """Configuration of RoPE Scaling.

    Attributes:
        factor: The scaling factor.
        low_freq_factor: The low frequency factor (llama3).
        high_freq_factor: The high frequency factor (llama3).
        original_max_position_embeddings: The original context length used in model training with
            the given RoPS settings.
        attention_factor: The post-processing scale applied to LongRoPE cos/sin tables.
        long_factor: List of scaling factors for long context (longrope).
        short_factor: List of scaling factors for short context (longrope).
        rope_type: The type of RoPE scaling method. Supported types are "linear" or "default",
            "llama3", "longrope", and "mrope".
        mrope_section: The section configuration for mRoPE (multimodal RoPE).
        mrope_interleaved: Whether to mix mRoPE frequencies across dimensions (for Qwen3-VL).
    """
    factor: float = 1.0
    low_freq_factor: float = 0
    high_freq_factor: float = 0
    original_max_position_embeddings: int = 0
    attention_factor: float | None = None
    long_factor: list[float] | None = None
    short_factor: list[float] | None = None
    rope_type: str = "default"
    mrope_section: list[int] | None = None
    mrope_interleaved: bool = False

    def set_config(self, cfg: dict):
        super().set_config(cfg)
        # Some models have type instead of rope_type
        if "type" in cfg and "rope_type" not in cfg:
            self.rope_type = cfg["type"]


@dataclass
class RoPEConfig(BaseConfig):
    """Configuration of Rotary Position Embedding.

    Attributes:
        rope_theta: The theta for RoPE.
        rope_local_base_freq: The local base frequency.
        rope_dimension_count: The number of head dimensions rotated by RoPE.
        sliding_rope_dimension_count: Optional sliding-attention RoPE width override.
        rope_scaling: The settings for RoPE scaling.
    """
    rope_theta: float = 10000
    rope_local_base_freq: float = 10000
    rope_dimension_count: int = 0
    sliding_rope_dimension_count: int | None = None
    rope_scaling: RopeScalingConfig = field(default_factory=RopeScalingConfig)

    def set_config(self, text_cfg: dict):
        sliding_head_dim = text_cfg.get("head_dim") or (
            text_cfg.get("hidden_size", 0) // text_cfg.get("num_attention_heads", 1)
        )
        head_dim = text_cfg.get("global_head_dim") or sliding_head_dim
        if "rope_parameters" in text_cfg:
            # HF transformers 5.x format, partial rotary factor is hf only
            rope_params = text_cfg["rope_parameters"]
            full_rope = rope_params.get("full_attention", rope_params)
            sliding_rope = rope_params.get("sliding_attention", full_rope)
            self.rope_theta = full_rope.get("rope_theta", 10000)
            self.rope_local_base_freq = sliding_rope.get("rope_theta", self.rope_theta)
            self.rope_scaling = RopeScalingConfig()
            self.rope_scaling.set_config(full_rope)
            if not self.rope_dimension_count and "partial_rotary_factor" in full_rope:
                self.rope_dimension_count = int(head_dim * full_rope["partial_rotary_factor"])
            # Gemma4 uses a different rotary width for sliding-attention heads.
            if text_cfg.get("global_head_dim") is not None:
                if "partial_rotary_factor" in sliding_rope:
                    self.sliding_rope_dimension_count = int(sliding_head_dim * sliding_rope["partial_rotary_factor"])
                else:
                    self.sliding_rope_dimension_count = sliding_head_dim
        else:
            # GGUF format: flat keys (rope_theta, rope_type, factor, etc.)
            # rope_dimension_count is gguf only
            # HF Legacy Format
            self.rope_theta = text_cfg.get("rope_theta", 10000)
            self.rope_local_base_freq = text_cfg.get("rope_local_base_freq", self.rope_theta)
            self.rope_scaling = RopeScalingConfig()
            self.rope_scaling.set_config(text_cfg)
            if "rope_scaling" in text_cfg:
                if text_cfg["rope_scaling"] is None:
                    # Some configs (e.g. EAGLE3 drafts) explicity set rope_scaling
                    # to null. Defaults to RopeScalingConfig;
                    # for drafts the actual values are inherited from the target
                    # model in VisionLanguageModel.from_hf_cache()
                    self.rope_scaling = RopeScalingConfig()
                else:
                    self.rope_scaling.set_config(text_cfg["rope_scaling"])
            self.rope_dimension_count = text_cfg.get("rope_dimension_count", 0)
            if not self.rope_dimension_count and "partial_rotary_factor" in text_cfg:
                self.rope_dimension_count = int(head_dim * text_cfg["partial_rotary_factor"])
            self.sliding_rope_dimension_count = text_cfg.get("sliding_rope_dimension_count")

        # HF Phi LongRoPE omits attention_factor; GGUF may provide it explicitly.
        if self.rope_scaling.rope_type == "longrope" and self.rope_scaling.attention_factor is None:
            if not self.rope_scaling.original_max_position_embeddings:
                self.rope_scaling.original_max_position_embeddings = text_cfg["original_max_position_embeddings"]
            factor = text_cfg["max_position_embeddings"] / self.rope_scaling.original_max_position_embeddings
            attention_factor = (
                1.0
                if factor <= 1.0
                else math.sqrt(
                    1 + math.log(factor) / math.log(self.rope_scaling.original_max_position_embeddings)
                )
            )
            # Cast to match gguf fp32 attention factor
            self.rope_scaling.attention_factor = float(np.float32(attention_factor))

        if not self.rope_dimension_count:
            self.rope_dimension_count = head_dim

    def get_rope_dimension_count(self, layer_type: str) -> int:
        if layer_type == "sliding_attention" and self.sliding_rope_dimension_count is not None:
            return self.sliding_rope_dimension_count
        return self.rope_dimension_count


@dataclass
class AttentionBlockConfig(BaseConfig):
    """Configuration of attention block.

    Attributes:
        num_attention_heads: The number of attention heads.
        num_key_value_heads: The number of key/value heads.
        head_dim: The full/default dimension of query, key, and value heads.
        sliding_head_dim: Optional sliding-attention head dimension override.
        swa_enable: The flag to turn on sliding window attention.
        sliding_window: The size of sliding window for SWA.
        attention_bias: Reserved for future.
        attention_dropout: Reserved for future.
        query_pre_attn_scalar: Reserved for future.
    """
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    sliding_head_dim: int | None = None
    swa_enable: bool = False
    sliding_window: int = 0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    query_pre_attn_scalar: int = 0

    def set_config(self, text_cfg: dict, layer_types: list[str]):
        super().set_config(text_cfg)
        hidden_size = text_cfg.get("hidden_size", 0)
        self.head_dim = text_cfg.get("head_dim", 0)
        if not self.head_dim and self.num_attention_heads:
            assert hidden_size % self.num_attention_heads == 0
            self.head_dim = hidden_size // self.num_attention_heads
        if "global_head_dim" in text_cfg and text_cfg["global_head_dim"] is not None:
            self.sliding_head_dim = self.head_dim
            self.head_dim = text_cfg["global_head_dim"]
        self.swa_enable = any("sliding_attention" in lt for lt in layer_types)
        if self.sliding_window is None:
            self.sliding_window = 0

    def get_head_dim(self, layer_type: str) -> int:
        if layer_type == "sliding_attention" and self.sliding_head_dim is not None:
            return self.sliding_head_dim
        return self.head_dim

    def get_q_size(self, layer_type: str) -> int:
        return self.num_attention_heads * self.get_head_dim(layer_type)

    def get_kv_size(self, layer_type: str) -> int:
        return self.num_key_value_heads * self.get_head_dim(layer_type)


@dataclass
class MlpBlockConfig(BaseConfig):
    """Configuration of MLP block.

    Attributes:
        intermediate_size: The dimension of MLP layer.
        act: The type of activation.
        num_layers: The number of layers in MLP.
        mlp_bias: Reserved for future use.
        swiglu_limit: Clamp limit for the gated SwiGLU activation (gpt_oss); None when
            the model's activation is not clamped. Read from the model config.
    """
    intermediate_size: int = 0
    act: str = "silu"
    num_layers: int = 3
    mlp_bias: bool = False
    swiglu_limit: float | None = None

    def set_config(self, text_cfg: dict, lm_arch: "LlmArchType"):
        super().set_config(text_cfg)
        self.act = (
            text_cfg.get("hidden_act") or text_cfg.get("hidden_activation")
            or ("gelu_pytorch_tanh" if lm_arch == LlmArchType.GEMMA else "silu")
        )


@dataclass
class MixtureOfExpertsConfig(BaseConfig):
    """Configuration of a Mixture-of-Experts (MoE) feed-forward block.

    For MoE models the dense MLP is replaced by a bank of expert MLPs plus a
    router that activates a subset of experts per token. The per-expert MLP
    dimensions and activation are described by the accompanying MlpBlockConfig.

    Attributes:
        num_experts: Total number of experts in each MoE layer.
        num_experts_per_tok: Number of experts the router activates per token (top-k).
    """
    num_experts: int = 0
    num_experts_per_tok: int = 0

    def set_config(self, text_cfg: dict):
        # gpt_oss/Mixtral use "num_local_experts"; OLMoE uses "num_experts".
        self.num_experts = text_cfg.get(
            "num_local_experts", text_cfg.get("num_experts", 0)
        )
        self.num_experts_per_tok = text_cfg.get(
            "num_experts_per_tok", text_cfg.get("experts_per_token", 0)
        )


@dataclass
class LoraConfig(BaseConfig):
    """Configuration of LoRA adapter for inference.

    If LoRA is enabled, the parallel path for LoRA will be added to the base model,
    so that the output becomes:
        y' = Wx + ((alpha/rank) * ABx) + b
    where the output of the base model is y = Wx + b.

    Attributes:
        lora_alpha: A scaling factor, equal to (alpha/rank), that controls the adapter's influence
            on the base model's weights. Typically, alpha = rank or 2*rank.
        r: The rank of LoRA decomposition.
        layers_to_transform: A list of integers for layers (or transformer blocks) to be adapted,
            for example, [0, 2, 5] means layers 0, 2, and 5 only. None means all layers.
        layers_pattern: Pattern to match layer names in target_modules, if layers_to_transform is
            specified. None means default pattern, which is "layers.<Idx>.<Blk>.<module>". Typically
            use it for custom models that deviate from transformer naming convention.
        target_modules: A list of string for module names or a regex string to apply LoRA.
            For list of names, either an exact match will be performed or it is checked if the name
            of the module ends with a string in the list. Typically names are selected from
            ["k_proj", "v_proj", "q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"].
            For a single string, a regex match will be performed.
    """
    lora_alpha: float = 0
    r: int = 0
    layers_to_transform: list[int] | None = None
    layers_pattern: list[str] | None = None
    target_modules: list[str] | str | None = None


@dataclass
class SpeculativeDecodingConfig(BaseConfig):
    """Configuration of a model in a speculative decoding setup.

    Attributes:
        is_draft: True if the model is a draft model in a speculative decoding setup.
        speculative_budget: Number of tokens the target/draft model processes in parallel decode per step.
            16 for target, 5 for draft model.
    """
    is_draft: bool = False
    speculative_budget: int = 16


@dataclass
class LanguageModelConfig(BaseConfig):
    """Configuration of LLM.

    Attributes:
        model_type: The type of LLM.
        data_type: The data type.
        arch: The architecture of the LLM.
        token_cfg: The settings of tokenizer.
        rope_cfg: The settings of RoPE.
        attn_cfg: The settings of attention block.
        mlp_cfg: The settings of MLP block.
        moe_cfg: The Mixture-of-Experts routing settings; None for dense models.
        hidden_size: The dimension of the embedding.
        num_hidden_layers: The number of transformer blocks.
        max_position_embeddings: The context length.
        rms_norm_eps: The small value in RMS norm to prevent zero devision.
        rms_norm_unit_offset: Whether to add 1 to the weights in
            RMS norm layers before multiplying.  This parameter is
            created during loading, not read from a configuration file.
        layer_types: Type of each layer ('full_attention', 'sliding_attention', or 'conv').
        hidden_size_per_layer_input: Gemma4 per-layer residual embedding input size.
        num_kv_shared_layers: Gemma4 count of final layers that reuse earlier K/V caches.
        use_double_wide_mlp: Gemma4 doubles shared-layer MLP width when enabled.
        attn_logit_softcapping: Gemma 2 attention logit soft capping.
        final_logit_softcapping: Gemma 2 final logit soft capping.
        lm_head_num_splits: The number of head splits by the compiler.
        lm_head_split_dim: The dimension of split head by the compiler.
        conv_L_cache: LFM2 short conv cache length; number of tokens used by conv.
        conv_bias: Whether conv layers use bias (LFM2).
        lora_cfg: The configuration of LoRA for the base model.
        draft_vocab_size: Output dimension of the draft model's lm_head over a
            reduced token subset. Not an input embedding size. 0 for non-draft.
        speculative_decoding_cfg: Speculative-decoding settings; None unless this
            model is part of a speculative-decoding pair.
    """
    model_type: str = ""
    data_type: LlmDataType = LlmDataType.BF16
    arch: LlmArchType = LlmArchType.LLAMA
    token_cfg: TokenEmbedConfig = field(default_factory=TokenEmbedConfig)
    rope_cfg: RoPEConfig = field(default_factory=RoPEConfig)
    attn_cfg: AttentionBlockConfig = field(default_factory=AttentionBlockConfig)
    mlp_cfg: MlpBlockConfig = field(default_factory=MlpBlockConfig)
    moe_cfg: MixtureOfExpertsConfig | None = None
    hidden_size: int = 0
    num_hidden_layers: int = 0
    max_position_embeddings: int = 0
    rms_norm_eps: float = 1e-05
    rms_norm_unit_offset: bool = False
    layer_types: list[str] = field(default_factory=list)
    hidden_size_per_layer_input: int = 0
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False
    attn_logit_softcapping: float | None = None
    final_logit_softcapping: float | None = None
    lm_head_num_splits: int = 1
    lm_head_split_dim: int = 0
    draft_vocab_size: int = 0
    conv_L_cache: int = 3
    conv_bias: bool = False
    lora_cfg: LoraConfig | None  = None
    speculative_decoding_cfg: SpeculativeDecodingConfig | None = None

    def __post_init__(self):
        self.lm_head_split_dim = (
            self.draft_vocab_size if self.draft_vocab_size > 0
            else self.token_cfg.vocab_size
        )

    @staticmethod
    def load(cfg: dict) -> "LanguageModelConfig":
        cfg["token_cfg"] = TokenEmbedConfig(**cfg["token_cfg"])
        if cfg["rope_cfg"].get("rope_scaling") is not None:
            cfg["rope_cfg"]["rope_scaling"] = RopeScalingConfig(**cfg["rope_cfg"]["rope_scaling"])
        cfg["rope_cfg"] = RoPEConfig(**cfg["rope_cfg"])
        cfg["attn_cfg"] = AttentionBlockConfig(**cfg["attn_cfg"])
        cfg["mlp_cfg"] = MlpBlockConfig(**cfg["mlp_cfg"])
        if cfg.get("moe_cfg") is not None:
            cfg["moe_cfg"] = MixtureOfExpertsConfig(**cfg["moe_cfg"])
        cfg["data_type"] = LlmDataType(cfg["data_type"])
        cfg["arch"] = LlmArchType(cfg["arch"])
        if cfg.get("lora_cfg") is not None:
            cfg["lora_cfg"] = LoraConfig(**cfg["lora_cfg"])
        if cfg.get("speculative_decoding_cfg") is not None:
            cfg["speculative_decoding_cfg"] = SpeculativeDecodingConfig(**cfg["speculative_decoding_cfg"])
        lmc = LanguageModelConfig(**cfg)
        lmc._calc_lm_head_splits()
        return lmc

    def set_config(self, text_cfg: dict, dtype: "LlmDataType", lm_arch: "LlmArchType", model_format: "ModelFormat"):
        self.model_type = text_cfg["model_type"]
        self.data_type = dtype
        self.arch = lm_arch

        layer_types = text_cfg.get("layer_types") or []
        sliding_window = text_cfg.get("sliding_window", 0) or 0
        use_sliding_window = text_cfg.get("use_sliding_window", True)
        num_hidden_layers = text_cfg.get("num_hidden_layers", 0)
        if not layer_types:
            if sliding_window > 0 and use_sliding_window:
                layer_types = ["sliding_attention"] * num_hidden_layers
            else:
                layer_types = ["full_attention"] * num_hidden_layers

        self.token_cfg.set_config(text_cfg)
        self.rope_cfg.set_config(text_cfg)
        self.attn_cfg.set_config(text_cfg, layer_types)
        self.mlp_cfg.set_config(text_cfg, lm_arch)
        # gpt_oss/Mixtral advertise experts via "num_local_experts"; OLMoE via "num_experts".
        # Dense models (e.g. Gemma) may still declare these keys with a null/0 value, so
        # gate on a positive expert count rather than mere key presence.
        if text_cfg.get("num_local_experts") or text_cfg.get("num_experts"):
            self.moe_cfg = MixtureOfExpertsConfig()
            self.moe_cfg.set_config(text_cfg)

        self.hidden_size = text_cfg.get("hidden_size", 0)
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = text_cfg.get("max_position_embeddings", 0)
        self.rms_norm_eps = text_cfg.get("rms_norm_eps", text_cfg.get("norm_eps", 1e-05))
        self.rms_norm_unit_offset = (
            model_format == ModelFormat.FORMAT_HF
            and lm_arch == LlmArchType.GEMMA
            and not self.model_type.startswith("gemma4")
        )
        self.layer_types = layer_types
        self.draft_vocab_size = text_cfg.get("draft_vocab_size", 0)

        for key in (
            "attn_logit_softcapping",
            "final_logit_softcapping",
            "conv_L_cache",
            "conv_bias",
            "hidden_size_per_layer_input",
            "num_kv_shared_layers",
            "use_double_wide_mlp",
        ):
            if key in text_cfg:
                setattr(self, key, text_cfg[key])

        # LFM2 adjusts intermediate_size in the model code, not in config.
        # Apply the same adjustment to match the actual weight dimensions.
        if text_cfg.get("block_auto_adjust_ff_dim", False):
            intermediate_size = int(2 * self.mlp_cfg.intermediate_size / 3)
            block_ffn_dim_multiplier = text_cfg.get("block_ffn_dim_multiplier")
            if block_ffn_dim_multiplier is not None:
                intermediate_size = int(block_ffn_dim_multiplier * intermediate_size)
                block_multiple_of = text_cfg.get("block_multiple_of", 256)
                intermediate_size = block_multiple_of * (
                    (intermediate_size + block_multiple_of - 1) // block_multiple_of
                )
            self.mlp_cfg.intermediate_size = intermediate_size

        # Enable mRoPE automatically when sections are provided
        rope_scaling_cfg = getattr(self.rope_cfg, "rope_scaling", None)
        if rope_scaling_cfg and getattr(rope_scaling_cfg, "mrope_section", None):
            rope_scaling_cfg.rope_type = "mrope"
        self._calc_lm_head_splits()

    def is_kv_shared_layer(self, layer_idx: int) -> bool:
        first_shared_layer = self.num_hidden_layers - self.num_kv_shared_layers
        return self.num_kv_shared_layers > 0 and layer_idx >= first_shared_layer

    def get_effective_intermediate_size(self, layer_idx: int) -> int:
        if self.use_double_wide_mlp and self.is_kv_shared_layer(layer_idx):
            return self.mlp_cfg.intermediate_size * 2
        return self.mlp_cfg.intermediate_size

    def set_lora_adapter(self, cfg: dict | None):
        if cfg is None:
            self.lora_cfg = None
        else:
            self.lora_cfg = LoraConfig()
            self.lora_cfg.set_config(cfg)

    def set_speculative_decoding_config(self, cfg: dict | None):
        if cfg is None:
            self.speculative_decoding_cfg = None
        else:
            if self.attn_cfg.swa_enable:
                raise ValueError(
                    "EAGLE3 speculative decoding does not support sliding-window attention"
                )
            self.speculative_decoding_cfg = SpeculativeDecodingConfig()
            self.speculative_decoding_cfg.set_config(cfg)

    def is_lora_target_module(self, base_name: str, module_name: str) -> bool:
        """
        Check if a module is a LoRA target.

        The layer index is checked against lora_cfg.layers_to_transform.
        The module name is checked against lora_cfg.target_modules.
        The pattern name is checked against lora_cfg.layers_pattern.

        Args:
            base_name: The base name for the named module, including layer index.
            module_name: The name of the module without layer and block.

        Returns:
            True if the module is a LoRA target.
        """
        lora_module_name = f"{base_name}.{module_name}"
        if "language_model.model." in lora_module_name:  # LoRA twist to base name.
            lora_module_name = lora_module_name.replace("language_model.model.", "language_model.")

        # Check layer index if specified.
        if self.lora_cfg.layers_to_transform is not None:
            idx = [int(word) for word in base_name.split(".") if word.isdigit()]
            assert len(idx) == 1, f"Found unexpected layer index in base name {base_name}"
            if idx[0] not in self.lora_cfg.layers_to_transform:
                return False

        # Check custom name pattern if specified.
        if self.lora_cfg.layers_pattern is not None:
            if not all(p in lora_module_name for p in self.lora_cfg.layers_pattern):
                return False

        # Check module name.
        if "all-linear" in self.lora_cfg.target_modules:
            return True
        if isinstance(self.lora_cfg.target_modules, list):
            full_match = any(
                lora_module_name == s or module_name == s for s in self.lora_cfg.target_modules
            )
            partial_match = any(lora_module_name.endswith(s) for s in self.lora_cfg.target_modules)
            if full_match or partial_match:
                return True
        else:
            assert isinstance(self.lora_cfg.target_modules, str)
            if re.fullmatch(lora_module_name, self.lora_cfg.target_modules):
                return True

        # Check for bundled name if the module name is not found.
        if module_name == "q_proj" or module_name == "k_proj" or module_name == "v_proj":
            bundle_name = "qkv_proj"
        elif module_name == "gate_proj" or module_name == "up_proj":
            bundle_name = "gate_up_proj"
        else:
            return False
        return bundle_name in self.lora_cfg.target_modules

    def get_lora_rank(self, base_name: str, module_name: str) -> int | None:
        lora_rank = self.lora_cfg.r if self.is_lora_target_module(base_name, module_name) else None
        return lora_rank

    def _calc_lm_head_splits(self):
        # Determine splitting the lm_head by the output feature dimensions is needed.
        num_bytes_per_element = 2
        mla_instr_max_num_channels = 8192
        compiler_max_num_channel_splits = 16
        num_channels_per_block = 16
        max_num_channels_per_split = (
            mla_instr_max_num_channels * compiler_max_num_channel_splits // num_bytes_per_element
        )
        num_channels = (
            self.draft_vocab_size if self.draft_vocab_size > 0
            else self.token_cfg.vocab_size
        )

        max_num_channel_blocks_per_split = ceil_div(
            max_num_channels_per_split, num_channels_per_block
        )
        num_channel_blocks = ceil_div(num_channels, num_channels_per_block)

        num_splits = ceil_div(num_channel_blocks, max_num_channel_blocks_per_split)
        assert num_splits > 0, f"{num_channel_blocks}, {max_num_channel_blocks_per_split}"
        split_dim = ceil_div(num_channel_blocks, num_splits) * num_channels_per_block
        last_split_dim = num_channels - (num_splits - 1) * split_dim

        assert 0 < split_dim <= max_num_channels_per_split
        assert 0 < last_split_dim <= max_num_channels_per_split

        self.lm_head_num_splits = num_splits
        self.lm_head_split_dim = split_dim


@dataclass
class PipelineConfig(BaseConfig):
    """Configuration of VLM pipeline.

    Attributes:
        system_prompt: The system prompt.
        max_num_tokens: The max number of tokens including the input and generated tokens.
        input_token_group_size: The group size of input token.
        input_token_group_offsets: The group offsets of input token.
        future_token_mask_size: The normal cache-model mask bucket size.
        long_context_future_token_mask_size: Optional full-attention override for long contexts.
        return_logits: Return logits at the last layer.
        enable_filter_sharing: Enables filter sharing between group and single models.
        quantize_embeddings: Enables embedding quantization to reduce memory consumption.
        quantize_kv_cache: Enables KV cache quantization to reduce memory consumption.
    """
    system_prompt: str | None = None
    chat_template: str | None = None
    max_num_tokens: int = 4096
    input_token_group_size: int = 1
    input_token_group_offsets: list[int] | None = None
    future_token_mask_size: int = 1
    long_context_future_token_mask_size: int | None = None
    return_logits: bool = False
    enable_filter_sharing: bool = False
    quantize_embeddings: bool = False
    quantize_kv_cache: bool = False

    def set_system_prompt(self, prompt: str | None):
        self.system_prompt = prompt

    def set_chat_template(self, chat_template: str | None):
        self.chat_template = chat_template

    def set_max_num_tokens(self, max_num_tokens: int):
        if max_num_tokens <= 0:
            raise ValueError("max_num_tokens must be greater than zero")
        if max_num_tokens % LONG_CONTEXT_FUTURE_TOKEN_MASK_SIZE:
            raise ValueError("max_num_tokens must be a multiple of 1024")
        self.max_num_tokens = max_num_tokens
        self._set_long_context_future_token_mask_size()

    def set_group_size(self, size: int | None):
        if size is None:
            self.input_token_group_size = 1
            self.input_token_group_offsets = None
            return
        if size <= 0:
            raise ValueError("language_group_size must be greater than zero")

        self.input_token_group_size = size
        self.input_token_group_offsets = list(
            range(0, self.max_num_tokens - size + 1, size)
        )

    def set_future_token_mask_size(self, mask_size: int):
        if mask_size <= 0:
            raise ValueError("future_token_mask_size must be greater than zero")
        self.future_token_mask_size = mask_size
        self._set_long_context_future_token_mask_size()

    def _set_long_context_future_token_mask_size(self) -> None:
        self.long_context_future_token_mask_size = (
            LONG_CONTEXT_FUTURE_TOKEN_MASK_SIZE
            if self.max_num_tokens > LONG_CONTEXT_MIN_TOKENS
            else None
        )

    def get_cache_mask_size(
        self, layer_type: str, context_length: int, *, is_group: bool
    ) -> int:
        """Return the cache bucket size for the model type and context."""
        if (
            layer_type != "sliding_attention"
            and self.long_context_future_token_mask_size is not None
            and context_length > LONG_CONTEXT_MIN_TOKENS
        ):
            return self.long_context_future_token_mask_size
        return self.input_token_group_size if is_group else self.future_token_mask_size

    def set_return_logits(self, return_logits: bool):
        self.return_logits = return_logits

    def set_enable_filter_sharing(self, enable_filter_sharing: bool):
        self.enable_filter_sharing = enable_filter_sharing

    def set_quantize_embeddings(self, quantize_embeddings: bool):
        self.quantize_embeddings = quantize_embeddings

    def set_quantize_kv_cache(self, quantize_kv_cache: bool):
        self.quantize_kv_cache = quantize_kv_cache


@dataclass
class VlmConfig(BaseConfig):
    """Configuration of Vision Language Model.

    Attributes:
        model_name (str): The name of the model.
        model_type (str): The type of the model.
        vm_cfg (VisionModelConfig | None): The settings of vision model.
        mm_cfg (MMConnectionConfig | None): The settings of multi-modal connection.
        lm_cfg (LanguageModelConfig): The settings of language model.
        pipeline_cfg (PipelineConfig): The settings of application pipeline.
    """
    model_name: str = ""
    model_type: VlmArchType | None = None
    vm_cfg: VisionModelConfig | None = None
    mm_cfg: MMConnectionConfig | None = None
    lm_cfg: LanguageModelConfig = field(default_factory=LanguageModelConfig)
    pipeline_cfg: PipelineConfig = field(default_factory=PipelineConfig)

    @staticmethod
    def load(vlm_cfg: dict) -> "VlmConfig":
        if vlm_cfg.get("vm_cfg") is not None:
            vlm_cfg["vm_cfg"]["arch"] = VisionArchType(vlm_cfg["vm_cfg"]["arch"])
            vlm_cfg["vm_cfg"] = VisionModelConfig(**vlm_cfg["vm_cfg"])
        if vlm_cfg.get("mm_cfg") is not None:
            vlm_cfg["mm_cfg"] = MMConnectionConfig(**vlm_cfg["mm_cfg"])
        vlm_cfg["lm_cfg"] = LanguageModelConfig.load(vlm_cfg["lm_cfg"])
        vlm_cfg["pipeline_cfg"] = PipelineConfig(**vlm_cfg["pipeline_cfg"])
        vlm_cfg["model_type"] = VlmArchType(vlm_cfg["model_type"])
        vc = VlmConfig(**vlm_cfg)
        return vc

    def set_config(
        self, dtype: LlmDataType, model_type: str, vm_arch: VisionArchType | None,
        lm_arch: LlmArchType, text_cfg: dict, vision_cfg: dict, model_cfg: dict, *,
        model_format: ModelFormat
    ):
        self.lm_cfg = LanguageModelConfig()
        self.lm_cfg.set_config(text_cfg, dtype, lm_arch, model_format)
        self.vm_cfg = VisionModelConfig()
        self.mm_cfg = MMConnectionConfig()
        if vm_arch is not None:
            self.vm_cfg.set_config(vm_arch, model_cfg, vision_cfg)
            self.mm_cfg.set_config(vm_arch, self.lm_cfg.hidden_size, model_cfg)
        self.pipeline_cfg = PipelineConfig()

    @staticmethod
    def from_hf_config(
        model_format: ModelFormat, model_path: Path, model_cfg: dict,
        image_resolution: list[int] | None = None
    ) -> "VlmConfig":
        """
        Generate SiMa's configuration for VLM
        from a HuggingFace config dict and MLA constraints.

        This function does not access the filesystem.

        Args:
            model_format: The format of the source model.
            model_path: The path of the source model.
            model_cfg: The config dict of the source model.
            image_resolution: The resolution of the input image.

        Returns:
            VlmConfig for the model.
        """
        source_model_name = model_path.name
        is_vlm = _is_vlm_model(model_cfg)
        text_config = model_cfg["text_config"] if is_vlm else model_cfg
        vision_config = model_cfg["vision_config"] if is_vlm else None

        # Check model type
        assert "model_type" in model_cfg and "architectures" in model_cfg
        model_type = model_cfg["model_type"]
        text_model_type = text_config["model_type"]
        vision_model_type = vision_config["model_type"] if is_vlm else None
        vm_arch, lm_arch, gen = get_model_arch_gen(
            is_vlm, model_type, text_model_type, vision_model_type
        )
        assert vm_arch is None or vm_arch in VisionArchType.values()
        assert lm_arch in LlmArchType.values()

        # Check data type
        if model_cfg.get("dtype"):
            data_type = LlmDataType(model_cfg["dtype"])
        elif model_cfg.get("torch_dtype"):
            data_type = LlmDataType(model_cfg["torch_dtype"])
        elif text_config.get("dtype"):
            data_type = LlmDataType(text_config["dtype"])
        elif text_config.get("torch_dtype"):
            data_type = LlmDataType(text_config["torch_dtype"])
        else:
            assert model_format == ModelFormat.FORMAT_GGUF
            gguf_type = model_cfg["data_type"]
            assert gguf_type in GgufFileType.values()
            gguf_name = GgufFileType.name_from_value(gguf_type)
            data_type = LlmDataType.value_from_name(gguf_name)

        vlm_cfg = VlmConfig()
        vlm_cfg.set_config(
            data_type, model_type, vm_arch, lm_arch, text_config, vision_config,
            model_cfg, model_format=model_format
        )
        vlm_cfg.model_name = source_model_name

        if is_vlm:
            vlm_cfg.model_type = VlmArchType(f"vlm-{model_type}")
            if image_resolution is None and vlm_cfg.vm_cfg.arch in (
                    VisionArchType.QWEN2_VISION_ENCODER,
                    VisionArchType.QWEN3_VISION_ENCODER,
                    VisionArchType.GEMMA4_VISION_ENCODER,
                ):
                raise RuntimeError(
                    f"{vlm_cfg.vm_cfg.arch} models require --input_height and --input_width arguments"
                )
            if image_resolution is not None:
                if vlm_cfg.vm_cfg.arch == VisionArchType.SIGLIP2:
                    height, width = image_resolution
                    if height * width > 262144:
                        raise RuntimeError(f"Input image resolution ({height}x{width}) exceeds the maximum allowed 262,144 pixels for single-block processing in Siglip2.")
                    if height % 32 != 0 or width % 32 != 0:
                        raise RuntimeError(f"Input image dimensions ({height}x{width}) must be divisible by 32 for Siglip2 based models.")
                    vlm_cfg.vm_cfg.image_size = image_resolution
                elif vlm_cfg.vm_cfg.arch in (VisionArchType.QWEN2_VISION_ENCODER, VisionArchType.QWEN3_VISION_ENCODER, VisionArchType.GEMMA4_VISION_ENCODER):
                    height, width = image_resolution
                    divisor = vlm_cfg.vm_cfg.patch_size * vlm_cfg.vm_cfg.spatial_merge_size
                    if height % divisor != 0 or width % divisor != 0:
                        raise RuntimeError(
                            f"For {vlm_cfg.vm_cfg.arch}, image dimensions ({height}x{width}) must be divisible by "
                            f"(patch_size * spatial_merge_size), which is {divisor}."
                        )
                    vlm_cfg.vm_cfg.image_size = image_resolution
                else:
                    sima_log_warning("Ignoring --input_height and --input_width as the model is not Siglip2, Qwen-VL, or Gemma4 based.")
            if vlm_cfg.vm_cfg.arch == VisionArchType.SIGLIP2:
                if isinstance(vlm_cfg.vm_cfg.image_size, list):
                    h, w = vlm_cfg.vm_cfg.image_size
                else:
                    h = w = vlm_cfg.vm_cfg.image_size
                ps = vlm_cfg.vm_cfg.patch_size
                df = vlm_cfg.mm_cfg.downsample_factor
                if h % (ps * df) != 0 or w % (ps * df) != 0:
                    raise ValueError(
                        f"Image dimensions ({h}x{w}) must be divisible by "
                        f"patch_size * downsample_factor ({ps * df})."
                    )
                vlm_cfg.mm_cfg.mm_tokens_per_image = (h // ps // df) * (w // ps // df)
            elif vlm_cfg.vm_cfg.arch in (VisionArchType.QWEN2_VISION_ENCODER, VisionArchType.QWEN3_VISION_ENCODER, VisionArchType.GEMMA4_VISION_ENCODER):
                if isinstance(vlm_cfg.vm_cfg.image_size, list):
                    h, w = vlm_cfg.vm_cfg.image_size
                else:
                    h = w = vlm_cfg.vm_cfg.image_size
                grid_h = h // vlm_cfg.vm_cfg.patch_size // vlm_cfg.vm_cfg.spatial_merge_size
                grid_w = w // vlm_cfg.vm_cfg.patch_size // vlm_cfg.vm_cfg.spatial_merge_size
                vlm_cfg.mm_cfg.mm_tokens_per_image = grid_h * grid_w
            elif vlm_cfg.vm_cfg.arch == VisionArchType.CLIP:
                imgs = vlm_cfg.vm_cfg.image_size
                ps = vlm_cfg.vm_cfg.patch_size
                vlm_cfg.mm_cfg.mm_tokens_per_image = (imgs // ps) ** 2
            if vlm_cfg.model_type in (VlmArchType.VLM_LFM2_VL, VlmArchType.VLM_GEMMA4):
                if not vlm_cfg.vm_cfg.temporal_patch_size:
                    vlm_cfg.vm_cfg.temporal_patch_size = 1
        else:
            # Keep GGUF Gemma4 naming consistent with the existing VLM_GEMMA4 language path.
            if lm_arch == LlmArchType.GEMMA and gen == "4":
                vlm_cfg.model_type = VlmArchType.VLM_GEMMA4
            else:
                vlm_cfg.model_type = VlmArchType(f"llm-{lm_arch}{gen or ''}")
            if vlm_cfg.model_type == VlmArchType.LLM_PHI3:
                # Phi3 has a sliding window that is much larger than the max_num_tokens
                # used during compilation, hence disable the sliding window.
                vlm_cfg.lm_cfg.attn_cfg.swa_enable = False
                vlm_cfg.lm_cfg.layer_types = ["full_attention"] * vlm_cfg.lm_cfg.num_hidden_layers
            vlm_cfg.mm_cfg = None
            vlm_cfg.vm_cfg = None

        apply_mla_constraint(vlm_cfg)
        return vlm_cfg

    @property
    def is_multimodal(self) -> bool:
        return self.vm_cfg is not None and self.mm_cfg is not None

    @property
    def is_supported_multimodal(self) -> bool:
        return (
            self.is_multimodal
            and not (self.model_type == VlmArchType.VLM_GEMMA3 and self.vm_cfg.image_size > 448)
        )

    @property
    def num_vision_layers(self) -> int:
        if self.vm_cfg is None:
            return 0
        # LLaVA consumes the penultimate vision hidden state.
        return self.vm_cfg.num_hidden_layers - int(
            self.model_type == VlmArchType.VLM_LLAVA
        )

    def get_vision_model_names(self, model_name: str) -> list[str]:
        return [
            f"{model_name}_layer{layer_idx}"
            for layer_idx in range(self.num_vision_layers)
        ]

    def config_pipeline(
        self,
        system_prompt: str | None,
        chat_template: str | None,
        max_num_tokens: int,
        language_group_size: int | None,
        future_token_mask_size: int,
    ):
        self.pipeline_cfg.set_system_prompt(system_prompt)
        self.pipeline_cfg.set_chat_template(chat_template)
        self.pipeline_cfg.set_max_num_tokens(max_num_tokens)
        self.pipeline_cfg.set_group_size(language_group_size)
        self.pipeline_cfg.set_future_token_mask_size(future_token_mask_size)

        if (
            self.lm_cfg.attn_cfg.swa_enable
            and self.pipeline_cfg.input_token_group_offsets
            and self.pipeline_cfg.input_token_group_size
            >= self.lm_cfg.attn_cfg.sliding_window
        ):
            raise ValueError(
                "language_group_size must be smaller than sliding_window "
                "for models with sliding attention"
            )

    def get_layer_ids(self) -> list[LayerID]:
        """
        Get IDs of all layers that comprise the model.
        """
        lm_cfg = self.lm_cfg
        pipeline_cfg = self.pipeline_cfg

        layers = []
        layer_types = getattr(lm_cfg, "layer_types", [])
        is_speculative_draft = (
            lm_cfg.speculative_decoding_cfg is not None
            and lm_cfg.speculative_decoding_cfg.is_draft
        )

        if layer_types:
            if len(layer_types) != lm_cfg.num_hidden_layers:
                raise ValueError(
                    f"layer_types length ({len(layer_types)}) must match "
                    f"num_hidden_layers ({lm_cfg.num_hidden_layers})."
                )
            has_attn = False
            has_conv = False
            conv_layer_indices: list[int] = []
            for i, t in enumerate(layer_types):
                if t == "conv":
                    has_conv = True
                    conv_layer_indices.append(i)
                    layers.append(LayerID("group_conv", i))
                    layers.append(LayerID("single_conv", i))
                elif t == "full_attention" or t == "sliding_attention":
                    has_attn = True
                    if lm_cfg.moe_cfg is not None:
                        # MoE layers replace the single post (MLP) part with a router
                        # plus one model per expert, for both group and single paths.
                        num_experts = lm_cfg.moe_cfg.num_experts
                        layers.append(LayerID("group_pre", i))
                        # The last layer's post (router/experts/combine, and the
                        # combine's folded-in lm_head) runs single-token only,
                        # mirroring the non-MoE path which skips group_post on the
                        # last layer. So skip the group router/experts/combine there.
                        if i < lm_cfg.num_hidden_layers - 1 or is_speculative_draft:
                            layers.append(LayerID("group_router", i))
                            layers.extend(
                                LayerID("group_expert", i, e) for e in range(num_experts)
                            )
                            layers.append(LayerID("group_weightedsum", i))
                        layers.append(LayerID("single_pre", i))
                        layers.append(LayerID("single_router", i))
                        layers.extend(
                            LayerID("single_expert", i, e) for e in range(num_experts)
                        )
                        layers.append(LayerID("single_weightedsum", i))
                    else:
                        layers.append(LayerID("group_pre", i))
                        if i < lm_cfg.num_hidden_layers - 1 or is_speculative_draft:
                            layers.append(LayerID("group_post", i))
                        layers.append(LayerID("single_pre", i))
                        layers.append(LayerID("single_post", i))
                else:
                    raise ValueError(f"Unsupported layer type: {t}")
            # Cache models are shared across layers; include only for kinds that exist
            if has_attn:
                group_cache_indices = group_cache_model_indices(pipeline_cfg)
                single_cache_indices = single_cache_model_indices(pipeline_cfg)
                has_sliding_attn = "sliding_attention" in layer_types
                separate_sliding_cache = (
                    lm_cfg.attn_cfg.sliding_head_dim is not None
                    and lm_cfg.attn_cfg.sliding_head_dim != lm_cfg.attn_cfg.head_dim
                )
                terminal_sliding_cache = False
                if has_sliding_attn and not separate_sliding_cache:
                    sliding_window = lm_cfg.attn_cfg.sliding_window
                    mask_context = min(sliding_window, pipeline_cfg.max_num_tokens)
                    sliding_cache_mask_differs = any(
                        pipeline_cfg.get_cache_mask_size(
                            "full_attention", mask_context, is_group=is_group
                        )
                        != pipeline_cfg.get_cache_mask_size(
                            "sliding_attention", mask_context, is_group=is_group
                        )
                        for is_group in (False, True)
                    )
                    terminal_sliding_cache = (
                        sliding_cache_mask_differs
                        and sliding_window <= pipeline_cfg.max_num_tokens
                    )
                    if not sliding_cache_mask_differs:
                        group_cache_indices = group_shared_sliding_cache_model_indices(
                            pipeline_cfg, sliding_window
                        )
                        single_cache_indices = single_shared_sliding_cache_model_indices(
                            pipeline_cfg, sliding_window
                        )
                layers.extend(LayerID("group_cache", n) for n in group_cache_indices)
                layers.extend(LayerID("single_cache", n) for n in single_cache_indices)
                if separate_sliding_cache:
                    layers.extend(
                        LayerID("group_sliding_cache", n)
                        for n in group_sliding_cache_model_indices(
                            pipeline_cfg, lm_cfg.attn_cfg.sliding_window
                        )
                    )
                    layers.extend(
                        LayerID("single_sliding_cache", n)
                        for n in single_sliding_cache_model_indices(
                            pipeline_cfg, lm_cfg.attn_cfg.sliding_window
                        )
                    )
                elif terminal_sliding_cache:
                    layers.append(LayerID(
                        "group_sliding_cache",
                        _cache_model_index(
                            pipeline_cfg,
                            "sliding_attention",
                            sliding_window,
                            pipeline_cfg.input_token_group_size,
                            is_group=True,
                        ),
                    ))
                    layers.append(LayerID(
                        "single_sliding_cache",
                        _cache_model_index(
                            pipeline_cfg,
                            "sliding_attention",
                            sliding_window,
                            1,
                            is_group=False,
                        ),
                    ))
            if has_conv and layer_types[-1] == "conv":
                layers.append(LayerID("conv_post_final", lm_cfg.num_hidden_layers - 1))
        else:
            # Fallback: attention-only if no layer_types
            layers.extend(LayerID("group_pre", n) for n in range(lm_cfg.num_hidden_layers))
            num_group_post_layers = (
                lm_cfg.num_hidden_layers
                if is_speculative_draft
                else lm_cfg.num_hidden_layers - 1
            )
            layers.extend(
                LayerID("group_post", n) for n in range(num_group_post_layers)
            )
            layers.extend(LayerID("group_cache", n) for n in group_cache_model_indices(pipeline_cfg))
            layers.extend(LayerID("single_pre", n) for n in range(lm_cfg.num_hidden_layers))
            layers.extend(LayerID("single_post", n) for n in range(lm_cfg.num_hidden_layers))
            layers.extend(LayerID("single_cache", n) for n in single_cache_model_indices(pipeline_cfg))
        if self.vm_cfg is not None and self.is_supported_multimodal:
            layers.extend(
                LayerID("vision", n)
                for n in range(self.num_vision_layers)
            )
        if is_speculative_draft:
            layers.append(LayerID("group_draft_fc", 0))
            layers.append(LayerID("single_draft_fc", 0))

        if (
            self.model_type == VlmArchType.VLM_GEMMA4
            and lm_cfg.hidden_size_per_layer_input > 0
        ):
            layers.append(LayerID("group_per_layer", 0))
            layers.append(LayerID("single_per_layer", 0))

        return layers


def _load_json_file(path: Path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in: {path}")


def _write_json_file(data: str, path: Path):
    with open(path, "w") as outfile:
        outfile.write(data)


def model_file_type(path: Path) -> ModelFormat:
    """
    Infer a model's file type.  Raise an exception if it is unknown.
    This function may try to access the file at the given path.
    """
    if str(path).endswith(".gguf"):
        return ModelFormat.FORMAT_GGUF
    elif path.is_dir():
        return ModelFormat.FORMAT_HF
    else:
        raise ValueError("Can't determine format of model file " + str(path))



def _is_vlm_model(hf_cfg: dict) -> bool:
    """Check if a HuggingFace model is VLM.

    There are two indications in HF's config file:
     - The "architectures" field is a list containing "xxxForConditionalGeneration";
     - There are separate "text_config" and "vision_config".
    """
    return "ConditionalGeneration" in hf_cfg["architectures"][0] \
        and "vision_config" in hf_cfg


def get_model_arch_gen(
    is_vlm: bool, model_type: str, text_type: str, vision_type : str
    ) -> tuple[VisionArchType | None, LlmArchType, str]:
    """Derive VLM architecture and version from the model types.

    Args:
        is_vlm: Is the model a vlm or llm
        model_type: The type of a model.
        text_type: The type of language model.
        vision_type: If vlm, type of vision model.

    Returns:
        Tuple of vision architecture, LLM architecture and version.
    """
    lm_arch = None
    t_reg = re.fullmatch(
        r"(?P<arch>[a-zA-Z]+(?:_[a-zA-Z]+)*?)(?P<gen>\d+(?:_\d+)*)?(?:_vl|_vision)?(?:_text)?",
        text_type,
    )
    for arch in LlmArchType.values():
        if arch == t_reg.group("arch"):
            lm_arch = LlmArchType(arch)
            break

    if lm_arch is None:
        raise NotImplementedError(f"Unsupported LLM architecture: {text_type}")

    if is_vlm:
        vm_arch = VisionArchType(vision_type.split("_vision_model")[0])
    else:
        vm_arch = None

    gen = t_reg.group("gen")
    return vm_arch, lm_arch, gen


def llm_parameter_count(text_cfg: dict) -> int:
    """Calculate parameter count of an LLM model.

    Args:
        cfg (dict): The configuration dictionary of the model.

    Returns:
        The size of the model in terms of the parameter count.
    """
    vocab_size = text_cfg["vocab_size"]

    embed_dim = text_cfg["hidden_size"]
    intermediate_size = text_cfg["intermediate_size"]
    num_hidden_layers = text_cfg["num_hidden_layers"]

    num_attention_heads = text_cfg["num_attention_heads"]
    num_key_value_heads = text_cfg["num_key_value_heads"]

    head_dim = text_cfg["head_dim"]

    # (vocab_size, embed_dim)
    count_token_embedding = vocab_size * embed_dim

    # Group Query Attention
    w_q = w_o = embed_dim * num_attention_heads * head_dim
    w_k = w_v = embed_dim * num_key_value_heads * head_dim
    count_attention = w_q + w_o + w_k + w_v

    # 3-layer MLP
    count_mlp = 3 * embed_dim * intermediate_size

    count_layer_rms = 2 * embed_dim

    count_lm_rms = embed_dim

    total = count_attention + count_mlp + count_layer_rms
    total *= num_hidden_layers
    total += count_token_embedding + count_lm_rms
    return total


def apply_mla_constraint(vlm_cfg: VlmConfig) -> None:
    """Apply MLA constraints.

    TODO: modify RoPE if context_length (max_position_embeddings) is changed.

    Args:
        vlm_cfg (VlmConfig): The configuration of VLM model.

    Returns:
        None. Change is made in place.
    """
    cfg = vlm_cfg.lm_cfg
    if cfg is not None:
        for key, value in MLA_CONSTRAINTS.items():
            if getattr(cfg, key) > value:
                setattr(cfg, key, value)


def _cache_model_index(
    cfg: PipelineConfig,
    layer_type: str,
    context_length: int,
    num_tokens: int,
    *,
    is_group: bool,
) -> int:
    mask_size = cfg.get_cache_mask_size(
        layer_type, context_length, is_group=is_group
    )
    if mask_size <= num_tokens:
        return context_length - num_tokens
    return min(round_up_to(context_length, mask_size), cfg.max_num_tokens) - num_tokens


def _group_cache_model_indices(cfg: PipelineConfig, layer_type: str) -> list[int]:
    if cfg.input_token_group_offsets is None:
        raise RuntimeError("Group token offsets have not been computed")

    return sorted({
        _cache_model_index(
            cfg,
            layer_type,
            offset + cfg.input_token_group_size,
            cfg.input_token_group_size,
            is_group=True,
        )
        for offset in cfg.input_token_group_offsets
    })


def group_cache_model_indices(cfg: PipelineConfig) -> list[int]:
    """
    Get the indices of all group cache models for the pipeline configuration.

    Returns:
        Indices of group cache models in ascending order.
    """
    return _group_cache_model_indices(cfg, "full_attention")


def group_sliding_cache_model_indices(cfg: PipelineConfig, sliding_window: int) -> list[int]:
    """
    Get the indices of all group sliding-window cache models.

    For sliding attention the effective cache_model_token_idx saturates at
    sliding_window - group_size once the window is fully filled.  Any group offset
    beyond that point maps to the same compiled model, so we only keep offsets
    strictly below the transition and add the transition itself.

    Returns:
        Indices of group sliding cache models in ascending order.
    """
    transition = sliding_window - cfg.input_token_group_size
    indices = [
        n for n in _group_cache_model_indices(cfg, "sliding_attention")
        if n < transition
    ]
    if transition > 0:
        indices.append(transition)
    return indices


def group_shared_sliding_cache_model_indices(
    cfg: PipelineConfig, sliding_window: int
) -> list[int]:
    """Get cache indices needed by sliding attention sharing the full cache."""
    return sorted(set(group_cache_model_indices(cfg)) | set(
        group_sliding_cache_model_indices(cfg, sliding_window)
    ))


def _single_cache_model_indices(cfg: PipelineConfig, layer_type: str) -> list[int]:
    return sorted({
        _cache_model_index(
            cfg, layer_type, context_length, 1, is_group=False
        )
        for context_length in range(1, cfg.max_num_tokens + 1)
    })


def single_cache_model_indices(cfg: PipelineConfig) -> list[int]:
    """
    Get the indices of all single cache models for the pipeline configuration.

    Returns:
        Indices of single cache models in ascending order.
    """
    # A model is used for each batch of future_token_mask_size tokens.
    # The model's index is the last token's index in the batch.  The last
    # batch's index is the last token's index, even if it is not evenly
    # spaced.  For example, given future_token_mask_size=192 and
    # max_num_tokens=1024, the indices will be 191, 383, 575, 767, 959, 1023.
    return _single_cache_model_indices(cfg, "full_attention")


def single_sliding_cache_model_indices(cfg: PipelineConfig, sliding_window: int) -> list[int]:
    """
    Get the indices of all single-token sliding-window cache models.

    For single-token decode, the effective cache index saturates at sliding_window - 1 once
    the window is fully filled. Only the future_token_mask_size-aligned bucket indices up to
    and including sliding_window - 1 are unique; everything beyond maps to the same compiled
    model.

    Returns:
        Indices of single sliding cache models in ascending order.
    """
    return [
        n for n in _single_cache_model_indices(cfg, "sliding_attention")
        if n < sliding_window
    ]


def single_shared_sliding_cache_model_indices(
    cfg: PipelineConfig, sliding_window: int
) -> list[int]:
    """Get cache indices needed by single-token full and sliding attention."""
    return sorted(set(single_cache_model_indices(cfg)) | set(
        single_sliding_cache_model_indices(cfg, sliding_window)
    ))


if __name__ == "__main__":
    import sys
    from sima_lmm.gguf.gguf_conversion import GgufModel
    from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel
    from sima_lmm.preproc.vlm_helper import VlmHelper

    input_path = sys.argv[1]
    lora_path = sys.argv[2] if len(sys.argv) == 3 else None

    print(f"Model is: {input_path}")
    print(f"LoRA is: {lora_path}")

    model_path = Path(input_path)
    model_format = model_file_type(model_path)
    match model_format:
        case ModelFormat.FORMAT_GGUF:
            gguf_model = GgufModel(model_path)
            model_config = gguf_model.model_config
        case ModelFormat.FORMAT_HF:
            hf_cache = LocalHuggingFaceModel.create_from_directory(
                directory=model_path,
                layer_names=None,
            )
            model_config = hf_cache.config
        case _:
            raise ValueError("Can't determine format of model file " + str(input_path))

    vlm_cfg = VlmConfig.from_hf_config(model_format, model_path, model_config)

    if lora_path is not None:
        lora_config = LocalHuggingFaceModel.load_lora_adapter(lora_path)
        vlm_cfg.lm_cfg.set_lora_adapter(lora_config)

    output_filename = (
        f"sima-{vlm_cfg.model_type}-{vlm_cfg.lm_cfg.arch.value}"
        f"-{vlm_cfg.lm_cfg.gen.value}-{vlm_cfg.lm_cfg.size}.json"
    )
    json_vlm = json.dumps(asdict(vlm_cfg), indent=4)
    _write_json_file(json_vlm, Path(output_filename))

    # Load back from the saved file.
    sima_dict = _load_json_file(Path(output_filename))
    sima_vlm = VlmConfig.load(sima_dict)

    assert sima_vlm == vlm_cfg
    print("Test of round-trip VLM config is successful!")
