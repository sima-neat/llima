from dataclasses import dataclass, field, fields
from pathlib import Path

from sima_lmm.config.vlm_config import BaseConfig


@dataclass
class WhisperConfig(BaseConfig):
    model_type: str = ""
    d_model: int = 768
    encoder_attention_heads: int = 12
    encoder_layers: int = 12
    decoder_attention_heads: int = 12
    decoder_layers: int = 12
    max_source_positions: int = 1500
    max_target_positions: int = 448
    num_mel_bins: int = 80
    suppress_tokens: list[int] = field(default_factory=list)
    vocab_size: int = 51865
    activation_function: str = "gelu"
    language_detect_enabled: bool = False
    num_languages: int = 99
    language_token_ids: list[int] = field(default_factory=list)
    language_codes: list[str] = field(default_factory=list)

    @staticmethod
    def from_hf_config(model_path: Path, model_cfg: dict) -> "WhisperConfig":
        architectures = model_cfg["architectures"]
        if len(architectures) != 1 or architectures[0] != "WhisperForConditionalGeneration":
            raise NotImplementedError(
                "Currently only WhisperForConditionalGeneration is supported."
                f" Got {architectures[0]}"
            )

        # Implement a simplified version to extract the HF configuration dictionary.
        field_names = set(x.name for x in fields(WhisperConfig))
        filtered_model_cfg = {
            name: value
            for name, value in model_cfg.items() if name in field_names
        }
        return WhisperConfig(**filtered_model_cfg)

    @staticmethod
    def load(model_cfg: dict) -> "WhisperConfig":
        return WhisperConfig(**model_cfg)

    @property
    def encoder_head_dim(self) -> int:
        return self.d_model // self.encoder_attention_heads

    @property
    def decoder_head_dim(self) -> int:
        return self.d_model // self.decoder_attention_heads
