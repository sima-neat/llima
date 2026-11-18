"""Component planning for the LFM2 Audio compiler, separate from runtime code."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from sima_lmm.config.vlm_config import LlmArchType, LlmDataType, ModelFormat, VlmConfig
from sima_lmm.hf.lfm2_audio import Lfm2AudioCheckpoint
from sima_lmm.model.base import BaseModel
from sima_lmm.model.language_conv_model import LanguageConvModel
from sima_lmm.model.language_pre_model import LanguagePreModel
from sima_lmm.model.lfm2_audio_model import (
    AudioAdapterModel, AudioCacheModel, AudioConformerLayerModel, AudioDepthPostModel,
    AudioEncoderSubsamplingModel, AudioFinalConvModel, AudioLanguagePostModel, AudioProjectionModel,
)


@dataclass
class AudioCompileUnit:
    component: str
    role: str
    model: BaseModel


def depth_configuration(checkpoint: Lfm2AudioCheckpoint) -> VlmConfig:
    source = checkpoint.config["depthformer"]
    width, layers = source["dim"], source["layers"]
    inner = 256 * ((int(2 * (4 * width) / 3) + 255) // 256)
    cfg = VlmConfig()
    # This is a dedicated audio component, not a registered ordinary LFM2 LLM.
    cfg.model_type = "audio-lfm2_depth"
    cfg.lm_cfg.set_config({
        "model_type": "lfm2_audio_depth", "hidden_size": width,
        "num_hidden_layers": layers, "num_attention_heads": 32, "num_key_value_heads": 8,
        "intermediate_size": inner, "vocab_size": 2049, "norm_eps": 1e-5,
        "rope_theta": 1000000., "layer_types": ["full_attention"] * layers,
    }, LlmDataType.BF16, LlmArchType.LFM, ModelFormat.FORMAT_HF)
    cfg.pipeline_cfg.future_token_mask_size = checkpoint.config["codebooks"]
    return cfg


def plan_audio_components(
    checkpoint: Lfm2AudioCheckpoint,
    output: Path,
    *,
    context_length: int = 1024,
    group_size: int = 16,
    audio_frames: int = 256,
    decoder_frames: int = 12,
) -> list[AudioCompileUnit]:
    """Plan fixed-shape native components with explicit BF16 cache interfaces.

    audio_frames counts input log-mel frames. decoder_frames counts generated
    eight-codebook frames; each is expanded to six decoder-network timesteps.
    """
    for name, value in (("context_length", context_length), ("group_size", group_size),
                        ("audio_frames", audio_frames), ("decoder_frames", decoder_frames)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if group_size > context_length:
        raise ValueError("group_size cannot exceed context_length")
    cfg = VlmConfig.from_hf_config(
        ModelFormat.FORMAT_HF, checkpoint.directory, checkpoint.backbone_config
    )
    cfg.pipeline_cfg.set_max_num_tokens(context_length)
    cfg.pipeline_cfg.quantize_kv_cache = False
    cfg.pipeline_cfg.quantize_embeddings = False
    cfg.pipeline_cfg.future_token_mask_size = context_length
    units = []

    def add(component, role, model):
        units.append(AudioCompileUnit(component, role, model))

    def language(component, source, config, lengths, capacity, *, depth=False):
        config = deepcopy(config)
        config.pipeline_cfg.max_num_tokens = capacity
        config.pipeline_cfg.future_token_mask_size = capacity
        for length in sorted(set(lengths)):
            for index, kind in enumerate(config.lm_cfg.layer_types):
                prefix = f"{component}_n{length}_layer{index}"
                common = dict(sima_path=output, hf_model=source, num_tokens=length, layer_idx=index)
                if kind == "conv":
                    add(component, "conv", LanguageConvModel(
                        config, prefix + "_conv", final_softcapping=None, **common
                    ))
                    if index == config.lm_cfg.num_hidden_layers - 1:
                        add(component, "final", AudioFinalConvModel(
                            config, prefix + "_final", final_softcapping=None, **common
                        ))
                else:
                    add(component, "pre", LanguagePreModel(config, prefix + "_pre", **common))
                    post = AudioDepthPostModel if depth else AudioLanguagePostModel
                    add(component, "post", post(config, prefix + "_post", final_softcapping=None, **common))
            add(component, "cache", AudioCacheModel(
                config, f"{component}_n{length}_cache{capacity}", sima_path=output, hf_model=source,
                num_tokens=length, token_idx=capacity - length, logit_softcapping=None,
            ))

    language("backbone", checkpoint.language_backbone(), cfg, [1, group_size], context_length)
    source = checkpoint.audio_source()
    for index in range(cfg.lm_cfg.lm_head_num_splits):
        begin = index * cfg.lm_cfg.lm_head_split_dim
        end = min(begin + cfg.lm_cfg.lm_head_split_dim, cfg.lm_cfg.token_cfg.vocab_size)
        add("backbone", "text_head", AudioProjectionModel(
            cfg, f"backbone_text_head{index}", sima_path=output, hf_model=source,
            num_tokens=1, input_dim=cfg.lm_cfg.hidden_size, weight_name="lfm.embed_tokens.weight",
            output_begin=begin, output_end=end,
        ))

    encoder = checkpoint.config["encoder"]
    if (
        encoder.get("subsampling") != "dw_striding" or encoder.get("subsampling_factor") != 8
        or encoder.get("causal_downsampling") or encoder.get("self_attention_model") != "rel_pos"
        or encoder.get("conv_norm_type") != "batch_norm" or encoder.get("xscaling")
        or encoder.get("att_context_size") != [-1, -1] or encoder.get("reduction") is not None
        or encoder.get("conv_context_size") is not None
    ):
        raise ValueError("Unsupported LFM2 Audio encoder configuration")
    encoded_frames = (audio_frames + 7) // 8
    add("encoder", "subsampling", AudioEncoderSubsamplingModel(
        cfg, f"encoder_subsampling_f{audio_frames}", sima_path=output, hf_model=source,
        input_frames=audio_frames, feature_size=encoder["feat_in"],
        channels=encoder["subsampling_conv_channels"],
    ))
    for index in range(encoder["n_layers"]):
        add("encoder", "layer", AudioConformerLayerModel(
            cfg, f"encoder_n{encoded_frames}_layer{index}", sima_path=output, hf_model=source,
            num_tokens=encoded_frames, layer_idx=index, width=encoder["d_model"],
            heads=encoder["n_heads"], kernel_size=encoder["conv_kernel_size"],
        ))
    add("encoder", "adapter", AudioAdapterModel(
        cfg, f"encoder_n{encoded_frames}_adapter", sima_path=output, hf_model=source,
        num_tokens=encoded_frames, input_dim=encoder["d_model"],
    ))

    depth_cfg = depth_configuration(checkpoint)
    language("depth", checkpoint.depth_backbone(), depth_cfg, [1], checkpoint.config["codebooks"], depth=True)
    add("depth", "input", AudioProjectionModel(
        cfg, "depth_input", sima_path=output, hf_model=source, num_tokens=1,
        input_dim=cfg.lm_cfg.hidden_size, weight_name="depth_linear.weight",
    ))
    for index in range(checkpoint.config["codebooks"]):
        base = f"depth_embeddings.{index}"
        add("depth", "head", AudioProjectionModel(
            depth_cfg, f"depth_head{index}", sima_path=output, hf_model=source, num_tokens=1,
            input_dim=depth_cfg.lm_cfg.hidden_size, weight_name=f"{base}.to_logits.weight",
            norm_name=f"{base}.embedding_norm", norm_epsilon=1e-5,
        ))

    detok = checkpoint.detokenizer_backbone()
    detok_cfg = VlmConfig.from_hf_config(ModelFormat.FORMAT_HF, detok.hf_cache, detok.config)
    decoder_tokens = decoder_frames * 6
    decoder_capacity = max(128, decoder_tokens + detok_cfg.lm_cfg.attn_cfg.sliding_window - 1)
    language("detokenizer", detok, detok_cfg, [decoder_tokens], decoder_capacity)
    add("detokenizer", "spectrum", AudioProjectionModel(
        detok_cfg, f"detokenizer_n{decoder_tokens}_spectrum", sima_path=output,
        hf_model=checkpoint.detokenizer_source(), num_tokens=decoder_tokens,
        input_dim=detok_cfg.lm_cfg.hidden_size, weight_name="lin.weight",
    ))
    return units
