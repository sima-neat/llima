"""Checkpoint ingestion for developing LFM2 Audio components.

This module does not register the complete audio model with llima-compile.
"""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from safetensors import safe_open
from transformers import Lfm2Config

from sima_lmm.config.vlm_config import ModelFormat, VlmConfig
from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel


LFM2_AUDIO_ARCHITECTURE = "Lfm2AudioForConditionalGeneration"


def _positive_integer(config: dict, key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"LFM2 Audio requires a positive integer {key}, got {value!r}")
    return value


def _backbone_config(config: dict) -> dict:
    if config.get("architectures") != [LFM2_AUDIO_ARCHITECTURE]:
        raise ValueError(f"Expected architectures=[{LFM2_AUDIO_ARCHITECTURE!r}]")
    for key in ("lfm", "encoder", "depthformer", "preprocessor"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"LFM2 Audio requires the {key} configuration object")
    if config.get("quantization_config") or config["lfm"].get("quantization_config"):
        raise ValueError("Pre-quantized LFM2 Audio checkpoints are not supported by this adapter")
    if _positive_integer(config, "codebooks") != 8:
        raise ValueError("The LFM2 Audio adapter currently requires eight codebooks")
    for key in ("interleaved_n_text", "interleaved_n_audio"):
        _positive_integer(config, key)

    source = config["lfm"]
    if source.get("model_type") != "lfm2":
        raise ValueError("LFM2 Audio requires lfm.model_type='lfm2'")
    if source.get("architectures") != ["Lfm2ForCausalLM"]:
        raise ValueError("LFM2 Audio requires lfm.architectures=['Lfm2ForCausalLM']")
    for key in (
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "vocab_size", "conv_L_cache",
    ):
        _positive_integer(source, key)
    schedule = source.get("layer_types")
    if not isinstance(schedule, list) or len(schedule) != source["num_hidden_layers"]:
        raise ValueError("LFM2 Audio layer_types must match num_hidden_layers")
    if any(layer not in ("conv", "full_attention") for layer in schedule):
        raise ValueError("LFM2 Audio backbone supports only conv and full_attention layers")
    heads, kv_heads = source["num_attention_heads"], source["num_key_value_heads"]
    if source["hidden_size"] % heads or heads % kv_heads:
        raise ValueError("LFM2 Audio hidden size and attention head counts are inconsistent")
    if source.get("conv_bias", False):
        raise ValueError("LFM2 Audio backbone currently requires conv_bias=False")

    # Match Liquid's explicit Lfm2Config(**config['lfm']) construction. Keep the
    # complete outer config separately instead of relabelling the audio model.
    normalized = Lfm2Config(**deepcopy(source)).to_dict()
    if not normalized["tie_word_embeddings"]:
        raise ValueError("LFM2 Audio text logits require tied input/output embeddings")
    return normalized


def _expected_backbone_shapes(config: dict, path: Path) -> dict[str, tuple[int, ...]]:
    # Reuse LLiMa's adjustment of the nominal SwiGLU width and attention layout.
    lm = VlmConfig.from_hf_config(ModelFormat.FORMAT_HF, path, config).lm_cfg
    hidden, inner = lm.hidden_size, lm.mlp_cfg.intermediate_size
    heads, kv_heads = lm.attn_cfg.num_attention_heads, lm.attn_cfg.num_key_value_heads
    head_dim = lm.attn_cfg.head_dim
    shapes = {
        "lfm.embed_tokens.weight": (lm.token_cfg.vocab_size, hidden),
        "lfm.embedding_norm.weight": (hidden,),
    }
    for index, layer_type in enumerate(lm.layer_types):
        base = f"lfm.layers.{index}"
        shapes.update({
            f"{base}.operator_norm.weight": (hidden,),
            f"{base}.ffn_norm.weight": (hidden,),
            f"{base}.feed_forward.w1.weight": (inner, hidden),
            f"{base}.feed_forward.w2.weight": (hidden, inner),
            f"{base}.feed_forward.w3.weight": (inner, hidden),
        })
        if layer_type == "conv":
            shapes.update({
                f"{base}.conv.in_proj.weight": (3 * hidden, hidden),
                f"{base}.conv.out_proj.weight": (hidden, hidden),
                f"{base}.conv.conv.weight": (hidden, 1, lm.conv_L_cache),
            })
        else:
            shapes.update({
                f"{base}.self_attn.q_proj.weight": (heads * head_dim, hidden),
                f"{base}.self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
                f"{base}.self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
                f"{base}.self_attn.out_proj.weight": (hidden, heads * head_dim),
                f"{base}.self_attn.q_layernorm.weight": (head_dim,),
                f"{base}.self_attn.k_layernorm.weight": (head_dim,),
            })
    return shapes


class _Lfm2AudioBackbone(LocalHuggingFaceModel):
    """A native language-component view; source tensors remain in their file."""

    def _load_safetensors_tensor(self, safetensors_file: Path, tensor_name: str):
        source_name = "lfm." + tensor_name.removeprefix("model.")
        return super()._load_safetensors_tensor(safetensors_file, source_name)

    def load_all_params(self):
        self.params = None
        self.params = {name: self.load_np_param(name) for name in self.weight_map}

    def execute_hf(self, *args, **kwargs):
        raise NotImplementedError(
            "This is a language-component view of LFM2 Audio. Use the pinned "
            "liquid-audio implementation for complete reference inference."
        )


@dataclass
class _AudioTensorView(_Lfm2AudioBackbone):
    tensor_aliases: dict[str, str] = field(default_factory=dict, kw_only=True)
    rotary_head_dim: int = field(default=0, kw_only=True)
    query_rows: int = field(default=0, kw_only=True)
    key_rows: int = field(default=0, kw_only=True)

    def _load_safetensors_tensor(self, safetensors_file: Path, tensor_name: str):
        value = LocalHuggingFaceModel._load_safetensors_tensor(
            self, safetensors_file, self.tensor_aliases[tensor_name]
        )
        # Liquid's depth transformer rotates adjacent pairs. Native graphs use
        # split-half RoPE. Permute Q/K and their per-head norm scales together;
        # V and the output projection keep their original order.
        dim = self.rotary_head_dim
        if dim and tensor_name.endswith("self_attn.qkv_proj.weight"):
            q, k, v = np.split(value, [self.query_rows, self.query_rows + self.key_rows])
            def reorder(rows):
                return rows.reshape(-1, dim // 2, 2, rows.shape[-1]).transpose(0, 2, 1, 3).reshape(rows.shape)
            value = np.concatenate([reorder(q), reorder(k), v])
        elif dim and tensor_name.endswith(("q_layernorm.weight", "k_layernorm.weight")):
            value = value.reshape(dim // 2, 2).T.reshape(dim)
        return value


def _local_tensor_source(directory: Path, config: dict) -> LocalHuggingFaceModel:
    filename = "model.safetensors"
    with safe_open(directory / filename, framework="numpy") as weights:
        index = dict.fromkeys(weights.keys(), filename)
        metadata = weights.metadata() or {}
    return LocalHuggingFaceModel(
        directory, {filename: directory / filename}, deepcopy(config), index, metadata, {}
    )


@dataclass
class Lfm2AudioCheckpoint:
    """Validated backbone metadata plus the preserved, complete audio config.

    Pass a local snapshot directory containing config.json and model.safetensors.
    Only file headers are read until a tensor is explicitly requested from the
    language_backbone() view. Encoder/depth/decoder semantics are not validated.
    """

    directory: Path
    config: dict
    backbone_config: dict
    tensor_shapes: dict[str, tuple[int, ...]]
    metadata: dict

    @classmethod
    def from_directory(cls, directory: str | Path) -> "Lfm2AudioCheckpoint":
        directory = Path(directory)
        # Do not search recursively: audio_detokenizer has its own config/weights.
        with (directory / "config.json").open(encoding="utf-8") as stream:
            config = json.load(stream)
        if not isinstance(config, dict):
            raise ValueError("LFM2 Audio config.json must contain an object")
        normalized = _backbone_config(config)
        expected = _expected_backbone_shapes(normalized, directory)
        weight_path = directory / "model.safetensors"
        with safe_open(weight_path, framework="numpy") as weights:
            names = {name for name in weights.keys() if name.startswith("lfm.")}
            missing, unexpected = expected.keys() - names, names - expected.keys()
            if missing or unexpected:
                raise ValueError(
                    f"LFM2 Audio backbone tensors differ: missing={sorted(missing)}, "
                    f"unexpected={sorted(unexpected)}"
                )
            for name, shape in expected.items():
                tensor = weights.get_slice(name)
                if tuple(tensor.get_shape()) != shape:
                    raise ValueError(
                        f"LFM2 Audio tensor {name} has shape {tensor.get_shape()}, expected {shape}"
                    )
                if tensor.get_dtype() not in ("F32", "F16", "BF16"):
                    raise ValueError(f"LFM2 Audio tensor {name} must have floating-point weights")
            metadata = weights.metadata() or {}
        return cls(directory, config, normalized, expected, metadata)

    def language_backbone(self) -> LocalHuggingFaceModel:
        """Expose only the backbone under native model.* names, without copies."""
        filename = "model.safetensors"
        return _Lfm2AudioBackbone(
            hf_cache=self.directory,
            weights={filename: self.directory / filename},
            config=deepcopy(self.backbone_config),
            weight_map={
                "model." + name.removeprefix("lfm."): filename
                for name in self.tensor_shapes
            },
            metadata=deepcopy(self.metadata),
            layer_names={
                "model": {"key": "model", "prefix": "layers"},
                "norm": {"key": "model.embedding_norm"},
                "lm_head": {"key": "model.embed_tokens"},
            },
        )

    def audio_source(self) -> LocalHuggingFaceModel:
        """Original names for encoder, projections, and codebook heads."""
        return _local_tensor_source(self.directory, self.config)

    def depth_backbone(self) -> LocalHuggingFaceModel:
        """Normalize depth names and rotary layout for shared graph primitives."""
        source = self.audio_source()
        dim = _positive_integer(self.config["depthformer"], "dim")
        layers = _positive_integer(self.config["depthformer"], "layers")
        if dim % 32:
            raise ValueError("LFM2 Audio depth width must be divisible by its 32 heads")
        aliases = {}
        for index in range(layers):
            native, original = f"model.layers.{index}", f"depthformer.layers.{index}"
            for name in ("operator_norm.weight", "ffn_norm.weight", "feed_forward.w1.weight",
                         "feed_forward.w2.weight", "feed_forward.w3.weight"):
                aliases[f"{native}.{name}"] = f"{original}.{name}"
            for name in ("qkv_proj.weight", "out_proj.weight"):
                aliases[f"{native}.self_attn.{name}"] = f"{original}.operator.{name}"
            for name in ("q_layernorm.weight", "k_layernorm.weight"):
                aliases[f"{native}.self_attn.{name}"] = f"{original}.operator.bounded_attention.{name}"
        for name in aliases.values():
            if not source.param_exists(name):
                raise ValueError(f"Missing LFM2 Audio depth tensor {name}")
        return _AudioTensorView(
            self.directory, source.weights, deepcopy(self.config["depthformer"]),
            {name: source.weight_map[original] for name, original in aliases.items()},
            source.metadata, {"model": {"key": "model", "prefix": "layers"}},
            tensor_aliases=aliases, rotary_head_dim=dim // 32,
            query_rows=dim, key_rows=8 * (dim // 32),
        )

    def detokenizer_backbone(self) -> LocalHuggingFaceModel:
        directory = self.directory / "audio_detokenizer"
        with (directory / "config.json").open(encoding="utf-8") as stream:
            config = json.load(stream)
        if config.get("model_type") != "lfm2":
            raise ValueError("Expected an LFM2 audio detokenizer")
        source = _local_tensor_source(directory, config)
        aliases = {
            "model." + name.removeprefix("lfm."): name
            for name in source.weight_map if name.startswith("lfm.")
        }
        normalized = Lfm2Config(**deepcopy(config)).to_dict()
        normalized["architectures"] = ["Lfm2ForCausalLM"]
        return _AudioTensorView(
            directory, source.weights, normalized,
            {name: source.weight_map[original] for name, original in aliases.items()},
            source.metadata, {"model": {"key": "model", "prefix": "layers"}},
            tensor_aliases=aliases,
        )

    def detokenizer_source(self) -> LocalHuggingFaceModel:
        directory = self.directory / "audio_detokenizer"
        with (directory / "config.json").open(encoding="utf-8") as stream:
            return _local_tensor_source(directory, json.load(stream))
