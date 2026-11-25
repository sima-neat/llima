"""Native compute components for LFM2 Audio.

Components expose dense hidden states. Audio scheduling and token lookup are
separate from the compiler-only graph builders in this module.
"""

from dataclasses import dataclass

import numpy as np

from sima_lmm.model.base import BaseModel
from sima_lmm.model.language_cache_model import LanguageCacheModel
from sima_lmm.model.language_conv_model import LanguageConvModel
from sima_lmm.model.language_part_base import LanguagePostBaseModel
from sima_lmm.model.language_post_model import LanguagePostModel
from sima_lmm.model.model_graph import ModelGraph


class AudioCacheModel(LanguageCacheModel):
    """A fixed-capacity cache with an explicit mask at every query length."""

    @property
    def _uses_group_future_token_mask(self):
        return self.num_tokens > 1


class AudioLanguagePostModel(LanguagePostModel):
    """Return the normalized backbone hidden state for text and audio heads."""

    def _build_post_transformer(self, graph, input_node):
        prefix = self.hf_model.language_model_param_base_name
        return [graph.rms_norm(f"{prefix}.embedding_norm", input_node)]


class AudioBackboneConvModel(LanguageConvModel):
    """Preserve backbone MLP activation rounding with 32-token regions."""
    _mlp_activation_token_block_size = 32


class AudioBackbonePostModel(AudioLanguagePostModel):
    _mlp_activation_token_block_size = 32


class AudioDepthPostModel(LanguagePostModel):
    """The depth backbone has no final norm or shared vocabulary head."""

    def _build_post_transformer(self, graph, input_node):
        return [input_node]


@dataclass
class AudioFinalConvModel(LanguagePostBaseModel):
    """Final convolution FFN and normalization for every input timestep."""

    def generate_graph(self, layer_cfg, quantizable):
        prefix = self.hf_model.language_model_param_base_name
        base = f"{prefix}.layers.{self.layer_idx}"
        graph = ModelGraph(
            self, {"input": (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)}, quantizable
        )
        residual = graph.inputs["input"]
        norm = graph.rms_norm(f"{base}.ffn_norm", residual)
        mlp = self._build_mlp(graph, f"{base}.feed_forward", [norm])
        hidden = graph.add(residual, mlp)
        graph.save([graph.rms_norm(f"{prefix}.embedding_norm", hidden)])


class AudioBackboneFinalConvModel(AudioFinalConvModel):
    _mlp_activation_token_block_size = 32


@dataclass
class AudioProjectionModel(BaseModel):
    num_tokens: int
    input_dim: int
    weight_name: str
    norm_name: str | None = None
    norm_epsilon: float = 1e-6
    output_begin: int = 0
    output_end: int | None = None

    def generate_graph(self, layer_cfg, quantizable):
        graph = ModelGraph(self, {"hidden": (1, 1, self.num_tokens, self.input_dim)}, quantizable)
        hidden = graph.inputs["hidden"]
        if self.norm_name is not None:
            hidden = graph.rms_norm(self.norm_name, hidden, epsilon=self.norm_epsilon)
        options = {
            "src_weight_name": self.weight_name,
            "src_bias_name": self.weight_name.removesuffix(".weight") + ".bias",
        }
        if self.output_begin or self.output_end is not None:
            begin, end = self.output_begin, self.output_end
            options["weight_process_func"] = lambda value: value[begin:end]
            options["bias_process_func"] = lambda value: value[begin:end]
        graph.save([graph.linear("projection", hidden, **options)])


@dataclass
class AudioAdapterModel(BaseModel):
    num_tokens: int
    input_dim: int

    def generate_graph(self, layer_cfg, quantizable):
        graph = ModelGraph(self, {"hidden": (1, 1, self.num_tokens, self.input_dim)}, quantizable)
        hidden = graph.layer_norm("audio_adapter.model.0", graph.inputs["hidden"])
        hidden = graph.activation(graph.linear("audio_adapter.model.1", hidden), "gelu")
        graph.save([graph.linear("audio_adapter.model.3", hidden)])


@dataclass
class AudioEncoderSubsamplingModel(BaseModel):
    input_frames: int
    feature_size: int = 128
    channels: int = 256

    def generate_graph(self, layer_cfg, quantizable):
        stages = [(self.input_frames + divisor - 1) // divisor for divisor in (2, 4, 8)]
        graph = ModelGraph(self, {
            "features": (1, 1, self.input_frames, self.feature_size),
            **{f"stage{index + 1}_valid": (1, length, 1, 1) for index, length in enumerate(stages)},
        }, quantizable)
        hidden = graph.reshape(graph.inputs["features"], (1, self.input_frames, self.feature_size, 1))
        base = "conformer.pre_encode"
        hidden = graph.relu(graph.conv(
            f"{base}.conv.0", hidden, stride=(2, 2), padding=((1, 1), (1, 1))
        ))
        hidden = graph.mul(hidden, graph.inputs["stage1_valid"])
        for stage, (depthwise, pointwise) in enumerate(((2, 3), (5, 6)), start=2):
            hidden = graph.conv(
                f"{base}.conv.{depthwise}", hidden, stride=(2, 2),
                padding=((1, 1), (1, 1)), is_depthwise=True,
                weight_process_func=lambda value: value.transpose(0, 2, 1, 3),
            )
            hidden = graph.relu(graph.conv(f"{base}.conv.{pointwise}", hidden))
            hidden = graph.mul(hidden, graph.inputs[f"stage{stage}_valid"])
        frames, features = (self.input_frames + 7) // 8, (self.feature_size + 7) // 8
        # PyTorch flattens [channels, frequency] after moving time to axis one.
        hidden = graph.transpose(hidden, (0, 1, 3, 2))
        hidden = graph.reshape(hidden, (1, 1, frames, self.channels * features))
        graph.save([graph.linear(f"{base}.out", hidden)])


def relative_position_table(length: int, width: int) -> np.ndarray:
    positions = np.arange(length - 1, -length, -1, dtype=np.float32)[:, None]
    frequencies = np.exp(np.arange(0, width, 2, dtype=np.float32) * np.float32(-np.log(10000.) / width))
    table = np.empty((2 * length - 1, width), dtype=np.float32)
    table[:, 0::2] = np.sin(positions * frequencies)
    table[:, 1::2] = np.cos(positions * frequencies)
    return table[None, None]


def _conformer_relative_shift(graph: ModelGraph, values, length: int, heads: int):
    """Select relative-position scores through aligned copies, without Conv."""
    width = 16
    rounded = ((length + width - 1) // width) * width
    channels = 2 * rounded + width
    zeros = graph.constant(np.zeros((1, heads, length, channels - 2 * length), np.float32))
    padded = graph.concat([values, zeros], 3)
    blocks = channels // width
    flattened = graph.reshape(padded, (1, heads, length * blocks, width))
    groups = {}
    for row in range(length):
        begin = length - row
        start = row * blocks + begin // width
        cropped = graph.create_slice_node(
            flattened, [start], [start + rounded // width + 1], [1], [2]
        )
        groups.setdefault(begin % width, []).append((row, cropped))
    rows = [None] * length
    for offset, entries in sorted(groups.items()):
        # Packing queries into height keeps each head independent. Selecting
        # between adjacent 16-channel blocks bounds the native channel shuffle
        # to 32 channels, instead of shuffling the complete score tensor.
        combined = graph.concat([cropped for _, cropped in entries], 1)
        count = rounded // width
        left = graph.create_slice_node(combined, [0], [count], [1], [2])
        right = graph.create_slice_node(combined, [1], [count + 1], [1], [2])
        paired = graph.concat([left, right], 3)
        selected = graph.create_slice_node(paired, [offset], [offset + width], [1], [3])
        selected = graph.reshape(selected, (1, heads * len(entries), 1, rounded))
        if rounded != length:
            selected = graph.create_slice_node(selected, [0], [length], [1], [3])
        for index, (row, _) in enumerate(entries):
            rows[row] = graph.create_slice_node(
                selected, [index * heads], [(index + 1) * heads], [1], [1]
            )
    return graph.concat(rows, 2)


@dataclass
class AudioConformerLayerModel(BaseModel):
    num_tokens: int
    layer_idx: int
    width: int = 512
    heads: int = 8
    kernel_size: int = 9

    def generate_graph(self, layer_cfg, quantizable):
        length, width, heads = self.num_tokens, self.width, self.heads
        graph = ModelGraph(self, {
            "hidden": (1, 1, length, width),
            "attention_valid": (1, 1, length, length),
            "tokens_valid": (1, 1, length, 1),
        }, quantizable)
        base = f"conformer.layers.{self.layer_idx}"

        def feed_forward(name, hidden):
            hidden = graph.activation(graph.linear(f"{base}.{name}.linear1", hidden), "silu")
            return graph.linear(f"{base}.{name}.linear2", hidden)

        residual = graph.inputs["hidden"]
        hidden = graph.layer_norm(f"{base}.norm_feed_forward1", residual)
        residual = graph.add(residual, graph.mul(feed_forward("feed_forward1", hidden), graph.constant([0.5])))
        hidden = graph.layer_norm(f"{base}.norm_self_att", residual)
        attn = f"{base}.self_attn"
        q, k, v = [graph.split_heads(graph.linear(f"{attn}.linear_{name}", hidden), heads) for name in ("q", "k", "v")]
        # This table and its bias-free projection are independent of all
        # inputs. Fold them together, avoiding a constant-only convolution
        # region for which the compiler has no predecessor layout.
        position = np.concatenate([
            np.zeros((1, 1, 1, width), np.float32), relative_position_table(length, width),
        ], axis=2)
        position = position @ self.get_hf_param(f"{attn}.linear_pos.weight").astype(np.float32).T
        position = graph.constant(position.reshape(1, 2 * length, heads, width // heads).transpose(0, 2, 1, 3).copy())
        bias_u = self.get_hf_param(f"{attn}.pos_bias_u").astype(np.float32).reshape(1, heads, 1, width // heads)
        bias_v = self.get_hf_param(f"{attn}.pos_bias_v").astype(np.float32).reshape(1, heads, 1, width // heads)
        ac = graph.matmul(graph.add(q, graph.constant(bias_u)), k, transpose_b=True)
        bd = graph.matmul(graph.add(q, graph.constant(bias_v)), position, transpose_b=True)
        # rel_shift(q, k) is exactly padded_bd[q, T-q+k]. Select each row's
        # contiguous interval directly, avoiding an unaligned reshape to 2*T-1.
        if length > 32:
            bd = _conformer_relative_shift(graph, bd, length, heads)
        else:
            shifted = []
            for row in range(length):
                values = graph.slice(bd, start=row, stop=row + 1, axis=2)
                shifted.append(graph.slice(values, start=length - row, stop=2 * length - row, axis=3))
            bd = graph.concat(shifted, 2)
        scores = graph.mul(graph.add(ac, bd), graph.constant([(width // heads) ** -0.5]))
        valid = graph.inputs["attention_valid"]
        invalid = graph.add(graph.mul(valid, graph.constant([-1.])), graph.constant([1.]))
        scores = graph.add(graph.mul(scores, valid), graph.mul(
            invalid, graph.constant([-10000.])
        ))
        probabilities = graph.mul(graph.softmax(scores), valid)
        hidden = graph.linear(f"{attn}.linear_out", graph.merge_heads(graph.matmul(probabilities, v)))
        residual = graph.add(residual, hidden)

        hidden = graph.layer_norm(f"{base}.norm_conv", residual)
        hidden = graph.conv(f"{base}.conv.pointwise_conv1", hidden)
        gate = graph.slice(hidden, start=width, stop=2 * width, axis=3)
        hidden = graph.mul(graph.slice(hidden, start=0, stop=width, axis=3), graph.sigmoid(gate))
        hidden = graph.mul(hidden, graph.inputs["tokens_valid"])
        padding = self.kernel_size // 2
        hidden = graph.conv(
            f"{base}.conv.depthwise_conv", hidden,
            padding=((0, 0), (padding, padding)), is_depthwise=True,
        )
        bn = f"{base}.conv.batch_norm"
        scale = self.get_hf_param(f"{bn}.weight").astype(np.float32)
        mean = self.get_hf_param(f"{bn}.running_mean").astype(np.float32)
        variance = self.get_hf_param(f"{bn}.running_var").astype(np.float32)
        bias = self.get_hf_param(f"{bn}.bias").astype(np.float32)
        scale = scale / np.sqrt(variance + np.float32(1e-5))
        # Express inference BatchNorm as a per-channel 1x1 affine convolution.
        # The current MLA constant-vector Mul lowering produces incorrect
        # channel scales here; the depthwise affine path matches SDK execution.
        hidden = graph.conv(
            bn, hidden, is_depthwise=True,
            weight_process_func=lambda value: scale.reshape(width, 1, 1, 1),
            bias_process_func=lambda value: bias - mean * scale,
        )
        hidden = graph.conv(f"{base}.conv.pointwise_conv2", graph.activation(hidden, "silu"))
        residual = graph.add(residual, hidden)
        hidden = graph.layer_norm(f"{base}.norm_feed_forward2", residual)
        hidden = graph.add(residual, graph.mul(feed_forward("feed_forward2", hidden), graph.constant([0.5])))
        graph.save([graph.layer_norm(f"{base}.norm_out", hidden)])
