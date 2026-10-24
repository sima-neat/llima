"""Layered Gemma 4 audio encoder, using the same component boundary as vision."""

from dataclasses import dataclass
import math

import numpy as np

from sima_lmm.model.base import BaseModel, EvalMode, GenConfiguration, FileGenMode, LayerConfiguration
from sima_lmm.model.model_graph import ModelGraph, Node


_ATTENTION_MASK_VALUE = -1e9
_FFN_RESIDUAL_SCALE = 0.5
_ACTIVATION_CLIP = 1e10


@dataclass
class AudioModel(BaseModel):
    enable_filter_sharing = False

    def _get_part_model(self, layer_idx: int) -> "Gemma4AudioLayerModel":
        if self.cfg.audio_cfg is None or not 0 <= layer_idx < self.cfg.audio_cfg.num_hidden_layers:
            raise ValueError(f"Invalid audio encoder layer index {layer_idx}")
        return Gemma4AudioLayerModel(
            self.cfg, self.cfg.get_audio_model_names(self.model_name)[layer_idx],
            sima_path=self.sima_path, hf_model=self.hf_model, layer_idx=layer_idx,
        )

    def gen_files(
        self, gen_mode: FileGenMode, *, gen_config: GenConfiguration,
        log_level: int = 0, num_processes: int = 1, resume: bool = False,
    ):
        models = [
            (self._get_part_model(layer.part_idx), {"precision": precision})
            for layer, precision in gen_config["precision"].items() if layer.part == "audio"
        ]
        self.gen_files_from_model_list(models, gen_mode, num_processes, log_level, resume)

    def run_model(self, eval_mode: EvalMode, ifms: list[np.ndarray]) -> list[np.ndarray]:
        hidden, mask = ifms
        for layer_idx in range(self.cfg.audio_cfg.num_hidden_layers):
            outputs = self._get_part_model(layer_idx).run_model(eval_mode, [hidden, mask])
            hidden = outputs[0]
        return outputs


@dataclass
class Gemma4AudioLayerModel(BaseModel):
    """One Conformer layer, with the stem/final projector fused at the boundaries."""

    layer_idx: int
    enable_filter_sharing = False

    def generate_graph(self, layer_cfg: LayerConfiguration, quantizable: bool):
        audio = self.cfg.audio_cfg
        # The supported stem normalizes the mel axis to its first conv width.
        mel_bins = audio.subsampling_conv_channels[0]
        shape = ((1, 1, audio.num_frames, mel_bins) if self.layer_idx == 0
                 else (1, 1, audio.padded_tokens, audio.hidden_size))
        graph = ModelGraph(self, {"hidden": shape, "mask": (1, 1, audio.num_frames, 1)}, quantizable)
        hidden, mask = graph.inputs["hidden"], graph.inputs["mask"]
        if self.layer_idx == 0:
            hidden = graph.reshape(hidden, [1, audio.num_frames, mel_bins, 1])
            for i, channels in enumerate(audio.subsampling_conv_channels):
                frames = audio.num_frames // (2 ** i)
                conv_mask = graph.slice(mask, begin=[0], end=[audio.num_frames],
                                        stride=[2 ** i], axis=[2])
                conv_mask = graph.reshape(conv_mask, [1, frames, 1, 1])
                hidden = graph.mul(hidden, conv_mask)
                base = f"model.audio_tower.subsample_conv_projection.layer{i}"
                hidden = graph.conv(f"{base}.conv", hidden, stride=(2, 2), padding=((1, 1), (1, 1)))
                hidden = graph.layer_norm(f"{base}.norm", hidden, epsilon=audio.rms_norm_eps)
                hidden = graph.relu(hidden)
            hidden = graph.reshape(hidden, [1, 1, audio.max_tokens,
                                            (mel_bins // 4) * channels])
            hidden = graph.linear("model.audio_tower.subsample_conv_projection.input_proj_linear", hidden)
            if audio.padded_tokens > audio.max_tokens:
                hidden = graph.concat([hidden, graph.constant(np.zeros(
                    (1, 1, audio.padded_tokens - audio.max_tokens, audio.hidden_size), np.float32))], axis=2)

        base = f"model.audio_tower.layers.{self.layer_idx}"
        hidden = self._feed_forward(graph, f"{base}.feed_forward1", hidden)
        residual = hidden
        hidden = graph.clip(hidden, -_ACTIVATION_CLIP, _ACTIVATION_CLIP)
        hidden = graph.rms_norm(f"{base}.norm_pre_attn", hidden, epsilon=1e-6)
        hidden = self._attention(graph, f"{base}.self_attn", hidden, mask)
        hidden = graph.clip(hidden, -_ACTIVATION_CLIP, _ACTIVATION_CLIP)
        hidden = graph.rms_norm(f"{base}.norm_post_attn", hidden, epsilon=1e-6)
        hidden = graph.add(residual, hidden)

        residual = hidden
        conv = f"{base}.lconv1d"
        hidden = graph.rms_norm(f"{conv}.pre_layer_norm", hidden, epsilon=audio.rms_norm_eps)
        hidden = self._linear(graph, f"{conv}.linear_start", hidden)
        first = graph.slice(hidden, start=0, stop=audio.hidden_size, axis=3)
        gate = graph.slice(hidden, start=audio.hidden_size, stop=2 * audio.hidden_size, axis=3)
        hidden = graph.mul(first, graph.sigmoid(gate))
        hidden = graph.conv(f"{conv}.depthwise_conv1d", hidden, is_depthwise=True,
                            padding=((0, 0), (audio.conv_kernel_size - 1, 0)))
        hidden = graph.clip(hidden, -_ACTIVATION_CLIP, _ACTIVATION_CLIP)
        hidden = graph.rms_norm(f"{conv}.conv_norm", hidden, epsilon=audio.rms_norm_eps)
        hidden = graph.activation(hidden, audio.hidden_act)
        hidden = graph.add(residual, self._linear(graph, f"{conv}.linear_end", hidden))
        hidden = self._feed_forward(graph, f"{base}.feed_forward2", hidden)
        hidden = graph.clip(hidden, -_ACTIVATION_CLIP, _ACTIVATION_CLIP)
        hidden = graph.rms_norm(f"{base}.norm_out", hidden, epsilon=1e-6)
        if self.layer_idx == audio.num_hidden_layers - 1:
            hidden = graph.slice(hidden, start=0, stop=audio.max_tokens, axis=2)
            hidden = graph.linear("model.audio_tower.output_proj", hidden)
            hidden = graph.rms_norm(None, hidden, epsilon=audio.rms_norm_eps)
            hidden = graph.linear("model.embed_audio.embedding_projection", hidden)
        outputs = (list(graph.quant(hidden)) if self.layer_idx == audio.num_hidden_layers - 1
                   and self.cfg.pipeline_cfg.quantize_embeddings else [hidden])
        graph.save(outputs)

    def _linear(self, graph: ModelGraph, name: str, hidden: Node) -> Node:
        """Apply the checkpoint's input/output clipping around a nested Linear."""
        def clip(hidden: Node, side: str) -> Node:
            if self.check_hf_param(f"{name}.{side}_min"):
                low = float(self.get_hf_param(f"{name}.{side}_min"))
                high = float(self.get_hf_param(f"{name}.{side}_max"))
                if np.isfinite(low) and np.isfinite(high):
                    return graph.clip(hidden, low, high)
            return hidden

        hidden = clip(hidden, "input")
        hidden = graph.linear(name, hidden, src_weight_name=f"{name}.linear.weight",
                              src_bias_name=f"{name}.linear.bias")
        return clip(hidden, "output")

    def _feed_forward(self, graph: ModelGraph, name: str, hidden: Node) -> Node:
        audio = self.cfg.audio_cfg
        residual = hidden
        hidden = graph.clip(hidden, -_ACTIVATION_CLIP, _ACTIVATION_CLIP)
        hidden = graph.rms_norm(f"{name}.pre_layer_norm", hidden, epsilon=1e-6)
        hidden = self._linear(graph, f"{name}.ffw_layer_1", hidden)
        hidden = graph.activation(hidden, audio.hidden_act)
        hidden = self._linear(graph, f"{name}.ffw_layer_2", hidden)
        hidden = graph.clip(hidden, -_ACTIVATION_CLIP, _ACTIVATION_CLIP)
        hidden = graph.rms_norm(f"{name}.post_layer_norm", hidden, epsilon=1e-6)
        return graph.add(residual, graph.mul(hidden, graph.constant([_FFN_RESIDUAL_SCALE])))

    def _attention(self, graph: ModelGraph, name: str, hidden: Node, mask: Node) -> Node:
        """Fold heads and local blocks into rank-four MLA batch matmuls."""
        audio = self.cfg.audio_cfg
        rows, chunk, heads = audio.padded_tokens, audio.attention_chunk_size, audio.num_attention_heads
        blocks, dim = rows // chunk, audio.hidden_size // heads
        q, k, v = [graph.split_heads(self._linear(graph, f"{name}.{proj}_proj", hidden), heads)
                   for proj in ("q", "k", "v")]
        scale = np.logaddexp(0, self.get_hf_param(f"{name}.per_dim_scale").astype(np.float32))
        q = graph.mul(q, graph.constant((scale * (dim ** -0.5 / math.log(2)))[None, None, None]))
        k = graph.mul(k, graph.constant([math.log1p(math.e) / math.log(2)]))

        def contexts(data: Node, count: int, width: int) -> Node:
            data = graph.concat([graph.constant(np.zeros((1, count, chunk, width), np.float32)), data], axis=2)
            parts = [graph.reshape(graph.slice(data, start=start, stop=start + rows, axis=2),
                                   [1, count * blocks, chunk, width]) for start in (0, chunk)]
            return graph.concat(parts, axis=2)

        keys, values = contexts(k, heads, dim), contexts(v, heads, dim)
        queries = graph.reshape(q, [1, heads * blocks, chunk, dim])
        scores = graph.matmul(queries, keys, transpose_b=True)

        inv = np.exp(np.arange(audio.hidden_size // 2, dtype=np.float32)
                     * -(math.log(10000) / max(audio.hidden_size // 2 - 1, 1)))
        times = np.arange(chunk, -1, -1, dtype=np.float32)[:, None] * inv
        positions = np.concatenate([np.sin(times), np.cos(times)], axis=-1)[None, None].astype(graph.dtype)
        # This projection depends only on source weights and fixed positions.
        # Evaluate through ModelGraph to preserve packed-weight/scaling semantics;
        # a constant-only convolution has no runtime input for the MLA planner.
        relative_graph = ModelGraph(self, {"positions": positions.shape}, graph.quantizable)
        relative_keys = relative_graph.split_heads(
            relative_graph.linear(f"{name}.relative_k_proj", relative_graph.inputs["positions"]), heads)
        relative_graph.finish([relative_keys])
        relative_keys = graph.constant(relative_graph.run(positions=positions)[0])
        relative = graph.matmul(q, relative_keys, transpose_b=True)
        relative = graph.concat([relative, graph.constant(np.zeros((1, heads, rows, chunk), np.float32))], axis=3)
        relative = graph.reshape(relative, [1, heads * blocks, 1, chunk * (2 * chunk + 1)])
        relative = graph.slice(relative, start=0, stop=chunk * 2 * chunk, axis=3)
        relative = graph.reshape(relative, [1, heads * blocks, chunk, 2 * chunk])
        scores = graph.softcap(graph.add(scores, relative), audio.attention_logit_cap)

        valid = graph.slice(mask, begin=[0], end=[audio.num_frames], stride=[4], axis=[2])
        if rows > audio.max_tokens:
            valid = graph.concat([valid, graph.constant(np.zeros((1, 1, rows - audio.max_tokens, 1), np.float32))], axis=2)
        valid = graph.transpose(contexts(valid, 1, 1), [0, 1, 3, 2])
        valid = graph.concat([valid] * heads, axis=1)
        qpos, kpos = np.arange(chunk)[:, None], np.arange(2 * chunk)[None]
        causal = graph.constant(((kpos >= qpos) & (kpos <= qpos + chunk)).astype(np.float32)[None, None])
        valid = graph.mul(valid, causal)
        penalty = graph.mul(graph.sub(graph.constant(np.ones((1, 1, 1, 1), np.float32)), valid),
                            graph.constant([_ATTENTION_MASK_VALUE]))
        probabilities = graph.softmax(graph.add(scores, penalty))
        output = graph.matmul(probabilities, values)
        output = graph.reshape(output, [1, heads, rows, dim])
        return self._linear(graph, f"{name}.post", graph.merge_heads(output))
