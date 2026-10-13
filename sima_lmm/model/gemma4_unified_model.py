"""Gemma 4 Unified's encoder-free image projection."""
from dataclasses import dataclass

import numpy as np
from ml_dtypes import bfloat16

from sima_lmm.model.base import BaseModel, LayerConfiguration
from sima_lmm.model.model_graph import ModelGraph


@dataclass
class Gemma4UnifiedImageModel(BaseModel):
    layer_idx: int = 0

    def generate_graph(self, layer_cfg: LayerConfiguration, quantizable: bool):
        vm = self.cfg.vm_cfg
        rows = self.cfg.pipeline_cfg.input_token_group_size
        patch_dim = 3 * (vm.patch_size * vm.spatial_merge_size) ** 2
        graph = ModelGraph(self, {
            "patches": (1, 1, rows, patch_dim),
            "positions": (1, 1, rows, vm.hidden_size),
        }, quantizable)
        base = self.position_weight_name.removesuffix(".pos_embedding")
        hidden = graph.layer_norm(f"{base}.patch_ln1", graph.inputs["patches"])
        hidden = graph.linear(f"{base}.patch_dense", hidden)
        hidden = graph.layer_norm(f"{base}.patch_ln2", hidden)
        hidden = graph.add(hidden, graph.inputs["positions"])
        hidden = graph.layer_norm(f"{base}.pos_norm", hidden)
        hidden = graph.rms_norm(None, hidden, epsilon=1e-6)
        projection = next(name for name in (
            "model.embed_vision.embedding_projection",
            "model.embed_vision.multimodal_embedder.embedding_projection",
        ) if self.check_hf_param(f"{name}.weight"))
        output = graph.linear(projection, hidden)
        graph.save(list(graph.quant(output)) if self.cfg.pipeline_cfg.quantize_embeddings else [output])

    @property
    def position_weight_name(self) -> str:
        return next(name for name in (
            "model.vision_embedder.pos_embedding", "model.embed_vision.pos_embedding"
        ) if self.check_hf_param(name))

    def position_embeddings(self, positions: np.ndarray) -> np.ndarray:
        """Gather factorized XY positions, rounding the sum as in BF16 reference."""
        table = self.get_hf_param(self.position_weight_name)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("Image positions must have shape [tokens, 2]")
        if np.any(positions < 0) or np.any(positions >= table.shape[0]):
            raise ValueError("Image position is outside the factorized position table")
        return (table[positions[:, 0], 0].astype(np.float32)
                + table[positions[:, 1], 1].astype(np.float32)).astype(bfloat16).astype(np.float32)


@dataclass
class Gemma4UnifiedAudioModel(BaseModel):
    """Project fixed waveform frames into the shared language decoder."""
    layer_idx: int = 0

    def generate_graph(self, layer_cfg: LayerConfiguration, quantizable: bool):
        audio = self.cfg.audio_cfg
        rows = self.cfg.pipeline_cfg.input_token_group_size
        graph = ModelGraph(self, {"waveform": (1, 1, rows, audio.samples_per_token)}, quantizable)
        hidden = graph.rms_norm(None, graph.inputs["waveform"], epsilon=audio.rms_norm_eps)
        output = graph.linear("model.embed_audio.embedding_projection", hidden)
        graph.save(list(graph.quant(output)) if self.cfg.pipeline_cfg.quantize_embeddings else [output])

    def run_model(self, eval_mode, ifms):
        frames = ifms[0].reshape(-1, self.cfg.audio_cfg.samples_per_token)
        if not 0 < len(frames) <= self.cfg.audio_cfg.max_tokens:
            raise ValueError("Audio is empty or exceeds the compiled token budget")
        rows = self.cfg.pipeline_cfg.input_token_group_size
        outputs = []
        for begin in range(0, len(frames), rows):
            valid = min(rows, len(frames) - begin)
            chunk = np.pad(frames[begin:begin + valid], ((0, rows - valid), (0, 0)))[None, None]
            outputs.append([output[..., :valid, :] for output in super().run_model(eval_mode, [chunk])])
        return [np.concatenate(parts, axis=2) for parts in zip(*outputs, strict=True)]
