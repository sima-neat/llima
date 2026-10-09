from dataclasses import dataclass
from typing import ClassVar

from sima_lmm.model.model_graph import ModelGraph
from sima_lmm.model.base import BaseModel, LayerConfiguration


@dataclass
class WhisperDecoderPreModel(BaseModel):
    """Implementation for the pre cache model of Whisper's decoder.

    This implements a simplified version of the LanguagePreModel. This model is only used when
    generating new tokens so the num_tokens is assumed to be 1.

    Attributes:
        num_tokens: Number of tokens. Set to a value greater than 1 to consume multiple input tokens
            in one model.
        layer_idx: Transformer layer index.
    """
    num_tokens: int
    layer_idx: int
    positioned_residual_output_idx: ClassVar[int] = 3

    def __post_init__(self):
        assert 0 <= self.layer_idx < self.cfg.decoder_layers

    @property
    def enable_filter_sharing(self) -> bool:
        return self.use_filter_sharing

    def generate_graph(
        self, layer_cfg: LayerConfiguration, quantizable: bool
    ):
        shape = (1, 1, self.num_tokens, self.cfg.d_model)
        shapes = {"input": shape}
        if self.layer_idx == 0:
            shapes["embed_positions"] = shape

        graph = ModelGraph(self, shapes, quantizable)
        outputs = self._build_nodes(graph, list(graph.inputs.values()))
        graph.save(outputs)

    def _build_nodes(self, graph, inputs):
        name = f"model.decoder.layers.{self.layer_idx}"
        residual = graph.add(*inputs) if self.layer_idx == 0 else inputs[0]
        norm = graph.layer_norm(f"{name}.self_attn_layer_norm", residual)
        query, key, value = [
            graph.linear(f"{name}.self_attn.{proj}_proj", norm) for proj in ("q", "k", "v")
        ]
        query = graph.mul(query, graph.constant(self.cfg.decoder_head_dim**-0.5))
        query = graph.split_heads(query, self.cfg.decoder_attention_heads)
        outputs = [query, key, value]
        if self.layer_idx == 0:
            outputs.append(residual)
        return outputs
