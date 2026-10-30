import numpy as np

from dataclasses import dataclass

from sima_lmm.model.model_graph import ModelGraph
from sima_lmm.model.base import BaseModel, LayerConfiguration


@dataclass
class WhisperDecoderCacheModel(BaseModel):
    """Implementation for the cache model of Whisper.

    This implements a simplified version of the LanguageCacheModel. This model is only used when
    generating new tokens so the num_tokens is assumed to be 1.

    Attributes:
        num_tokens: Number of tokens. Set to a value greater than 1 to consume multiple input tokens
            in one model.
        token_idx: Token index.
    """
    num_tokens: int
    token_idx: int
    use_future_token_mask: bool

    def __post_init__(self):
        assert self.token_idx >= 0

    def generate_graph(
        self, layer_cfg: LayerConfiguration, quantizable: bool
    ):
        shapes = {
            "input": (1, self.cfg.decoder_attention_heads, self.num_tokens, self.cfg.decoder_head_dim),
            "cached_keys": (1, 1, self.token_idx + self.num_tokens, self.cfg.d_model),
            "cached_values": (1, 1, self.token_idx + self.num_tokens, self.cfg.d_model),
        }
        if self.use_future_token_mask and self.num_tokens == 1:
            shapes["attn_mask"] = (1, 1, 1, self.token_idx + 1)

        graph = ModelGraph(self, shapes, quantizable)
        outputs = self._build_nodes(graph, list(graph.inputs.values()))
        graph.save(outputs)

    def _build_nodes(self, graph, inputs):
        key, value = [graph.split_heads(node, self.cfg.decoder_attention_heads) for node in inputs[1:3]]
        mask = None
        if self.num_tokens > 1:
            mask = np.zeros((1, 1, self.num_tokens, self.token_idx + self.num_tokens), np.float32)
            for i in range(self.num_tokens):
                mask[:, :, i, self.token_idx + i + 1 :] = np.finfo(np.float32).min
            mask = graph.constant(mask)
        elif self.use_future_token_mask:
            mask = inputs[3]
        attn = graph.attention(inputs[0], key, value, mask=mask)
        return [graph.merge_heads(attn)]
