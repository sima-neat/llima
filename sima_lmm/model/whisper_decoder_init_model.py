
from dataclasses import dataclass
from typing import ClassVar

from sima_lmm.model.model_graph import ModelGraph
from sima_lmm.model.base import BaseModel, LayerConfiguration
from sima_lmm.model.whisper_decoder_cache_model import WhisperDecoderCacheModel
from sima_lmm.model.whisper_decoder_post_model import WhisperDecoderPostModel
from sima_lmm.model.whisper_decoder_pre_model import WhisperDecoderPreModel


@dataclass
class WhisperDecoderInitModel(BaseModel):
    """Implementation Whisper's decoder layers to process the input tokens and encoder outputs.

    Attributes:
        layer_idx: Transformer layer index.
        enable_log_probe: Whether to also output filtered logits from the final layer.
    """
    layer_idx: int
    enable_log_probe: bool = False
    # 4 init tokens used in the init model:
    #   1. <|startoftranscript|>,
    #   2. `language`,
    #   3. <|transcribe|>
    #   4. <|notimestamps|>
    num_tokens: ClassVar[int] = 4
    token_idx: ClassVar[int] = 0

    def __post_init__(self):
        assert 0 <= self.layer_idx < self.cfg.decoder_layers

    @property
    def enable_filter_sharing(self) -> bool:
        return self.use_filter_sharing

    def generate_graph(
        self, layer_cfg: LayerConfiguration, quantizable: bool
    ):
        shapes = {
            "input": (1, 1, self.num_tokens, self.cfg.d_model),
            "audio_features": (1, 1, self.cfg.max_source_positions, self.cfg.d_model),
        }

        graph = ModelGraph(self, shapes, quantizable)
        inputs = list(graph.inputs.values())
        pre = WhisperDecoderPreModel(
            self.cfg,
            self.model_name,
            hf_model=self.hf_model,
            num_tokens=self.num_tokens,
            layer_idx=self.layer_idx,
        )
        final_layer = self.layer_idx == self.cfg.decoder_layers - 1
        num_tokens = 1 if final_layer else self.num_tokens
        cache = WhisperDecoderCacheModel(
            self.cfg,
            self.model_name,
            num_tokens=num_tokens,
            token_idx=self.num_tokens - num_tokens,
            use_future_token_mask=False,
        )
        post = WhisperDecoderPostModel(
            self.cfg,
            self.model_name,
            hf_model=self.hf_model,
            num_tokens=num_tokens,
            layer_idx=self.layer_idx,
            skip_encoder_kv_proj=False,
            output_encoder_kv_cache=True,
            enable_log_probe=self.enable_log_probe,
        )
        pre_inputs = [inputs[0]]
        if self.layer_idx == 0:
            positions = graph.parameter("model.decoder.embed_positions.weight")[: self.num_tokens]
            pre_inputs.append(
                graph.constant(positions.reshape(1, 1, self.num_tokens, self.cfg.d_model))
            )
        pre_outputs = pre._build_nodes(graph, pre_inputs)
        query, key, value = pre_outputs[:3]
        residual = pre_outputs[pre.positioned_residual_output_idx] if self.layer_idx == 0 else inputs[0]
        if final_layer:
            query = graph.slice(query, [self.num_tokens - 1], [self.num_tokens], [1], [2])
            residual = graph.slice(
                residual, [self.num_tokens - 1], [self.num_tokens], [1], [2]
            )
        attn = cache._build_nodes(graph, [query, key, value])[0]
        outputs = post._build_nodes(graph, [residual, attn, inputs[1]])
        # Hidden/token, optional logits, self K/V, encoder K/V.
        graph.save([*outputs[:-2], key, value, *outputs[-2:]])
