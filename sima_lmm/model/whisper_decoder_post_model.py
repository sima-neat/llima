import numpy as np

from dataclasses import dataclass

from sima_lmm.hf.hf_transformer import find_file
from sima_lmm.model.model_graph import ModelGraph
from sima_lmm.model.base import BaseModel, LayerConfiguration
from sima_lmm.tokenizer.whisper_tokenizer import get_tokenizer


@dataclass
class WhisperDecoderPostModel(BaseModel):
    """Implementation for the post cache model of Whisper.

    This implements a simplified version of the LanguagePostModel. This model is only used when
    generating new tokens so the num_tokens is assumed to be 1.

    Attributes:
        num_tokens: Number of tokens. Set to a value greater than 1 to consume multiple input tokens
            in one model.
        layer_idx: Transformer layer index.
        skip_encoder_kv_proj: Whether to skip the key/value projections in cross attention.
        output_encoder_kv_cache: Whether to output the key/value projections from cross attention.
        enable_log_probe: Whether to also output filtered logits from the final layer.
    """
    num_tokens: int
    layer_idx: int
    skip_encoder_kv_proj: bool
    output_encoder_kv_cache: bool
    enable_log_probe: bool = False

    def __post_init__(self):
        assert 0 <= self.layer_idx < self.cfg.decoder_layers

    @property
    def enable_filter_sharing(self) -> bool:
        return self.use_filter_sharing

    def generate_graph(
        self, layer_cfg: LayerConfiguration, quantizable: bool
    ):
        assert not self.output_encoder_kv_cache
        hidden_shape = (1, 1, self.num_tokens, self.cfg.d_model)
        cache_shape = (
            1,
            self.cfg.decoder_attention_heads,
            self.cfg.max_source_positions,
            self.cfg.decoder_head_dim,
        )
        shapes = {
            "input": hidden_shape,
            "self_attn": hidden_shape,
            "encoder_k_cache": cache_shape,
            "encoder_v_cache": cache_shape,
        }

        graph = ModelGraph(self, shapes, quantizable)
        outputs = self._build_nodes(graph, list(graph.inputs.values()))
        graph.save(outputs)

    def _build_transformer(self, graph, inputs):
        name = f"model.decoder.layers.{self.layer_idx}"
        proj = graph.linear(f"{name}.self_attn.out_proj", inputs[1])
        hidden = graph.add(inputs[0], proj)
        norm = graph.layer_norm(f"{name}.encoder_attn_layer_norm", hidden)
        kv = inputs[2:4] if self.skip_encoder_kv_proj else [inputs[-1], inputs[-1]]
        queries = graph.split_heads(
            graph.linear(
                f"{name}.encoder_attn.q_proj", norm, scale=self.cfg.decoder_head_dim**-0.5,
            ),
            self.cfg.decoder_attention_heads,
        )
        kv_projs = []
        for proj, node in zip(("k_proj", "v_proj"), kv):
            if self.skip_encoder_kv_proj:
                heads = node
            else:
                heads = graph.split_heads(
                    graph.linear(f"{name}.encoder_attn.{proj}", node),
                    self.cfg.decoder_attention_heads,
                )
            kv_projs.append(heads)
        keys, values = kv_projs
        context = graph.attention(queries, keys, values)
        attn = graph.linear(
            f"{name}.encoder_attn.out_proj", graph.merge_heads(context)
        )
        hidden = graph.add(hidden, attn)
        norm = graph.layer_norm(f"{name}.final_layer_norm", hidden)
        hidden = graph.mlp(name, norm, self.cfg.activation_function, residual=hidden)
        return hidden, keys, values

    def _build_nodes(self, graph, inputs):
        hidden, keys, values = self._build_transformer(graph, inputs)
        if self.layer_idx < self.cfg.decoder_layers - 1:
            outputs = [hidden]
        else:
            norm = graph.layer_norm("model.decoder.layer_norm", hidden)
            logits = graph.linear("model.decoder.embed_tokens", norm)
            mask = np.zeros((1, 1, 1, self.cfg.vocab_size), dtype=np.float32)
            mask[..., self.cfg.suppress_tokens + self._get_extra_suppress_tokens()] = np.finfo(
                np.float32
            ).min
            mask = graph.constant(mask)
            logits = graph.add(logits, mask)
            outputs = [graph.argmax(logits)]
            if self.enable_log_probe:
                outputs.append(logits)
        if self.output_encoder_kv_cache:
            outputs.extend([keys, values])
        return outputs

    def _get_extra_suppress_tokens(self) -> list[int]:
        hf_tokenizer_json_file = find_file(
            directory=self.hf_model.hf_cache, filename="tokenizer.json"
        )
        tokenizer = get_tokenizer(
            multilingual=True, num_languages=self.cfg.num_languages, language=None, task=None,
            hf_tokenizer_json_file=hf_tokenizer_json_file
        )
        return [
            220,  # standalone space
            tokenizer.no_timestamps,
            tokenizer.timestamp_begin,
        ]
