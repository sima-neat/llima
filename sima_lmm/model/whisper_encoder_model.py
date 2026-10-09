from dataclasses import dataclass

from sima_lmm.model.model_graph import ModelGraph
from sima_lmm.model.base import BaseModel, LayerConfiguration


@dataclass
class WhisperEncoderModel(BaseModel):
    layer_idx: int | None = None

    def generate_graph(
        self, layer_cfg: LayerConfiguration, quantizable: bool
    ):
        if self.layer_idx is not None and not 0 <= self.layer_idx < self.cfg.encoder_layers:
            raise ValueError(f"Invalid Whisper encoder layer index {self.layer_idx}")
        shape = (
            (1, 1, self.cfg.max_source_positions * 2, self.cfg.num_mel_bins)
            if self.layer_idx in (None, 0)
            else (1, 1, self.cfg.max_source_positions, self.cfg.d_model)
        )
        shapes = {"input": shape}

        graph = ModelGraph(self, shapes, quantizable)
        hidden = graph.inputs["input"]
        if self.layer_idx in (None, 0):
            for idx, stride in ((1, 1), (2, 2)):
                conv = graph.conv(
                    f"model.encoder.conv{idx}",
                    hidden,
                    padding=((0, 0), (1, 1)),
                    stride=(1, stride),
                )
                hidden = graph.activation(conv, "gelu")
            positions = graph.parameter("model.encoder.embed_positions.weight")
            positions = graph.constant(
                positions.reshape(1, 1, self.cfg.max_source_positions, self.cfg.d_model)
            )
            hidden = graph.add(hidden, positions)
        layers = range(self.cfg.encoder_layers) if self.layer_idx is None else [self.layer_idx]
        for idx in layers:
            name = f"model.encoder.layers.{idx}"
            norm = graph.layer_norm(f"{name}.self_attn_layer_norm", hidden)
            queries, keys, values = [
                graph.split_heads(
                    graph.linear(
                        f"{name}.self_attn.{proj}", norm,
                        scale=self.cfg.encoder_head_dim**-0.5 if proj == "q_proj" else 1.0,
                    ),
                    self.cfg.encoder_attention_heads,
                )
                for proj in ("q_proj", "k_proj", "v_proj")
            ]
            context = graph.attention(queries, keys, values)
            attn = graph.linear(
                f"{name}.self_attn.out_proj", graph.merge_heads(context)
            )
            hidden = graph.add(hidden, attn)
            norm = graph.layer_norm(f"{name}.final_layer_norm", hidden)
            hidden = graph.mlp(name, norm, self.cfg.activation_function, residual=hidden)
        if self.layer_idx in (None, self.cfg.encoder_layers - 1):
            hidden = graph.layer_norm("model.encoder.layer_norm", hidden)
        graph.save([hidden])
