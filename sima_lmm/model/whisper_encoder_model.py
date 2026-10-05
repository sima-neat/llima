from dataclasses import dataclass
import numpy as np

from sima_lmm.model.model_graph import ModelGraph
from sima_lmm.model.base import BaseModel, LayerConfiguration
from sima_lmm.model.onnx_builder import OnnxNode


@dataclass
class WhisperEncoderModel(BaseModel):
    layer_idx: int | None = None

    def gen_model_sdk_files_directly(
        self, layer_cfg: LayerConfiguration, log_level: int, quantizable: bool
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
        outputs = self._build_sima_nodes(graph.raw, list(graph.inputs.values()), quantizable)
        graph.save(outputs)

    def _build_sima_nodes(self, builder, inputs, quantizable):
        graph = ModelGraph.from_builder(self, builder)
        hidden = inputs[0]
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
            hidden = builder.create_add_node(hidden, positions)
        layers = range(self.cfg.encoder_layers) if self.layer_idx is None else [self.layer_idx]
        for idx in layers:
            name = f"model.encoder.layers.{idx}"
            norm = graph.layer_norm(f"{name}.self_attn_layer_norm", hidden)
            queries, keys, values = [
                graph.project_heads(
                    f"{name}.self_attn.{proj}",
                    norm,
                    self.cfg.encoder_attention_heads,
                    scale=self.cfg.encoder_head_dim**-0.5 if proj == "q_proj" else 1.0,
                )
                for proj in ("q_proj", "k_proj", "v_proj")
            ]
            heads = []
            for query, key, value in zip(queries, keys, values):
                heads.append(graph.attention(query, key, value))
            attn = graph.project_merged_heads(
                f"{name}.self_attn.out_proj",
                heads,
                self.cfg.encoder_attention_heads,
            )
            hidden = builder.create_add_node(hidden, attn)
            norm = graph.layer_norm(f"{name}.final_layer_norm", hidden)
            hidden = graph.mlp(name, norm, self.cfg.activation_function, residual=hidden)
        if self.layer_idx in (None, self.cfg.encoder_layers - 1):
            hidden = graph.layer_norm("model.encoder.layer_norm", hidden)
        return [hidden]

    def gen_onnx_files(self):
        base_name = "model.encoder"
        self.create_onnx_builder()

        if self.layer_idx is None:
            input_shape = (
                1, self.cfg.num_mel_bins, 1, self.cfg.max_source_positions * 2
            )
        else:
            if not 0 <= self.layer_idx < self.cfg.encoder_layers:
                raise ValueError(
                    f"Invalid Whisper encoder layer index {self.layer_idx}; expected "
                    f"0 <= layer_idx < {self.cfg.encoder_layers}"
                )
            input_shape = (
                (1, self.cfg.num_mel_bins, 1, self.cfg.max_source_positions * 2)
                if self.layer_idx == 0
                else (1, self.cfg.d_model, 1, self.cfg.max_source_positions)
            )

        self._onnx_builder.create_input_node("input", input_shape)
        if self.layer_idx is None:
            output_nodes = self._build_onnx_nodes(base_name, self._onnx_builder.input_nodes)
        else:
            output_nodes = self._build_layer_onnx_nodes(
                base_name, self._onnx_builder.input_nodes
            )
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[0]),
            (1, self.cfg.d_model, 1, self.cfg.max_source_positions)
        )
        self._onnx_builder.create_and_save_model()

        # Set to None to deallocate the memory.
        self._onnx_builder = None

    def _build_onnx_nodes(self, base_name: str, input_nodes: list[OnnxNode]) -> list[OnnxNode]:
        feature_extractor_output = self._build_feature_extractor(base_name, input_nodes[0])

        encoder_input = feature_extractor_output
        for layer_idx in range(self.cfg.encoder_layers):
            encoder_output = self._build_encoder_layer(
                f"{base_name}.layers.{layer_idx}", encoder_input
            )
            encoder_input = encoder_output

        layer_norm = self._onnx_builder.build_layer_norm(f"{base_name}.layer_norm", encoder_output)
        return [layer_norm]

    def _build_layer_onnx_nodes(
        self, base_name: str, input_nodes: list[OnnxNode]
    ) -> list[OnnxNode]:
        assert self.layer_idx is not None
        encoder_input = input_nodes[0]
        if self.layer_idx == 0:
            encoder_input = self._build_feature_extractor(base_name, encoder_input)

        encoder_output = self._build_encoder_layer(
            f"{base_name}.layers.{self.layer_idx}", encoder_input
        )
        if self.layer_idx == self.cfg.encoder_layers - 1:
            encoder_output = self._onnx_builder.build_layer_norm(
                f"{base_name}.layer_norm", encoder_output
            )
        return [encoder_output]

    def _build_feature_extractor(self, base_name: str, input_node: OnnxNode) -> OnnxNode:
        conv1 = self._onnx_builder.build_conv(
            f"{base_name}.conv1", input_node, is_fc=False, reshape_str="ncw->nchw",
            pads=[0, 1, 0, 1], kernel_shape=[1, 3], strides=[1, 1]
        )
        gelu1 = self._onnx_builder.build_activation(f"{base_name}.gelu1", conv1, "gelu")
        conv2 = self._onnx_builder.build_conv(
            f"{base_name}.conv2", gelu1, is_fc=False, reshape_str="ncw->nchw",
            pads=[0, 1, 0, 1], kernel_shape=[1, 3], strides=[1, 2]
        )
        gelu2 = self._onnx_builder.build_activation(f"{base_name}.gelu2", conv2, "gelu")
        embed_positions = self._onnx_builder.create_initializer(
            f"{base_name}.embed_positions.weight", reshape_str="wc->nchw"
        )
        add = self._onnx_builder.build_op(f"{base_name}.add", [gelu2, embed_positions], "Add")
        return add

    def _build_encoder_layer(self, base_name: str, input_node: OnnxNode) -> OnnxNode:
        layer_norm1 = self._onnx_builder.build_layer_norm(
            f"{base_name}.self_attn_layer_norm", input_node
        )
        self_attn, *_ = self._onnx_builder.build_attention(
            f"{base_name}.self_attn", [layer_norm1], self.cfg.encoder_attention_heads, 
            self.cfg.encoder_head_dim, self.cfg.max_source_positions
        )
        add1 = self._onnx_builder.build_op(f"{base_name}.add1", [input_node, self_attn], "Add")
        layer_norm2 = self._onnx_builder.build_layer_norm(f"{base_name}.final_layer_norm", add1)
        mlp = self._onnx_builder.build_encoder_decoder_mlp(
            base_name, layer_norm2, self.cfg.activation_function
        )
        add2 = self._onnx_builder.build_op(f"{base_name}.add2", [add1, mlp], "Add")
        return add2
