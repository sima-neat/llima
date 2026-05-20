from dataclasses import dataclass

from sima_lmm.model.base import BaseModel, TensorTessellateParameters
from sima_lmm.model.onnx_builder import OnnxNode


@dataclass
class WhisperEncoderModel(BaseModel):
    def gen_onnx_files(self):
        base_name = "model.encoder"
        self.create_onnx_builder()
        self._onnx_builder.create_input_node(
            "input", (1, self.cfg.num_mel_bins, 1, self.cfg.max_source_positions * 2)
        )
        output_nodes = self._build_onnx_nodes(base_name, self._onnx_builder.input_nodes)
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

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters] :
        """
        Get the DRAM layouts to use for this model's inputs on the MLA.
        """
        return {}

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters] :
        """
        Get the DRAM layouts to use for this model's inputs on the MLA.
        """
        return {}
