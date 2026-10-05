from dataclasses import dataclass
import numpy as np

from afe.apis.defines import gen2_target
from afe.backends.backends import Backend
from afe.ir.defines import Status, get_expected_tensor_value
from afe.ir.serializer import save_awesomenet
from afe.ir.tensor_type import ScalarType, TensorType

from sima_lmm.model.base import BaseModel, LayerConfiguration, TensorTessellateParameters
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import (
    SimaBuilder, activation_type, activation_dtype, build_activation, build_conv,
    build_two_stage_layer_norm, build_matmul_and_split_heads, build_merge_heads_and_matmul,
)


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

        builder = SimaBuilder(Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target)
        model_inputs = [
            builder.create_placeholder_node(name, TensorType(activation_type(quantizable), shape))
            for name, shape in shapes.items()
        ]
        builder.begin_subnet(model_inputs)
        inputs = [
            builder.create_placeholder_node(
                f"MLA_0/{name}", TensorType(activation_type(quantizable), shape)
            )
            for name, shape in shapes.items()
        ]
        outputs = self._build_sima_nodes(builder, inputs, quantizable)
        if len(outputs) > 1:
            builder.create_tuple_node(outputs)
        mla = builder.finish_subnet("MLA_0")
        outputs = builder.create_tuple_get_item_nodes(mla) if len(outputs) > 1 else [mla]
        for i, output in enumerate(outputs):
            if get_expected_tensor_value(output.get_type().output).scalar == ScalarType.bfloat16:
                outputs[i] = builder.create_cast_node(output, ScalarType.float32, backend=Backend.EV)
        if len(outputs) > 1:
            builder.create_tuple_node(outputs)
        net = builder.finish(self.model_name)
        save_awesomenet(
            net, self.model_name + (".fp32" if quantizable else ""),
            str(self.sima_model_sdk_path),
        )

    def _build_sima_nodes(self, builder, inputs, quantizable):
        hidden = inputs[0]
        if self.layer_idx in (None, 0):
            for idx, stride in ((1, 1), (2, 2)):
                conv = build_conv(
                    builder, self.get_hf_param, self.check_hf_param,
                    f"model.encoder.conv{idx}", hidden, is_fc=False,
                    reshape_str="oiw->oihw", padding=((0, 0), (1, 1)), stride=(1, stride),
                )
                hidden = build_activation(builder, conv, "gelu", quantizable)
            positions = self.get_hf_param("model.encoder.embed_positions.weight")
            positions = builder.create_constant_node(
                positions.reshape(1, 1, self.cfg.max_source_positions, self.cfg.d_model)
                .astype(activation_dtype(quantizable))
            )
            hidden = builder.create_add_node(hidden, positions)
        layers = range(self.cfg.encoder_layers) if self.layer_idx is None else [self.layer_idx]
        for idx in layers:
            name = f"model.encoder.layers.{idx}"
            norm = build_two_stage_layer_norm(
                builder, self.get_hf_param, self.check_hf_param,
                f"{name}.self_attn_layer_norm", hidden, axis=-1, epsilon=float(np.float32(1e-5)),
            )
            queries, keys, values = [
                build_matmul_and_split_heads(
                    builder, self.get_hf_param, self.check_hf_param, f"{name}.self_attn.{proj}",
                    norm, self.cfg.encoder_attention_heads, self.cfg.max_source_positions,
                    post_matmul_scale=self.cfg.encoder_head_dim ** -0.5 if proj == "q_proj" else 1.0,
                )
                for proj in ("q_proj", "k_proj", "v_proj")
            ]
            heads = []
            for query, key, value in zip(queries, keys, values):
                scores = builder.create_einsum_node(query, key, "nhwc,nhqc->nhwq")
                probs = builder.create_softmax_node(scores, axis=3)
                heads.append(builder.create_einsum_node(probs, value, "nhwc,nhcq->nhwq"))
            attn = build_merge_heads_and_matmul(
                builder, self.get_hf_param, self.check_hf_param,
                f"{name}.self_attn.out_proj", heads, self.cfg.encoder_attention_heads,
            )
            hidden = builder.create_add_node(hidden, attn)
            norm = build_two_stage_layer_norm(
                builder, self.get_hf_param, self.check_hf_param,
                f"{name}.final_layer_norm", hidden, axis=-1, epsilon=float(np.float32(1e-5)),
            )
            fc1 = build_conv(builder, self.get_hf_param, self.check_hf_param, f"{name}.fc1", norm)
            act = build_activation(builder, fc1, self.cfg.activation_function, quantizable)
            fc2 = build_conv(builder, self.get_hf_param, self.check_hf_param, f"{name}.fc2", act)
            hidden = builder.create_add_node(hidden, fc2)
        if self.layer_idx in (None, self.cfg.encoder_layers - 1):
            hidden = build_two_stage_layer_norm(
                builder, self.get_hf_param, self.check_hf_param,
                "model.encoder.layer_norm", hidden, axis=-1, epsilon=float(np.float32(1e-5)),
            )
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
