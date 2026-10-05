from dataclasses import dataclass
from typing import ClassVar
import numpy as np

from afe.apis.defines import gen2_target
from afe.backends.backends import Backend
from afe.ir.defines import Status, get_expected_tensor_value
from afe.ir.serializer import save_awesomenet
from afe.ir.tensor_type import ScalarType, TensorType

from sima_lmm.model.base import BaseModel, LayerConfiguration, TensorTessellateParameters
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import (
    SimaBuilder, activation_type, activation_dtype, build_conv, build_two_stage_layer_norm,
)


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

    def gen_model_sdk_files_directly(
        self, layer_cfg: LayerConfiguration, log_level: int, quantizable: bool
    ):
        shape = (1, 1, self.num_tokens, self.cfg.d_model)
        shapes = {"input": shape}
        if self.layer_idx == 0:
            shapes["embed_positions"] = shape

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
        name = f"model.decoder.layers.{self.layer_idx}"
        residual = builder.create_add_node(*inputs) if self.layer_idx == 0 else inputs[0]
        norm = build_two_stage_layer_norm(
            builder, self.get_hf_param, self.check_hf_param,
            f"{name}.self_attn_layer_norm", residual, axis=-1, epsilon=float(np.float32(1e-5)),
        )
        query, key, value = [
            build_conv(
                builder, self.get_hf_param, self.check_hf_param, f"{name}.self_attn.{proj}_proj", norm
            )
            for proj in ("q", "k", "v")
        ]
        scale = builder.create_constant_node(
            np.array(self.cfg.decoder_head_dim ** -0.5, dtype=activation_dtype(quantizable))
        )
        query = builder.create_mul_node(query, scale)
        query = builder.create_slice_concat_node(
            query, axis=1, split_axis=3, split_block=self.cfg.decoder_attention_heads, split_repeat=1
        )
        outputs = [query, key, value]
        if self.layer_idx == 0:
            outputs.append(residual)
        return outputs

    def gen_onnx_files(self):
        base_name = f"model.decoder.layers.{self.layer_idx}"
        self.create_onnx_builder()
        self._onnx_builder.create_input_node("input", (1, self.cfg.d_model, 1, self.num_tokens))
        if self.layer_idx == 0:
            self._onnx_builder.create_input_node(
                "embed_positions", (1, self.cfg.d_model, 1, self.num_tokens)
            )
        output_nodes = self._build_onnx_nodes(base_name, self._onnx_builder.input_nodes)

        # q_proj
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[0]),
            (1, self.cfg.decoder_head_dim, self.cfg.decoder_attention_heads, self.num_tokens)
        )
        # self_k_cache
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[1]),
            (1, self.cfg.d_model, 1, self.num_tokens)
        )
        # self_v_cache
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[2]),
            (1, self.cfg.d_model, 1, self.num_tokens)
        )
        if self.layer_idx == 0:
            # Layer 0 adds the learned position embedding before entering the decoder. Preserve
            # that complete hidden state for the residual path in the separately compiled post
            # model.
            self._onnx_builder.create_output_node(
                self._onnx_builder.get_node_output_name(
                    output_nodes[self.positioned_residual_output_idx]
                ),
                (1, self.cfg.d_model, 1, self.num_tokens)
            )

        self._onnx_builder.create_and_save_model()

        # Set to None to deallocate the memory.
        self._onnx_builder = None

    def _build_onnx_nodes(self, base_name: str, input_nodes: list[OnnxNode]) -> list[OnnxNode]:
        residual = input_nodes[0]
        if self.layer_idx == 0:
            assert len(input_nodes) == 2
            residual = self._onnx_builder.build_op(
                f"{base_name}.add_embed", input_nodes, "Add"
            )
            layer_norm = self._onnx_builder.build_layer_norm(
                f"{base_name}.self_attn_layer_norm", residual
            )
        else:
            assert len(input_nodes) == 1
            layer_norm = self._onnx_builder.build_layer_norm(
                f"{base_name}.self_attn_layer_norm", input_nodes[0]
            )
        
        q_proj = self._onnx_builder.build_conv(f"{base_name}.self_attn.q_proj", layer_norm)
        k_proj = self._onnx_builder.build_conv(f"{base_name}.self_attn.k_proj", layer_norm)
        v_proj = self._onnx_builder.build_conv(f"{base_name}.self_attn.v_proj", layer_norm)

        scaled_q_proj = self._onnx_builder.build_op(
            f"{base_name}.self_attn.scaled_q_proj", [q_proj, self.cfg.decoder_head_dim ** -0.5],
            "Mul"
        )
        reshaped_q_proj = self._onnx_builder.build_split_and_concat(
            f"{base_name}.self_attn.scaled_q_proj.reshape", scaled_q_proj,
            self.cfg.decoder_attention_heads, split_axis=1, concat_axis=2
        )
        output_nodes = [reshaped_q_proj, k_proj, v_proj]
        if self.layer_idx == 0:
            output_nodes.append(residual)
        return output_nodes

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
