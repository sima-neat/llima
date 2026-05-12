#########################################################
# Copyright (C) 2025 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################

from dataclasses import dataclass

from sima_lmm.model.base import BaseModel, TensorTessellateParameters
from sima_lmm.model.onnx_builder import OnnxNode


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

    def __post_init__(self):
        assert 0 <= self.layer_idx < self.cfg.decoder_layers

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

        self._onnx_builder.create_and_save_model()

        # Set to None to deallocate the memory.
        self._onnx_builder = None

    def _build_onnx_nodes(self, base_name: str, input_nodes: list[OnnxNode]) -> list[OnnxNode]:
        if self.layer_idx == 0:
            assert len(input_nodes) == 2
            add = self._onnx_builder.build_op(f"{base_name}.add_embed", input_nodes, "Add")
            layer_norm = self._onnx_builder.build_layer_norm(
                f"{base_name}.self_attn_layer_norm", add
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
        return [reshaped_q_proj, k_proj, v_proj]

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
