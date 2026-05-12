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

import numpy as np

from dataclasses import dataclass

from sima_lmm.model.base import BaseModel, TensorTessellateParameters
from sima_lmm.model.onnx_builder import OnnxNode


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

    def gen_onnx_files(self):
        base_name = f"model.decoder.tokens.{self.token_idx}"
        self.create_onnx_builder()
        self._onnx_builder.create_input_node(
            "input",
            (1, self.cfg.decoder_head_dim, self.cfg.decoder_attention_heads, self.num_tokens)
        )
        self._onnx_builder.create_input_node(
            "cached_keys", (1, self.cfg.d_model, 1, self.token_idx + self.num_tokens)
        )
        self._onnx_builder.create_input_node(
            "cached_values", (1, self.cfg.d_model, 1, self.token_idx + self.num_tokens)
        )
        if self.use_future_token_mask and self.num_tokens == 1:
            self._onnx_builder.create_input_node("attn_mask", (1, self.token_idx + 1, 1, 1))
        output_nodes = self._build_onnx_nodes(base_name, self._onnx_builder.input_nodes)
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[0]),
            (1, self.cfg.d_model, 1, self.num_tokens)
        )
        self._onnx_builder.create_and_save_model()

        # Set to None to deallocate the memory.
        self._onnx_builder = None

    def _build_onnx_nodes(self, base_name: str, input_nodes: list[OnnxNode]) -> list[OnnxNode]:
        reshape_keys = self._onnx_builder.build_split_and_concat(
            f"{base_name}.cached_keys.reshape", input_nodes[1], self.cfg.decoder_attention_heads,
            split_axis=1, concat_axis=2
        )
        reshape_values = self._onnx_builder.build_split_and_concat(
            f"{base_name}.cached_values.reshape", input_nodes[2], self.cfg.decoder_attention_heads,
            split_axis=1, concat_axis=2
        )
        bmm1 = self._onnx_builder.build_op(
            f"{base_name}.bmm1", [input_nodes[0], reshape_keys], "Einsum",
            equation="nchw,nchq->nqhw"
        )
        if self.num_tokens > 1:
            mask = np.zeros(
                (1, self.token_idx + self.num_tokens, 1, self.num_tokens), dtype=np.float32
            )
            for i in range(self.num_tokens):
                for j in range(self.token_idx + i + 1, self.token_idx + self.num_tokens):
                    mask[0, j, 0, i] = np.finfo(np.float32).min
            bmm1 = self._onnx_builder.build_op(f"{base_name}.masked_bmm1", [bmm1, mask], "Add")
        elif self.use_future_token_mask:
            bmm1 = self._onnx_builder.build_op(
                f"{base_name}.masked_bmm1", [bmm1, input_nodes[3]], "Add"
            )
        softmax = self._onnx_builder.build_op(f"{base_name}.softmax", [bmm1], "Softmax", axis=1)
        bmm2 = self._onnx_builder.build_op(
            f"{base_name}.bmm2", [softmax, reshape_values], "Einsum",
            equation="nchw,nqhc->nqhw"
        )
        reshape_bmm2 = self._onnx_builder.build_split_and_concat(
            f"{base_name}.bmm2.reshape", bmm2, self.cfg.decoder_attention_heads,
            split_axis=2, concat_axis=1
        )
        return [reshape_bmm2]

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
