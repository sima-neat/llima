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
from typing import ClassVar

from sima_lmm.model.base import BaseModel, TensorTessellateParameters
from sima_lmm.model.whisper_decoder_cache_model import WhisperDecoderCacheModel
from sima_lmm.model.whisper_decoder_post_model import WhisperDecoderPostModel
from sima_lmm.model.whisper_decoder_pre_model import WhisperDecoderPreModel
from sima_lmm.model.onnx_builder import OnnxNode


@dataclass
class WhisperDecoderInitModel(BaseModel):
    """Implementation Whisper's decoder layers to process the input tokens and encoder outputs.

    Attributes:
        layer_idx: Transformer layer index.
    """
    layer_idx: int
    # 4 init tokens used in the init model:
    #   1. <|startoftranscript|>,
    #   2. `language`,
    #   3. <|transcribe|>
    #   4. <|notimestamps|>
    num_tokens: ClassVar[int] = 4
    token_idx: ClassVar[int] = 0

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
        self._onnx_builder.create_input_node(
            "audio_features", (1, self.cfg.d_model, 1, self.cfg.max_source_positions)
        )
        output_nodes = self._build_onnx_nodes(base_name, self._onnx_builder.input_nodes)

        if self.layer_idx < self.cfg.decoder_layers - 1:
            # Decoder layer output.
            self._onnx_builder.create_output_node(
                self._onnx_builder.get_node_output_name(output_nodes[0]),
                (1, self.cfg.d_model, 1, self.num_tokens)
            )
        else:
            # Argmax output.
            self._onnx_builder.create_output_node(
                self._onnx_builder.get_node_output_name(output_nodes[0]), (1, 1, 1, 1), np.int64
            )
        # Self-attention key projections.
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[1]),
            (1, self.cfg.d_model, 1, self.num_tokens)
        )
        # Self-attention value projections.
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[2]),
            (1, self.cfg.d_model, 1, self.num_tokens)
        )
        # Cross-attention key projections.
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[3]),
            (
                1, self.cfg.decoder_head_dim, self.cfg.decoder_attention_heads,
                self.cfg.max_source_positions
            )
        )
        # Cross-attention value projections.
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[4]),
            (
                1, self.cfg.decoder_head_dim, self.cfg.decoder_attention_heads,
                self.cfg.max_source_positions
            )
        )
        self._onnx_builder.create_and_save_model()

        # Set to None to deallocate the memory.
        self._onnx_builder = None

    def _build_onnx_nodes(self, base_name: str, input_nodes: list[OnnxNode]) -> list[OnnxNode]:
        assert self.token_idx == 0
        pre_model = WhisperDecoderPreModel(
            self.cfg, self.model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
            hf_model=self.hf_model, num_tokens=self.num_tokens, layer_idx=self.layer_idx
        )
        pre_model._onnx_builder = self._onnx_builder
        if self.layer_idx < self.cfg.decoder_layers - 1:
            num_tokens = self.num_tokens
            token_idx = self.token_idx
        else:
            num_tokens = 1
            token_idx = self.token_idx + self.num_tokens - 1
        cache_model = WhisperDecoderCacheModel(
            self.cfg, self.model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
            hf_model=self.hf_model, num_tokens=num_tokens, token_idx=token_idx,
            use_future_token_mask=False
        )
        cache_model._onnx_builder = self._onnx_builder
        post_model = WhisperDecoderPostModel(
            self.cfg, self.model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
            hf_model=self.hf_model, num_tokens=num_tokens, layer_idx=self.layer_idx,
            skip_encoder_kv_proj=False, output_encoder_kv_cache=True
        )
        post_model._onnx_builder = self._onnx_builder

        if self.layer_idx == 0:
            pre_input_nodes = [input_nodes[0], input_nodes[1]]
        else:
            pre_input_nodes = [input_nodes[0]]
        pre_output_nodes = pre_model._build_onnx_nodes(base_name, pre_input_nodes)

        if self.layer_idx < self.cfg.decoder_layers - 1:
            cache_input_nodes = pre_output_nodes
        else:
            slice_begin = self.token_idx + self.num_tokens - 1
            slice_end = slice_begin + 1
            slice_axis = 3
            last_token_q_proj = self._onnx_builder.build_op(
                f"{base_name}.last_token.slice_q_proj",
                [
                    pre_output_nodes[0],
                    np.array([slice_begin], dtype=np.int32),
                    np.array([slice_end], dtype=np.int32),
                    np.array([slice_axis], dtype=np.int32),
                ],
                "Slice"
            )
            cache_input_nodes = [last_token_q_proj, pre_output_nodes[1], pre_output_nodes[2]]
        cache_output_nodes = cache_model._build_onnx_nodes(base_name, cache_input_nodes)

        if self.layer_idx < self.cfg.decoder_layers - 1:
            post_input_nodes = [pre_input_nodes[0], cache_output_nodes[0], input_nodes[-1]]
        else:
            slice_begin = self.token_idx + self.num_tokens - 1
            slice_end = slice_begin + 1
            slice_axis = 3
            last_token_input_embed = self._onnx_builder.build_op(
                f"{base_name}.last_token.slice_input_embed",
                [
                    pre_input_nodes[0],
                    np.array([slice_begin], dtype=np.int32),
                    np.array([slice_end], dtype=np.int32),
                    np.array([slice_axis], dtype=np.int32),
                ],
                "Slice"
            )
            post_input_nodes = [last_token_input_embed, cache_output_nodes[0], input_nodes[-1]]
        post_output_nodes = post_model._build_onnx_nodes(base_name, post_input_nodes)
        return [
            # Decoder layer output or argmax output.
            post_output_nodes[0],
            # Self-attention key projections.
            pre_output_nodes[1],
            # Self-attention value projections.
            pre_output_nodes[2],
            # Cross-attention key projections.
            post_output_nodes[1],
            # Cross-attention value projections.
            post_output_nodes[2],
        ]

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
