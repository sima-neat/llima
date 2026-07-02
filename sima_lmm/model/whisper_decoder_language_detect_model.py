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

from sima_lmm.hf.hf_transformer import find_file
from sima_lmm.model.base import BaseModel, TensorTessellateParameters
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.whisper_decoder_cache_model import WhisperDecoderCacheModel
from sima_lmm.model.whisper_decoder_post_model import WhisperDecoderPostModel
from sima_lmm.model.whisper_decoder_pre_model import WhisperDecoderPreModel
from sima_lmm.tokenizer.whisper_tokenizer import get_tokenizer


@dataclass
class WhisperDecoderLanguageDetectModel(BaseModel):
    """One-shot Whisper decoder graph for automatic language detection."""

    NUM_TOKENS = 1
    TOKEN_IDX = 0

    @property
    def enable_filter_sharing(self) -> bool:
        return self.use_filter_sharing

    def gen_onnx_files(self):
        self.create_onnx_builder()
        self._onnx_builder.create_input_node(
            "audio_features", (1, self.cfg.d_model, 1, self.cfg.max_source_positions)
        )

        hf_tokenizer_json_file = find_file(
            directory=self.hf_model.hf_cache, filename="tokenizer.json"
        )
        tokenizer = get_tokenizer(
            multilingual=True, num_languages=self.cfg.num_languages, language=None, task=None,
            hf_tokenizer_json_file=hf_tokenizer_json_file
        )
        language_token_ids = tokenizer.all_language_tokens
        language_start_token_id = language_token_ids[0]
        num_languages = len(language_token_ids)
        if language_token_ids != tuple(
            range(language_start_token_id, language_start_token_id + num_languages)
        ):
            raise RuntimeError("Whisper language tokens must be contiguous.")

        audio_features = self._onnx_builder.input_nodes[0]
        hidden = self._build_sot_hidden(tokenizer.sot)
        for layer_idx in range(self.cfg.decoder_layers):
            hidden = self._build_decoder_layer(
                layer_idx, hidden, audio_features, language_start_token_id, num_languages
            )

        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(hidden), (1, 1, 1, 1), np.int64
        )
        self._onnx_builder.create_and_save_model()

        # Set to None to deallocate the memory.
        self._onnx_builder = None

    def _build_sot_hidden(self, sot_token_id: int) -> list[OnnxNode]:
        token_embeddings = self.get_hf_param("model.decoder.embed_tokens.weight")
        position_embeddings = self.get_hf_param("model.decoder.embed_positions.weight")
        assert isinstance(token_embeddings, np.ndarray)
        assert isinstance(position_embeddings, np.ndarray)

        sot_embedding = token_embeddings[sot_token_id].reshape(1, self.cfg.d_model, 1, 1)
        sot_position_embedding = position_embeddings[0].reshape(1, self.cfg.d_model, 1, 1)
        sot_embedding_node = self._onnx_builder.create_initializer(
            "language_detect.sot_embedding", sot_embedding
        )
        sot_position_node = self._onnx_builder.create_initializer(
            "language_detect.sot_position_embedding", sot_position_embedding
        )
        return [sot_embedding_node, sot_position_node]

    def _build_decoder_layer(
        self, layer_idx: int, hidden: OnnxNode | list[OnnxNode], audio_features: OnnxNode,
        language_start_token_id: int, num_languages: int
    ) -> OnnxNode:
        base_name = f"model.decoder.layers.{layer_idx}"

        pre_model = WhisperDecoderPreModel(
            self.cfg, self.model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
            hf_model=self.hf_model, num_tokens=self.NUM_TOKENS, layer_idx=layer_idx
        )
        pre_model._onnx_builder = self._onnx_builder
        pre_input_nodes = hidden if layer_idx == 0 else [hidden]
        pre_output_nodes = pre_model._build_onnx_nodes(base_name, pre_input_nodes)

        cache_model = WhisperDecoderCacheModel(
            self.cfg, self.model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
            hf_model=self.hf_model, num_tokens=self.NUM_TOKENS, token_idx=self.TOKEN_IDX,
            use_future_token_mask=False
        )
        cache_model._onnx_builder = self._onnx_builder
        cache_output_nodes = cache_model._build_onnx_nodes(base_name, pre_output_nodes)

        if layer_idx < self.cfg.decoder_layers - 1:
            post_model = WhisperDecoderPostModel(
                self.cfg, self.model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
                hf_model=self.hf_model, num_tokens=self.NUM_TOKENS, layer_idx=layer_idx,
                skip_encoder_kv_proj=False, output_encoder_kv_cache=False
            )
            post_model._onnx_builder = self._onnx_builder
            post_output_nodes = post_model._build_onnx_nodes(
                base_name, [pre_input_nodes[0], cache_output_nodes[0], audio_features]
            )
            return post_output_nodes[0]

        return self._build_final_post_nodes(
            base_name, [pre_input_nodes[0], cache_output_nodes[0], audio_features],
            language_start_token_id, num_languages
        )

    def _build_final_post_nodes(
        self, base_name: str, input_nodes: list[OnnxNode],
        language_start_token_id: int, num_languages: int
    ) -> OnnxNode:
        o_proj = self._onnx_builder.build_conv(f"{base_name}.self_attn.out_proj", input_nodes[1])
        add1 = self._onnx_builder.build_op(f"{base_name}.add1", [input_nodes[0], o_proj], "Add")
        encoder_attn_layer_norm = self._onnx_builder.build_layer_norm(
            f"{base_name}.encoder_attn_layer_norm", add1
        )
        encoder_attn, _, _ = self._onnx_builder.build_attention(
            base_name=f"{base_name}.encoder_attn",
            input_nodes=[encoder_attn_layer_norm, input_nodes[-1], input_nodes[-1]],
            num_heads=self.cfg.decoder_attention_heads,
            head_dim=self.cfg.decoder_head_dim,
            seq_len=1,
            kv_len=self.cfg.max_source_positions,
            skip_kv_projs_and_split_head=False,
            output_kv_projs=True,
        )
        add2 = self._onnx_builder.build_op(f"{base_name}.add2", [add1, encoder_attn], "Add")
        layer_norm1 = self._onnx_builder.build_layer_norm(f"{base_name}.final_layer_norm", add2)
        mlp = self._onnx_builder.build_encoder_decoder_mlp(
            base_name, layer_norm1, self.cfg.activation_function
        )
        add3 = self._onnx_builder.build_op(f"{base_name}.add3", [add2, mlp], "Add")
        layer_norm2 = self._onnx_builder.build_layer_norm("model.decoder.layer_norm", add3)
        lm_head = self._onnx_builder.build_conv(
            "model.decoder.embed_tokens",
            layer_norm2,
            weight_slice=(language_start_token_id, num_languages, 0, 0)
        )
        return self._onnx_builder.build_op(
            "language_argmax", [lm_head], "ArgMax", axis=1, keepdims=1,
            output_names=["detected_language_index"]
        )

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """
        Get the DRAM layouts to use for this model's inputs on the MLA.
        """
        return {}

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """
        Get the DRAM layouts to use for this model's output on the MLA.
        """
        return {}
