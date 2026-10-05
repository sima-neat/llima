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

from dataclasses import dataclass, replace

from afe.ir import build_node
from afe.ir.execute import create_node_executor
from afe.ir.net import AwesomeNet
from afe.ir.operations import AddActivationOp, ConvAddActivationOp

from sima_lmm.hf.hf_transformer import find_file
from sima_lmm.model.model_graph import ModelGraph
from sima_lmm.model.base import BaseModel, LayerConfiguration
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

    def gen_model_sdk_files_directly(
        self, layer_cfg: LayerConfiguration, log_level: int, quantizable: bool
    ):
        shapes = {"audio_features": (1, 1, self.cfg.max_source_positions, self.cfg.d_model)}

        graph = ModelGraph(self, shapes, quantizable)
        outputs = self._build_sima_nodes(graph, list(graph.inputs.values()))
        graph.save(outputs, transform_subnet=self._fold_sot_prefix if quantizable else None)

    @staticmethod
    def _fold_sot_prefix(net: AwesomeNet):
        """Fold the constant start-token branch in FP32, as the ONNX importer does."""
        values = {}
        execute = create_node_executor(fast_mode=False)
        for name in net.execution_order:
            node = net.nodes[name]
            if all(producer in values for producer in node.input_node_names):
                inputs = dict(zip(node.input_names, (values[p] for p in node.input_node_names)))
                execute(node, inputs, values)
                folded = build_node.create_constant_node(0, values[name], status=net.status)
                net.nodes[name] = replace(folded, name=name)
            elif isinstance(node.ir.operation, AddActivationOp):
                lhs, rhs = (net.nodes[p] for p in node.input_node_names)
                constant, conv = (lhs, rhs) if lhs.name in values else (rhs, lhs)
                if constant.name not in values or not isinstance(conv.ir.operation, ConvAddActivationOp):
                    continue
                attrs = conv.ir.attrs
                residual = values[constant.name]
                if attrs.activ_attrs is not None or residual.shape != (1, 1, 1, attrs.conv_attrs.channels):
                    continue
                # Fold the constant residual into the projection bias before quantization.
                bias = residual.reshape(-1)
                if attrs.bias_attrs is not None:
                    bias = bias + attrs.bias_attrs.data
                folded = build_node.create_conv_node(
                    net.nodes[conv.input_node_names[0]], 0, attrs.weights_attrs.data, bias,
                    attrs.conv_attrs, node.ir.attrs.activ_attrs, status=net.status,
                )
                net.nodes[name] = replace(folded, name=name)
        net.topological_sort()
        net.nodes = {name: net.nodes[name] for name in net.execution_order}

    def _build_sima_nodes(self, graph, inputs):
        tokenizer = get_tokenizer(
            multilingual=True,
            num_languages=self.cfg.num_languages,
            language=None,
            task=None,
            hf_tokenizer_json_file=find_file(self.hf_model.hf_cache, "tokenizer.json"),
        )
        language_ids = tokenizer.all_language_tokens
        start, count = language_ids[0], len(language_ids)
        if language_ids != tuple(range(start, start + count)):
            raise RuntimeError("Whisper language tokens must be contiguous.")
        hidden = []
        for name, index in (("embed_tokens", tokenizer.sot), ("embed_positions", 0)):
            weight = graph.parameter(f"model.decoder.{name}.weight")[index]
            hidden.append(graph.constant(weight.reshape(1, 1, 1, self.cfg.d_model)))
        cache = WhisperDecoderCacheModel(
            self.cfg,
            self.model_name,
            num_tokens=1,
            token_idx=0,
            use_future_token_mask=False,
        )
        for idx in range(self.cfg.decoder_layers):
            pre = WhisperDecoderPreModel(
                self.cfg,
                self.model_name,
                hf_model=self.hf_model,
                num_tokens=1,
                layer_idx=idx,
            )
            pre_outputs = pre._build_sima_nodes(graph, hidden)
            residual = pre_outputs[pre.positioned_residual_output_idx] if idx == 0 else hidden[0]
            attn = cache._build_sima_nodes(graph, pre_outputs)[0]
            post = WhisperDecoderPostModel(
                self.cfg,
                self.model_name,
                hf_model=self.hf_model,
                num_tokens=1,
                layer_idx=idx,
                skip_encoder_kv_proj=False,
                output_encoder_kv_cache=False,
            )
            output, _, _ = post._build_sima_transformer(
                graph, [residual, attn, inputs[0]]
            )
            hidden = [output]
        norm = graph.layer_norm("model.decoder.layer_norm", hidden[0])
        logits = graph.linear("model.decoder.embed_tokens", norm)
        language_logits = graph.slice(logits, [start], [start + count], [1], [3])
        return [graph.argmax(language_logits), logits]

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

        detected_language_index, full_lm_head_logits = hidden
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(detected_language_index),
            (1, 1, 1, 1),
            np.int64
        )
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(full_lm_head_logits),
            (1, self.cfg.vocab_size, 1, 1)
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
    ) -> OnnxNode | list[OnnxNode]:
        base_name = f"model.decoder.layers.{layer_idx}"

        pre_model = WhisperDecoderPreModel(
            self.cfg, self.model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
            hf_model=self.hf_model, num_tokens=self.NUM_TOKENS, layer_idx=layer_idx
        )
        pre_model._onnx_builder = self._onnx_builder
        pre_input_nodes = hidden if layer_idx == 0 else [hidden]
        pre_output_nodes = pre_model._build_onnx_nodes(base_name, pre_input_nodes)
        residual_input = pre_input_nodes[0]
        if layer_idx == 0:
            residual_input = self._onnx_builder.build_op(
                f"{base_name}.residual_add_embed", pre_input_nodes, "Add"
            )

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
                base_name, [residual_input, cache_output_nodes[0], audio_features]
            )
            return post_output_nodes[0]

        return self._build_final_post_nodes(
            base_name, [residual_input, cache_output_nodes[0], audio_features],
            language_start_token_id, num_languages
        )

    def _build_final_post_nodes(
        self, base_name: str, input_nodes: list[OnnxNode],
        language_start_token_id: int, num_languages: int
    ) -> list[OnnxNode]:
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
        full_lm_head_logits = self._onnx_builder.build_conv(
            "model.decoder.embed_tokens", layer_norm2
        )
        language_logits = self._onnx_builder.build_op(
            "language_logits",
            [
                full_lm_head_logits,
                np.array([language_start_token_id], dtype=np.int32),
                np.array([language_start_token_id + num_languages], dtype=np.int32),
                np.array([1], dtype=np.int32),
            ],
            "Slice",
            output_names=["language_logits"]
        )
        detected_language_index = self._onnx_builder.build_op(
            "language_argmax", [language_logits], "ArgMax", axis=1, keepdims=1,
            output_names=["detected_language_index"]
        )
        return [detected_language_index, full_lm_head_logits]
