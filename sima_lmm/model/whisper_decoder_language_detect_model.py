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


from dataclasses import dataclass, replace

from afe.ir import build_node
from afe.ir.execute import create_node_executor
from afe.ir.net import AwesomeNet
from afe.ir.operations import AddActivationOp, ConvAddActivationOp

from sima_lmm.hf.hf_transformer import find_file
from sima_lmm.model.model_graph import ModelGraph
from sima_lmm.model.base import BaseModel, LayerConfiguration
from sima_lmm.model.whisper_decoder_cache_model import WhisperDecoderCacheModel
from sima_lmm.model.whisper_decoder_post_model import WhisperDecoderPostModel
from sima_lmm.model.whisper_decoder_pre_model import WhisperDecoderPreModel
from sima_lmm.tokenizer.whisper_tokenizer import get_tokenizer


@dataclass
class WhisperDecoderLanguageDetectModel(BaseModel):
    """One-shot Whisper decoder graph for automatic language detection."""

    NUM_TOKENS = 1
    TOKEN_IDX = 0

    def generate_graph(
        self, layer_cfg: LayerConfiguration, quantizable: bool
    ):
        shapes = {"audio_features": (1, 1, self.cfg.max_source_positions, self.cfg.d_model)}

        graph = ModelGraph(self, shapes, quantizable)
        inputs = list(graph.inputs.values())
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
            pre_outputs = pre._build_nodes(graph, hidden)
            residual = pre_outputs[pre.positioned_residual_output_idx] if idx == 0 else hidden[0]
            attn = cache._build_nodes(graph, pre_outputs)[0]
            post = WhisperDecoderPostModel(
                self.cfg,
                self.model_name,
                hf_model=self.hf_model,
                num_tokens=1,
                layer_idx=idx,
                skip_encoder_kv_proj=False,
                output_encoder_kv_cache=False,
            )
            output, _, _ = post._build_transformer(
                graph, [residual, attn, inputs[0]]
            )
            hidden = [output]
        norm = graph.layer_norm("model.decoder.layer_norm", hidden[0])
        logits = graph.linear("model.decoder.embed_tokens", norm)
        language_logits = graph.slice(logits, start=start, stop=start + count, axis=3)
        graph.save([graph.argmax(language_logits), logits], transform_subnet=self._fold_sot_prefix if graph.quantizable else None)

    @staticmethod
    def _fold_sot_prefix(net: AwesomeNet):
        """Fold the constant start-token branch in FP32 before quantization."""
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
