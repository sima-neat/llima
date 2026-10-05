import numpy as np

from dataclasses import dataclass

from afe.ir.tensor_type import ScalarType

from sima_lmm.hf.hf_transformer import find_file
from sima_lmm.model.model_graph import ModelGraph
from sima_lmm.model.base import BaseModel, LayerConfiguration
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.tokenizer.whisper_tokenizer import get_tokenizer


@dataclass
class WhisperDecoderPostModel(BaseModel):
    """Implementation for the post cache model of Whisper.

    This implements a simplified version of the LanguagePostModel. This model is only used when
    generating new tokens so the num_tokens is assumed to be 1.

    Attributes:
        num_tokens: Number of tokens. Set to a value greater than 1 to consume multiple input tokens
            in one model.
        layer_idx: Transformer layer index.
        skip_encoder_kv_proj: Whether to skip the key/value projections in cross attention.
        output_encoder_kv_cache: Whether to output the key/value projections from cross attention.
        enable_log_probe: Whether to also output filtered logits from the final layer.
    """
    num_tokens: int
    layer_idx: int
    skip_encoder_kv_proj: bool
    output_encoder_kv_cache: bool
    enable_log_probe: bool = False

    def __post_init__(self):
        assert 0 <= self.layer_idx < self.cfg.decoder_layers

    @property
    def enable_filter_sharing(self) -> bool:
        return self.use_filter_sharing

    def gen_model_sdk_files_directly(
        self, layer_cfg: LayerConfiguration, log_level: int, quantizable: bool
    ):
        assert not self.output_encoder_kv_cache
        hidden_shape = (1, 1, self.num_tokens, self.cfg.d_model)
        cache_shape = (
            1,
            self.cfg.decoder_attention_heads,
            self.cfg.max_source_positions,
            self.cfg.decoder_head_dim,
        )
        shapes = {
            "input": hidden_shape,
            "self_attn": hidden_shape,
            "encoder_k_cache": cache_shape,
            "encoder_v_cache": cache_shape,
        }

        graph = ModelGraph(self, shapes, quantizable)
        outputs = self._build_sima_nodes(graph.raw, list(graph.inputs.values()), quantizable)
        graph.save(outputs)

    def _build_sima_transformer(self, builder, inputs, quantizable):
        graph = ModelGraph.from_builder(self, builder)
        name = f"model.decoder.layers.{self.layer_idx}"
        proj = graph.linear(f"{name}.self_attn.out_proj", inputs[1])
        hidden = builder.create_add_node(inputs[0], proj)
        norm = graph.layer_norm(f"{name}.encoder_attn_layer_norm", hidden)
        kv = inputs[2:4] if self.skip_encoder_kv_proj else [inputs[-1], inputs[-1]]
        queries = graph.project_heads(
            f"{name}.encoder_attn.q_proj",
            norm,
            self.cfg.decoder_attention_heads,
            scale=self.cfg.decoder_head_dim**-0.5,
            kv_len=self.cfg.max_source_positions,
        )
        kv_projs = []
        for proj, node in zip(("k_proj", "v_proj"), kv):
            if self.skip_encoder_kv_proj:
                heads = (
                    [node]
                    if len(queries) == 1
                    else [
                        builder.create_slice_node(node, [i], [i + 1], [1], [1])
                        for i in range(self.cfg.decoder_attention_heads)
                    ]
                )
            else:
                heads = graph.project_heads(
                    f"{name}.encoder_attn.{proj}",
                    node,
                    self.cfg.decoder_attention_heads,
                    kv_len=self.cfg.max_source_positions,
                    query_len=self.num_tokens,
                )
            kv_projs.append(heads)
        keys, values = kv_projs
        heads = []
        for query, key, value in zip(queries, keys, values):
            heads.append(graph.attention(query, key, value))
        attn = graph.project_merged_heads(
            f"{name}.encoder_attn.out_proj",
            heads,
            self.cfg.decoder_attention_heads,
        )
        hidden = builder.create_add_node(hidden, attn)
        norm = graph.layer_norm(f"{name}.final_layer_norm", hidden)
        hidden = graph.mlp(name, norm, self.cfg.activation_function, residual=hidden)
        return hidden, keys, values

    def _build_sima_nodes(self, builder, inputs, quantizable):
        graph = ModelGraph.from_builder(self, builder)
        hidden, keys, values = self._build_sima_transformer(builder, inputs, quantizable)
        if self.layer_idx < self.cfg.decoder_layers - 1:
            outputs = [hidden]
        else:
            norm = graph.layer_norm("model.decoder.layer_norm", hidden)
            logits = graph.linear("model.decoder.embed_tokens", norm)
            mask = np.zeros((1, 1, 1, self.cfg.vocab_size), dtype=np.float32)
            mask[..., self.cfg.suppress_tokens + self._get_extra_suppress_tokens()] = np.finfo(
                np.float32
            ).min
            mask = graph.constant(mask)
            logits = builder.create_add_node(logits, mask)
            outputs = [builder.create_argmax_node(logits, ScalarType.int32)]
            if self.enable_log_probe:
                outputs.append(logits)
        if self.output_encoder_kv_cache:
            assert len(keys) == len(values) == 1
            outputs.extend([keys[0], values[0]])
        return outputs

    def gen_onnx_files(self):
        base_name = f"model.decoder.layers.{self.layer_idx}"
        self.create_onnx_builder()
        self._onnx_builder.create_input_node("input", (1, self.cfg.d_model, 1, self.num_tokens))
        self._onnx_builder.create_input_node("self_attn", (1, self.cfg.d_model, 1, self.num_tokens))
        self._onnx_builder.create_input_node(
            "encoder_k_cache",
            (
                1,
                self.cfg.decoder_head_dim,
                self.cfg.decoder_attention_heads,
                self.cfg.max_source_positions,
            ),
        )
        self._onnx_builder.create_input_node(
            "encoder_v_cache",
            (
                1,
                self.cfg.decoder_head_dim,
                self.cfg.decoder_attention_heads,
                self.cfg.max_source_positions,
            ),
        )
        output_nodes = self._build_onnx_nodes(base_name, self._onnx_builder.input_nodes)
        output_name = self._onnx_builder.get_node_output_name(output_nodes[0])
        if self.layer_idx < self.cfg.decoder_layers - 1:
            self._onnx_builder.create_output_node(
                output_name, (1, self.cfg.d_model, 1, self.num_tokens)
            )
        else:
            self._onnx_builder.create_output_node(output_name, (1, 1, 1, self.num_tokens), np.int64)
            if self.enable_log_probe:
                self._onnx_builder.create_output_node(
                    self._onnx_builder.get_node_output_name(output_nodes[1]),
                    (1, self.cfg.vocab_size, 1, self.num_tokens)
                )
        assert not self.output_encoder_kv_cache
        self._onnx_builder.create_and_save_model()

        # Set to None to deallocate the memory.
        self._onnx_builder = None

    def _build_onnx_nodes(self, base_name: str, input_nodes: list[OnnxNode]) -> list[OnnxNode]:
        o_proj = self._onnx_builder.build_conv(f"{base_name}.self_attn.out_proj", input_nodes[1])
        add1 = self._onnx_builder.build_op(f"{base_name}.add1", [input_nodes[0], o_proj], "Add")
        encoder_attn_layer_norm = self._onnx_builder.build_layer_norm(
            f"{base_name}.encoder_attn_layer_norm", add1
        )
        if self.skip_encoder_kv_proj:
            encoder_attn_input_nodes = [encoder_attn_layer_norm, input_nodes[2], input_nodes[3]]
        else:
            encoder_attn_input_nodes = [encoder_attn_layer_norm, input_nodes[-1], input_nodes[-1]]
        encoder_attn, encoder_k_proj, encoder_v_proj = self._onnx_builder.build_attention(
            base_name=f"{base_name}.encoder_attn",
            input_nodes=encoder_attn_input_nodes,
            num_heads=self.cfg.decoder_attention_heads,
            head_dim=self.cfg.decoder_head_dim,
            seq_len=1,
            kv_len=self.cfg.max_source_positions,
            skip_kv_projs_and_split_head=self.skip_encoder_kv_proj,
            output_kv_projs=True,
        )
        add2 = self._onnx_builder.build_op(f"{base_name}.add2", [add1, encoder_attn], "Add")
        layer_norm1 = self._onnx_builder.build_layer_norm(f"{base_name}.final_layer_norm", add2)
        mlp = self._onnx_builder.build_encoder_decoder_mlp(
            base_name, layer_norm1, self.cfg.activation_function
        )
        add3 = self._onnx_builder.build_op(f"{base_name}.add3", [add2, mlp], "Add")

        output_nodes = list()
        if self.layer_idx < self.cfg.decoder_layers - 1:
            output_nodes.append(add3)
        else:
            # Include the operations after the last transformer layer into last post cache model.
            layer_norm2 = self._onnx_builder.build_layer_norm("model.decoder.layer_norm", add3)
            lm_head = self._onnx_builder.build_conv("model.decoder.embed_tokens", layer_norm2)

            suppress_tokens = self.cfg.suppress_tokens + self._get_extra_suppress_tokens()
            logit_mask = np.zeros((1, self.cfg.vocab_size, 1, 1), dtype=np.float32)
            logit_mask[:, suppress_tokens, :, :] = np.finfo(np.float32).min
            filtered_lm_head = self._onnx_builder.build_op(
                "filtered_lm_head", [lm_head, logit_mask], "Add"
            )
            argmax = self._onnx_builder.build_op(
                "argmax", [filtered_lm_head], "ArgMax", axis=1, keepdims=1
            )
            output_nodes.append(argmax)
            if self.enable_log_probe:
                output_nodes.append(filtered_lm_head)
        if self.output_encoder_kv_cache:
            output_nodes.append(encoder_k_proj)
            output_nodes.append(encoder_v_proj)
        return output_nodes

    def _get_extra_suppress_tokens(self) -> list[int]:
        hf_tokenizer_json_file = find_file(
            directory=self.hf_model.hf_cache, filename="tokenizer.json"
        )
        tokenizer = get_tokenizer(
            multilingual=True, num_languages=self.cfg.num_languages, language=None, task=None,
            hf_tokenizer_json_file=hf_tokenizer_json_file
        )
        return [
            220,  # standalone space
            tokenizer.no_timestamps,
            tokenizer.timestamp_begin,
        ]
