import numpy as np

from dataclasses import dataclass

from afe.apis.defines import gen2_target
from afe.backends.backends import Backend
from afe.ir.defines import Status, get_expected_tensor_value
from afe.ir.serializer import save_awesomenet
from afe.ir.tensor_type import ScalarType, TensorType

from sima_lmm.hf.hf_transformer import find_file
from sima_lmm.model.base import BaseModel, LayerConfiguration, TensorTessellateParameters
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import (
    SimaBuilder, activation_type, activation_dtype, build_conv, build_two_stage_layer_norm,
    build_activation, build_matmul_and_split_heads, build_merge_heads_and_matmul,
)
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
            1, self.cfg.decoder_attention_heads, self.cfg.max_source_positions,
            self.cfg.decoder_head_dim,
        )
        shapes = {
            "input": hidden_shape, "self_attn": hidden_shape,
            "encoder_k_cache": cache_shape, "encoder_v_cache": cache_shape,
        }

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

    def _build_sima_transformer(self, builder, inputs, quantizable):
        name = f"model.decoder.layers.{self.layer_idx}"
        proj = build_conv(
            builder, self.get_hf_param, self.check_hf_param, f"{name}.self_attn.out_proj", inputs[1]
        )
        hidden = builder.create_add_node(inputs[0], proj)
        norm = build_two_stage_layer_norm(
            builder, self.get_hf_param, self.check_hf_param,
            f"{name}.encoder_attn_layer_norm", hidden, axis=-1, epsilon=float(np.float32(1e-5)),
        )
        kv = inputs[2:4] if self.skip_encoder_kv_proj else [inputs[-1], inputs[-1]]
        queries = build_matmul_and_split_heads(
            builder, self.get_hf_param, self.check_hf_param, f"{name}.encoder_attn.q_proj", norm,
            self.cfg.decoder_attention_heads, self.num_tokens,
            post_matmul_scale=self.cfg.decoder_head_dim ** -0.5, kv_len=self.cfg.max_source_positions,
        )
        kv_projs = []
        for proj, node in zip(("k_proj", "v_proj"), kv):
            if self.skip_encoder_kv_proj:
                heads = [node] if len(queries) == 1 else [
                    builder.create_slice_node(node, [i], [i + 1], [1], [1])
                    for i in range(self.cfg.decoder_attention_heads)
                ]
            else:
                heads = build_matmul_and_split_heads(
                    builder, self.get_hf_param, self.check_hf_param,
                    f"{name}.encoder_attn.{proj}", node, self.cfg.decoder_attention_heads,
                    self.num_tokens, kv_len=self.cfg.max_source_positions,
                )
            kv_projs.append(heads)
        keys, values = kv_projs
        heads = []
        for query, key, value in zip(queries, keys, values):
            scores = builder.create_einsum_node(query, key, "nhwc,nhqc->nhwq")
            probs = builder.create_softmax_node(scores, axis=3)
            heads.append(builder.create_einsum_node(probs, value, "nhwc,nhcq->nhwq"))
        attn = build_merge_heads_and_matmul(
            builder, self.get_hf_param, self.check_hf_param,
            f"{name}.encoder_attn.out_proj", heads, self.cfg.decoder_attention_heads,
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
        return hidden, keys, values

    def _build_sima_nodes(self, builder, inputs, quantizable):
        hidden, keys, values = self._build_sima_transformer(builder, inputs, quantizable)
        if self.layer_idx < self.cfg.decoder_layers - 1:
            outputs = [hidden]
        else:
            norm = build_two_stage_layer_norm(
                builder, self.get_hf_param, self.check_hf_param,
                "model.decoder.layer_norm", hidden, axis=-1, epsilon=float(np.float32(1e-5)),
            )
            logits = build_conv(
                builder, self.get_hf_param, self.check_hf_param, "model.decoder.embed_tokens", norm
            )
            mask = np.zeros((1, 1, 1, self.cfg.vocab_size), dtype=np.float32)
            mask[..., self.cfg.suppress_tokens + self._get_extra_suppress_tokens()] = np.finfo(np.float32).min
            mask = builder.create_constant_node(mask.astype(activation_dtype(quantizable)))
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
