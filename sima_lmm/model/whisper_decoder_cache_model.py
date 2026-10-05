import numpy as np

from dataclasses import dataclass

from afe.apis.defines import gen2_target
from afe.backends.backends import Backend
from afe.ir.defines import Status, get_expected_tensor_value
from afe.ir.serializer import save_awesomenet
from afe.ir.tensor_type import ScalarType, TensorType

from sima_lmm.model.base import BaseModel, LayerConfiguration, TensorTessellateParameters
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import SimaBuilder, activation_type, activation_dtype


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

    def gen_model_sdk_files_directly(
        self, layer_cfg: LayerConfiguration, log_level: int, quantizable: bool
    ):
        shapes = {
            "input": (1, self.cfg.decoder_attention_heads, self.num_tokens, self.cfg.decoder_head_dim),
            "cached_keys": (1, 1, self.token_idx + self.num_tokens, self.cfg.d_model),
            "cached_values": (1, 1, self.token_idx + self.num_tokens, self.cfg.d_model),
        }
        if self.use_future_token_mask and self.num_tokens == 1:
            shapes["attn_mask"] = (1, 1, 1, self.token_idx + 1)

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
        key, value = [
            builder.create_slice_concat_node(
                node, axis=1, split_axis=3, split_block=self.cfg.decoder_attention_heads,
                split_repeat=1,
            )
            for node in inputs[1:3]
        ]
        scores = builder.create_einsum_node(inputs[0], key, "nhwc,nhqc->nhwq")
        if self.num_tokens > 1:
            mask = np.zeros((1, 1, self.num_tokens, self.token_idx + self.num_tokens), np.float32)
            for i in range(self.num_tokens):
                mask[:, :, i, self.token_idx + i + 1:] = np.finfo(np.float32).min
            mask = builder.create_constant_node(mask.astype(activation_dtype(quantizable)))
            scores = builder.create_add_node(scores, mask)
        elif self.use_future_token_mask:
            scores = builder.create_add_node(scores, inputs[3])
        probs = builder.create_softmax_node(scores, axis=3)
        attn = builder.create_einsum_node(probs, value, "nhwc,nhcq->nhwq")
        return [builder.create_slice_concat_node(
            attn, axis=3, split_axis=1, split_block=self.cfg.decoder_attention_heads, split_repeat=1
        )]

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
