import numpy as np
from dataclasses import dataclass

from afe.ir.tensor_type import TensorType, ScalarType

from sima_lmm.model.base import LoraGenMode, LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.model.model_graph import ModelGraph, save_model_graph
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import activation_type, create_channel_slice


@dataclass
class LanguageConvModel(LanguagePartBaseModel):
    """Fused conv layer (pre + cache + post) for LFM2.

    Inputs:
        - input: residual input (1, hidden, 1, num_tokens)
        - conv_cache: rolling window buffer (1, hidden, 1, L-1)

    Outputs:
        - hidden (1, hidden, 1, num_tokens) for non-last layer
        - if last layer, same as LanguagePostModel (argmax or split logits)
        - conv_cache_out:
            * grouped prefill (num_tokens > 1): full concat state
              (1, hidden, 1, num_tokens + L - 1), where concat = [conv_cache, bx]
            * decode (num_tokens == 1): rolling window state
              (1, hidden, 1, L - 1), i.e. concat[..., 1:]
    """

    num_tokens: int
    layer_idx: int
    final_softcapping: float | None

    def __post_init__(self):
        assert self.num_tokens >= 1
        assert 0 <= self.layer_idx < self.cfg.lm_cfg.num_hidden_layers
        assert (
            not self.cfg.lm_cfg.conv_bias
        ), "LanguageConvModel requires conv_bias=False due to missing padding mask logic."

    @property
    def enable_filter_sharing(self) -> bool:
        return self.cfg.pipeline_cfg.enable_filter_sharing

    def gen_onnx_files(self):
        base_layer = f"{self.hf_model.language_model_param_base_name}.layers.{self.layer_idx}"
        base_name = f"{base_layer}.conv"

        self.create_onnx_builder()
        self._onnx_builder.create_input_node(
            "input", (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens)
        )
        cache_shape = (1, self.cfg.lm_cfg.hidden_size, 1, self.cfg.lm_cfg.conv_L_cache - 1)
        output_cache_shape = (
            1,
            self.cfg.lm_cfg.hidden_size,
            1,
            self.num_tokens + self.cfg.lm_cfg.conv_L_cache - 2,
        )

        self._onnx_builder.create_input_node("conv_cache", cache_shape)

        output_nodes = self._build_onnx_nodes(base_layer, base_name, self._onnx_builder.input_nodes)

        out_name = self._onnx_builder.get_node_output_name(output_nodes[0])
        self._onnx_builder.create_output_node(
            out_name, (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens)
        )

        cache_out_name = self._onnx_builder.get_node_output_name(output_nodes[1])
        self._onnx_builder.create_output_node(cache_out_name, output_cache_shape)

        self._onnx_builder.create_and_save_model()
        self._onnx_builder = None

    def _build_onnx_nodes(
        self, base_layer: str, base_name: str, input_nodes: list[OnnxNode]
    ) -> list[OnnxNode]:

        norm_input = self._build_rms_norm(f"{base_layer}.operator_norm", input_nodes[0])
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "in_proj")
        in_proj = self._onnx_builder.build_conv_from_dense_with_lora(f"{base_name}.in_proj", norm_input, lora_rank=lora_rank)
        split = self._onnx_builder.build_op(
            f"{base_name}.in_proj.split",
            [in_proj],
            "Split",
            axis=1,
            output_names=[f"{base_name}.B", f"{base_name}.C", f"{base_name}.x"],
        )
        b = [split, 0]
        c = [split, 1]
        x = [split, 2]
        bx = self._onnx_builder.build_op(f"{base_name}.mul_bx", [b, x], "Mul")

        prev_last = input_nodes[1]
        tail = self._onnx_builder.build_op(
            f"{base_name}.tail.concat", [prev_last, bx], "Concat", axis=3
        )
        conv_cache_out = self._onnx_builder.build_op(
            f"{base_name}.tail.window",
            [
                tail,
                np.array([1], dtype=np.int64),
                np.array([self.num_tokens + self.cfg.lm_cfg.conv_L_cache - 1], dtype=np.int64),
                np.array([3], dtype=np.int64),
            ],
            "Slice",
        )

        w_raw = self.get_hf_param(f"{base_name}.conv.weight")
        if isinstance(w_raw, tuple):
            w_raw = w_raw[1]
        w_conv2d = w_raw.reshape(self.cfg.lm_cfg.hidden_size, 1, 1, self.cfg.lm_cfg.conv_L_cache)
        w_node = self._onnx_builder.create_initializer(f"{base_name}.conv.weight", value=w_conv2d)

        conv_inputs = [tail, w_node]
        if self.check_hf_param(f"{base_name}.conv.bias"):
            b_raw = self.get_hf_param(f"{base_name}.conv.bias")
            if isinstance(b_raw, tuple):
                b_raw = b_raw[1]
            b_node = self._onnx_builder.create_initializer(f"{base_name}.conv.bias", value=b_raw)
            conv_inputs.append(b_node)

        conv_out = self._onnx_builder.build_op(
            f"{base_name}.depthwise_conv2d",
            conv_inputs,
            "Conv",
            dilations=[1, 1],
            group=self.cfg.lm_cfg.hidden_size,
            kernel_shape=[1, self.cfg.lm_cfg.conv_L_cache],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        )

        gated = self._onnx_builder.build_op(f"{base_name}.gate", [conv_out, c], "Mul")
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "out_proj")
        out_proj = self._onnx_builder.build_conv_from_dense_with_lora(f"{base_name}.out_proj", gated, lora_rank=lora_rank)

        add1 = self._onnx_builder.build_op(f"{base_name}.add1", [input_nodes[0], out_proj], "Add")

        if self.layer_idx == self.cfg.lm_cfg.num_hidden_layers - 1:
            return [add1, conv_cache_out]

        rms_norm2 = self._build_rms_norm(f"{base_layer}.ffn_norm", add1)
        mlp = self._build_onnx_mlp(f"{base_layer}.feed_forward", [rms_norm2])
        add2 = self._onnx_builder.build_op(f"{base_name}.add2", [add1, mlp], "Add")

        return [add2, conv_cache_out]

    def gen_model_sdk_files_directly(
        self,
        layer_cfg: LayerConfiguration,
        log_level: int,
        quantizable: bool,
    ):
        base_layer = f"{self.hf_model.language_model_param_base_name}.layers.{self.layer_idx}"
        base_name = f"{base_layer}.conv"
        merged_lora = layer_cfg.get("lora", LoraGenMode.LORA_DISABLED) == LoraGenMode.LORA_MERGED
        g = self._build_sima_nodes(base_layer, base_name, quantizable, merged_lora)
        save_model_graph(self, g, quantizable)

    def _build_sima_nodes(self, base_layer: str, base_name: str, quantizable: bool, merged_lora: bool):
        hidden_size = self.cfg.lm_cfg.hidden_size

        input_shape = (1, 1, self.num_tokens, hidden_size)
        scale_shape = (1, 1, self.num_tokens, 1)
        cache_shape = (1, 1, self.cfg.lm_cfg.conv_L_cache - 1, hidden_size)

        input_specs = {"input": input_shape}
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            input_specs["input"] = TensorType(ScalarType.int8, input_shape)
            input_specs["input_scale"] = scale_shape
        input_specs["conv_cache"] = cache_shape
        graph = ModelGraph(self, input_specs, quantizable)
        builder = graph.raw
        mla_input_input = graph.inputs["input"]
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            mla_input_scale = graph.inputs["input_scale"]
        mla_input_conv_cache = graph.inputs["conv_cache"]
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            residual = graph.dequantize(mla_input_input, mla_input_scale)
        else:
            residual = mla_input_input

        norm_input = self._build_sima_rms_norm(builder, f"{base_layer}.operator_norm", residual)
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "in_proj")
        in_proj = graph.linear(
            f"{base_name}.in_proj", norm_input, lora_rank=lora_rank, merged_lora=merged_lora
        )

        b = create_channel_slice(builder, in_proj, 0, hidden_size)
        c = create_channel_slice(builder, in_proj, hidden_size, 2 * hidden_size)
        x = create_channel_slice(builder, in_proj, 2 * hidden_size, 3 * hidden_size)
        bx = builder.create_mul_node(b, x)

        tail = builder.create_concat_node([mla_input_conv_cache, bx], 2)
        conv_cache_out = builder.create_slice_node(
            tail,
            [1],
            [self.num_tokens + self.cfg.lm_cfg.conv_L_cache - 1],
            [1],
            [2],
        )

        conv_out = graph.conv(f"{base_name}.conv", tail, is_depthwise=True)

        gated = builder.create_mul_node(conv_out, c)
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "out_proj")
        out_proj = graph.linear(
            f"{base_name}.out_proj", gated, lora_rank=lora_rank, merged_lora=merged_lora
        )

        add1 = builder.create_add_node(residual, out_proj)

        if self.layer_idx == self.cfg.lm_cfg.num_hidden_layers - 1:
            outputs = [add1, conv_cache_out]
        else:
            rms_norm2 = self._build_sima_rms_norm(builder, f"{base_layer}.ffn_norm", add1)
            mlp = self._build_sima_mlp(
                builder, f"{base_layer}.feed_forward", [rms_norm2], quantizable, merged_lora
            )
            add2 = builder.create_add_node(add1, mlp)
            outputs = [add2, conv_cache_out]

        return graph.finish(outputs)
