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

from afe.ir.serializer import save_awesomenet
from afe.ir.defines import Status
from afe.apis.defines import gen2_target, TensorDRAMLayout
from afe.ir.tensor_type import TensorType, ScalarType

from sima_lmm.model.base import TensorTessellateParameters, LoraGenMode, LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import (
    SimaBuilder, build_conv, build_conv_from_dense_with_lora,
    build_activation, activation_type, activation_dtype,
)


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

    @property
    def split_mlp(self) -> bool:
        return self.cfg.pipeline_cfg.split_mlp

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
        save_awesomenet(g, self.model_name + (".fp32" if quantizable else ""), str(self.sima_model_sdk_path))

    def _build_sima_nodes(self, base_layer: str, base_name: str, quantizable: bool, merged_lora: bool):
        hidden_size = self.cfg.lm_cfg.hidden_size

        input_shape = (1, 1, self.num_tokens, hidden_size)
        cache_shape = (1, 1, self.cfg.lm_cfg.conv_L_cache - 1, hidden_size)

        builder = SimaBuilder(Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target)

        model_input_input = builder.create_placeholder_node(
            "input", TensorType(activation_type(quantizable), input_shape)
        )
        model_input_conv_cache = builder.create_placeholder_node(
            "conv_cache", TensorType(activation_type(quantizable), cache_shape)
        )

        builder.begin_subnet([model_input_input, model_input_conv_cache])

        mla_input_input = builder.create_placeholder_node(
            "input", TensorType(activation_type(quantizable), input_shape)
        )
        mla_input_conv_cache = builder.create_placeholder_node(
            "conv_cache", TensorType(activation_type(quantizable), cache_shape)
        )

        norm_input = self._build_sima_rms_norm(
            builder, f"{base_layer}.operator_norm", mla_input_input
        )
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "in_proj")
        in_proj = build_conv_from_dense_with_lora(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.in_proj", norm_input, lora_rank=lora_rank, merged_lora=merged_lora
        )

        b = builder.create_slice_node(in_proj, [0], [hidden_size], [1], [3])
        c = builder.create_slice_node(in_proj, [hidden_size], [2 * hidden_size], [1], [3])
        x = builder.create_slice_node(in_proj, [2 * hidden_size], [3 * hidden_size], [1], [3])
        bx = builder.create_mul_node(b, x)

        tail = builder.create_concat_node([mla_input_conv_cache, bx], 2)
        conv_cache_out = builder.create_slice_node(
            tail,
            [1],
            [self.num_tokens + self.cfg.lm_cfg.conv_L_cache - 1],
            [1],
            [2],
        )

        conv_out = build_conv(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.conv", tail,
            is_fc=False, is_depthwise=True
        )

        gated = builder.create_mul_node(conv_out, c)
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "out_proj")
        out_proj = build_conv_from_dense_with_lora(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.out_proj", gated, lora_rank=lora_rank, merged_lora=merged_lora
        )

        add1 = builder.create_add_node(mla_input_input, out_proj)

        if self.layer_idx == self.cfg.lm_cfg.num_hidden_layers - 1:
            _ = builder.create_tuple_node([add1, conv_cache_out])
        else:
            rms_norm2 = self._build_sima_rms_norm(builder, f"{base_layer}.ffn_norm", add1)
            mlp = self._build_sima_mlp(
                builder, f"{base_layer}.feed_forward", [rms_norm2], quantizable, merged_lora
            )
            add2 = builder.create_add_node(add1, mlp)
            _ = builder.create_tuple_node([add2, conv_cache_out])

        mla_node = builder.finish_subnet("MLA_0")

        tuple_items = builder.create_tuple_get_item_nodes(mla_node)
        if activation_type(quantizable) == ScalarType.float32:
            builder.create_tuple_node(tuple_items)
        else:
            builder.create_tuple_node(
                [builder.create_cast_node(item, ScalarType.float32) for item in tuple_items]
            )

        net = builder.finish(self.model_name)
        return net

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """Provide tessellation for fused conv inputs to satisfy MLA alignment."""
        input_tess = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=None,
        )
        cache_tess = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=None,
        )
        return {0: input_tess, 1: cache_tess}

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """Provide tessellation for fused conv outputs to satisfy MLA alignment."""
        main_out = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=None,
        )
        cache_out = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=None,
        )
        # Outputs order: [main, conv_cache_out]
        return {0: main_out, 1: cache_out}
