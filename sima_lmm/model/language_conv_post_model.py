import numpy as np
from dataclasses import dataclass

from afe.apis.defines import gen2_target
from afe.ir.serializer import save_awesomenet
from afe.ir.defines import Status, TensorValue, TupleValue, get_expected_tensor_value
from afe.ir.tensor_type import TensorType, ScalarType
from afe.ir.build_node import NodeHandle, NodeOrHandle

from sima_lmm.model.base import LayerConfiguration, LoraGenMode, TensorTessellateParameters
from sima_lmm.model.language_part_base import LanguagePostBaseModel
from sima_lmm.model.sima_builder import (
    SimaBuilder, build_conv, build_conv_from_dense_with_lora,
    build_logit_softcapping, build_activation, activation_type
)
from sima_lmm.model.onnx_builder import OnnxNode


@dataclass
class LanguageConvPostModel(LanguagePostBaseModel):
    """Post-convolution model for the final layer of LFM2. 
    This reusable model is compiled for num_tokens=1 and contains the FFN and lm_head.
    """

    def __post_init__(self):
        assert self.num_tokens == 1, "LanguageConvPostModel only supports num_tokens=1"

    @property
    def enable_filter_sharing(self) -> bool:
        return self.cfg.pipeline_cfg.enable_filter_sharing

    @property
    def split_mlp(self) -> bool:
        return self.cfg.pipeline_cfg.split_mlp
        
    def gen_onnx_files(self):
        base_layer = f"{self.hf_model.language_model_param_base_name}.layers.{self.layer_idx}"
        self.create_onnx_builder()

        # This model takes one input: the intermediate hidden state from the final conv layer.
        self._onnx_builder.create_input_node(
            "input", (1, self.cfg.lm_cfg.hidden_size, 1, 1)
        )

        output_nodes = self._build_onnx_nodes(base_layer, self._onnx_builder.input_nodes)

        self._create_final_layer_output_nodes(output_nodes)

        self._onnx_builder.create_and_save_model()
        self._onnx_builder = None

    def _build_onnx_nodes(self, base_layer: str, input_nodes: list[OnnxNode]) -> list[OnnxNode]:
        # Input is the intermediate hidden state from the modified final conv layer.
        add1_input = input_nodes[0]

        # FFN block
        rms_norm2 = self._build_rms_norm(f"{base_layer}.ffn_norm", add1_input)
        mlp_base = f"{base_layer}.feed_forward" if self.check_hf_param(f"{base_layer}.feed_forward.w2.weight") else f"{base_layer}.mlp"
        mlp = self._build_onnx_mlp(mlp_base, [rms_norm2])
        add2 = self._onnx_builder.build_op(f"{base_layer}.conv.add2", [add1_input, mlp], "Add")

        # Final projection block (reuse the shared method from parent class)
        return self._build_onnx_post_transformer(base_layer, add2)

    def gen_model_sdk_files_directly(self, layer_cfg: LayerConfiguration, log_level: int, quantizable: bool):
        base_name = f"{self.hf_model.language_model_param_base_name}.layers.{self.layer_idx}"
        merged_lora = layer_cfg.get("lora", LoraGenMode.LORA_DISABLED) == LoraGenMode.LORA_MERGED
        g = self._build_sima_nodes(base_name, quantizable, merged_lora)
        save_awesomenet(g, self.model_name + (".fp32" if quantizable else ""), str(self.sima_model_sdk_path))

    def _build_sima_nodes(self, base_name: str, quantizable: bool, merged_lora: bool = False):
        input_shape = (1, 1, 1, self.cfg.lm_cfg.hidden_size)
        builder = SimaBuilder(Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target)
        model_input_input = builder.create_placeholder_node("input", TensorType(activation_type(quantizable), input_shape))
        builder.begin_subnet([model_input_input])
        mla_input_input = builder.create_placeholder_node("input", TensorType(activation_type(quantizable), input_shape))

        rms_norm2 = self._build_sima_rms_norm(builder, f"{base_name}.ffn_norm", mla_input_input)
        
        mlp_base = f"{base_name}.feed_forward" if self.check_hf_param(f"{base_name}.feed_forward.w2.weight") else f"{base_name}.mlp"
        mlp = self._build_sima_mlp(builder, mlp_base, [rms_norm2], quantizable, merged_lora=merged_lora)
        add2 = builder.create_add_node(mla_input_input, mlp)

        outputs = self._build_post_transformer(builder, add2, quantizable)

        if len(outputs) > 1:
            tuple_node = builder.create_tuple_node(outputs)

        mla_node = builder.finish_subnet("MLA_0")

        self._cast_bf16_outputs_to_fp32(builder, mla_node)

        net = builder.finish(self.model_name)
        return net

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """
        Get the custom tessellate params for model's inputs on the MLA.
        """
        # Use default tessellate params.
        return {}

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """
        Get the custom tessellate params for model's output on the MLA.
        """
        # Use default tessellate params.
        return {}