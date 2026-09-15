from dataclasses import dataclass

from afe.apis.defines import gen2_target
from afe.backends.backends import Backend
from afe.ir.serializer import save_awesomenet
from afe.ir.defines import Status, get_expected_tensor_value
from afe.ir.tensor_type import TensorType, ScalarType

from sima_lmm.model.base import LayerConfiguration, TensorTessellateParameters
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.model.sima_builder import (
    SimaBuilder, build_conv, activation_type
)

@dataclass
class LanguageDraftFCModel(LanguagePartBaseModel):
    """Target-context fusion layer for a speculative draft model.

    EAGLE3 projects one concatenated 3H tensor. DFlash accepts eight target-layer
    tensors, concatenates them on device, projects 8H to H, and applies hidden_norm.
    """
    num_tokens: int

    def __post_init__(self):
        assert self.num_tokens >= 1

    def gen_onnx_files(self):
        self.create_onnx_builder()
        input_count = len(self.cfg.lm_cfg.speculative_decoding_cfg.target_layer_ids)
        if self.is_dflash:
            for index in range(input_count):
                self._onnx_builder.create_input_node(
                    f"input_{index}",
                    (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens),
                )
            fc_input = self._onnx_builder.build_op(
                "target_hidden.concat", self._onnx_builder.input_nodes, "Concat", axis=1
            )
        else:
            self._onnx_builder.create_input_node(
                "input", (1, self.cfg.lm_cfg.hidden_size * 3, 1, self.num_tokens)
            )
            fc_input = self._onnx_builder.input_nodes[0]
        output_node = self._onnx_builder.build_conv(
            "fc", fc_input, is_fc=True
        )
        if self.is_dflash:
            output_node = self._build_rms_norm("hidden_norm", output_node)
        output_name = self._onnx_builder.get_node_output_name(output_node)
        self._onnx_builder.create_output_node(
            output_name, (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens)
        )
        self._onnx_builder.create_and_save_model()

        # Set to None to deallocate memory
        self._onnx_builder = None

    def gen_model_sdk_files_directly(
        self,
        layer_cfg: LayerConfiguration,
        log_level: int,
        quantizable: bool,
    ):
        g = self._build_sima_nodes(quantizable)
        save_awesomenet(g, self.model_name + (".fp32" if quantizable else ""), str(self.sima_model_sdk_path))

    def _build_sima_nodes(self, quantizable: bool):
        input_count = len(self.cfg.lm_cfg.speculative_decoding_cfg.target_layer_ids)
        input_channels = self.cfg.lm_cfg.hidden_size * (input_count if self.is_dflash else 3)
        input_shape = (1, 1, self.num_tokens, input_channels)
        dflash_input_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        output_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)

        builder = SimaBuilder(Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target)

        if self.is_dflash:
            model_inputs = [
                builder.create_placeholder_node(
                    f"input_{index}",
                    TensorType(activation_type(quantizable), dflash_input_shape),
                )
                for index in range(input_count)
            ]
            builder.begin_subnet(model_inputs)
            mla_inputs = [
                builder.create_placeholder_node(
                    f"MLA_0/input_{index}",
                    TensorType(activation_type(quantizable), dflash_input_shape),
                )
                for index in range(input_count)
            ]
            mla_input = builder.create_concat_node(mla_inputs, 3)
        else:
            model_input = builder.create_placeholder_node(
                "input", TensorType(activation_type(quantizable), input_shape)
            )
            builder.begin_subnet([model_input])
            mla_input = builder.create_placeholder_node(
                "MLA_0/input", TensorType(activation_type(quantizable), input_shape)
            )
        output = build_conv(
            builder, self.get_hf_param, self.check_hf_param, "fc", mla_input
        )
        if self.is_dflash:
            output = self._build_sima_rms_norm(builder, "hidden_norm", output)
        assert get_expected_tensor_value(output.get_type().output).shape == output_shape

        mla_node = builder.finish_subnet("MLA_0")
        if activation_type(quantizable) != ScalarType.float32:
            _ = builder.create_cast_node(mla_node, ScalarType.float32, backend=Backend.EV)
        return builder.finish(self.model_name)

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
