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
    """FC Fusion layer for the EAGLE3 draft model.

    Projects concatenated hidden states of shape (1, hidden_size * 3, 1, num_tokens)
    to (1, hidden_size, 1, num_tokens) using a single linear layer

    EAGLE3 conditions the draft model on hidden states from three specific layers of
    the target model (low, mid, and high), which are concatenated along the channel
    dimension to form a tensor of shape (1, hidden_size * 3, 1, num_tokens).
    """
    num_tokens: int

    def __post_init__(self):
        assert self.num_tokens >= 1

    def gen_onnx_files(self):
        base_name = self.hf_model.language_model_param_base_name
        self.create_onnx_builder()
        self._onnx_builder.create_input_node(
            "input", (1, self.cfg.lm_cfg.hidden_size * 3, 1, self.num_tokens)
        )
        output_node = self._onnx_builder.build_conv(
            f"fc", self._onnx_builder.input_nodes[0], is_fc=True
        )
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
        base_name = self.hf_model.language_model_param_base_name
        g = self._build_sima_nodes(base_name, quantizable)
        save_awesomenet(g, self.model_name + (".fp32" if quantizable else ""), str(self.sima_model_sdk_path))

    def _build_sima_nodes(self, base_name: str, quantizable: bool):
        input_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size * 3)
        output_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)

        builder = SimaBuilder(Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target)

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
