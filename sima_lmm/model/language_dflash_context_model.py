from dataclasses import dataclass

from afe.apis.defines import gen2_target
from afe.ir.defines import Status
from afe.ir.serializer import save_awesomenet
from afe.ir.tensor_type import TensorType

from sima_lmm.model.base import LayerConfiguration, LoraGenMode
from sima_lmm.model.language_pre_model import LanguagePreModel
from sima_lmm.model.sima_builder import SimaBuilder, activation_type


@dataclass
class LanguageDFlashContextModel(LanguagePreModel):
    """Project fused target hidden states into one DFlash layer's K/V cache."""

    def gen_onnx_files(self):
        self.create_onnx_builder()
        self._onnx_builder.create_input_node(
            "input", (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens)
        )
        rope_channels = (
            self.cfg.lm_cfg.rope_cfg.get_rope_dimension_count(self.layer_type) // 2
        )
        self._onnx_builder.create_input_node(
            "freq_real", (1, rope_channels, 1, self.num_tokens)
        )
        self._onnx_builder.create_input_node(
            "freq_imag", (1, rope_channels, 1, self.num_tokens)
        )
        attention_name = f"layers.{self.layer_idx}.self_attn"
        k_out = self._build_onnx_attn_key(
            attention_name, self._onnx_builder.input_nodes
        )
        v_out = self._build_onnx_attn_value(
            attention_name, self._onnx_builder.input_nodes[0]
        )
        shape = (
            1,
            self._head_dim,
            self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
            self.num_tokens,
        )
        for output in (k_out, v_out):
            self._onnx_builder.create_output_node(
                self._onnx_builder.get_node_output_name(output), shape
            )
        self._onnx_builder.create_and_save_model()
        self._onnx_builder = None

    def gen_model_sdk_files_directly(
        self,
        layer_cfg: LayerConfiguration,
        log_level: int,
        quantizable: bool,
    ):
        merged_lora = (
            layer_cfg.get("lora", LoraGenMode.LORA_DISABLED)
            == LoraGenMode.LORA_MERGED
        )
        graph = self._build_sima_nodes(quantizable, merged_lora)
        save_awesomenet(
            graph,
            self.model_name + (".fp32" if quantizable else ""),
            str(self.sima_model_sdk_path),
        )

    def _build_sima_nodes(self, quantizable: bool, merged_lora: bool = False):
        input_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        freq_shape = (
            1,
            1,
            self.num_tokens,
            self.cfg.lm_cfg.rope_cfg.get_rope_dimension_count(self.layer_type) // 2,
        )
        builder = SimaBuilder(
            Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target
        )
        model_inputs = [
            builder.create_placeholder_node(
                "input", TensorType(activation_type(quantizable), input_shape)
            ),
            builder.create_placeholder_node(
                "freq_real", TensorType(activation_type(quantizable), freq_shape)
            ),
            builder.create_placeholder_node(
                "freq_imag", TensorType(activation_type(quantizable), freq_shape)
            ),
        ]
        builder.begin_subnet(model_inputs)
        mla_input = builder.create_placeholder_node(
            "MLA_0/input", TensorType(activation_type(quantizable), input_shape)
        )
        mla_freq_real = builder.create_placeholder_node(
            "MLA_0/freq_real", TensorType(activation_type(quantizable), freq_shape)
        )
        mla_freq_imag = builder.create_placeholder_node(
            "MLA_0/freq_imag", TensorType(activation_type(quantizable), freq_shape)
        )
        attention_name = f"layers.{self.layer_idx}.self_attn"
        k_out = self._build_sima_attn_key(
            builder,
            attention_name,
            mla_input,
            mla_freq_real,
            mla_freq_imag,
            merged_lora,
        )
        v_out = self._build_sima_attn_value(
            builder, attention_name, mla_input, merged_lora
        )
        if self.cfg.pipeline_cfg.quantize_kv_cache:
            k_scale = builder.create_dynamic_quant_scale_node(k_out, per_token_quant=True)
            v_scale = builder.create_dynamic_quant_scale_node(v_out, per_token_quant=True)
            outputs = [
                builder.create_dynamic_quant_node(k_out, k_scale),
                k_scale,
                builder.create_dynamic_quant_node(v_out, v_scale),
                v_scale,
            ]
        else:
            outputs = [k_out, v_out]
        builder.create_tuple_node(outputs)
        mla_node = builder.finish_subnet("MLA_0")
        self._cast_bf16_outputs_to_fp32(builder, mla_node)
        return builder.finish(self.model_name)

    def get_mla_output_tessellate_params(self):
        return {
            index - 1: params
            for index, params in super().get_mla_output_tessellate_params().items()
        }
