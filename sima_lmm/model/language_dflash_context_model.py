from dataclasses import dataclass

import numpy as np

from afe.apis.defines import TensorDRAMLayout, gen2_target
from afe.ir.defines import Status
from afe.ir.serializer import save_awesomenet
from afe.ir.tensor_type import ScalarType, TensorType

from sima_lmm.model.base import LayerConfiguration, TensorTessellateParameters
from sima_lmm.model.language_pre_model import LanguagePreModel
from sima_lmm.model.sima_builder import SimaBuilder, activation_type, build_conv


@dataclass
class LanguageDFlashContextModel(LanguagePreModel):
    """Fuse target hidden states and populate every DFlash draft K/V cache."""

    def __post_init__(self):
        super().__post_init__()
        assert self.is_dflash_draft

    def gen_onnx_files(self):
        raise NotImplementedError(
            "The combined DFlash context model is generated directly with Model SDK"
        )

    def gen_model_sdk_files_directly(
        self,
        layer_cfg: LayerConfiguration,
        log_level: int,
        quantizable: bool,
    ):
        graph = self._build_sima_nodes(quantizable)
        save_awesomenet(
            graph,
            self.model_name + (".fp32" if quantizable else ""),
            str(self.sima_model_sdk_path),
        )

    def _get_packed_projection_param(self, name: str):
        if name.endswith(".bias"):
            raise KeyError(name)
        assert name.endswith(".weight")
        params = []
        for projection in ("k_proj", "v_proj"):
            for layer_idx in range(self.cfg.lm_cfg.num_hidden_layers):
                params.append(
                    self.get_hf_param(
                        f"layers.{layer_idx}.self_attn.{projection}.weight"
                    )
                )
        if not isinstance(params[0], tuple):
            return np.concatenate(params, axis=0)

        scales = [param[0] for param in params]
        weights = [param[1] for param in params]
        metadata = params[0][2:]
        if any(param[2:] != metadata for param in params[1:]):
            raise ValueError("DFlash K/V projections use incompatible quantization metadata")
        return (
            np.concatenate(scales, axis=0),
            np.concatenate(weights, axis=0),
            *metadata,
        )

    @staticmethod
    def _check_packed_projection_param(name: str) -> bool:
        return name.endswith(".weight")

    def _build_sima_nodes(self, quantizable: bool):
        hidden_size = self.cfg.lm_cfg.hidden_size
        input_count = len(
            self.cfg.lm_cfg.speculative_decoding_cfg.target_layer_ids
        )
        num_layers = self.cfg.lm_cfg.num_hidden_layers
        num_heads = self.cfg.lm_cfg.attn_cfg.num_key_value_heads
        head_dim = self._head_dim
        kv_size = num_heads * head_dim
        hidden_shape = (1, 1, self.num_tokens, hidden_size)
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
                f"input_{index}",
                TensorType(activation_type(quantizable), hidden_shape),
            )
            for index in range(input_count)
        ]
        model_inputs.extend(
            [
                builder.create_placeholder_node(
                    "freq_real", TensorType(activation_type(quantizable), freq_shape)
                ),
                builder.create_placeholder_node(
                    "freq_imag", TensorType(activation_type(quantizable), freq_shape)
                ),
            ]
        )
        builder.begin_subnet(model_inputs)
        mla_hidden = [
            builder.create_placeholder_node(
                f"MLA_0/input_{index}",
                TensorType(activation_type(quantizable), hidden_shape),
            )
            for index in range(input_count)
        ]
        mla_freq_real = builder.create_placeholder_node(
            "MLA_0/freq_real", TensorType(activation_type(quantizable), freq_shape)
        )
        mla_freq_imag = builder.create_placeholder_node(
            "MLA_0/freq_imag", TensorType(activation_type(quantizable), freq_shape)
        )

        fused = build_conv(
            builder,
            self.get_hf_param,
            self.check_hf_param,
            "fc",
            builder.create_concat_node(mla_hidden, 3),
        )
        fused = self._build_sima_rms_norm(builder, "hidden_norm", fused)
        projected = build_conv(
            builder,
            self._get_packed_projection_param,
            self._check_packed_projection_param,
            "packed_kv",
            fused,
        )

        keys = []
        values = []
        for layer_idx in range(num_layers):
            key = builder.create_slice_node(
                projected,
                [layer_idx * kv_size],
                [(layer_idx + 1) * kv_size],
                [1],
                [3],
            )
            key = builder.create_slice_concat_node(
                key, axis=1, split_axis=3, split_block=num_heads, split_repeat=1
            )
            norm_name = f"layers.{layer_idx}.self_attn.k_norm"
            if self.check_hf_param(f"{norm_name}.weight"):
                key = self._build_sima_rms_norm(builder, norm_name, key)
            key = self._build_sima_rotary_emb(
                builder, key, mla_freq_real, mla_freq_imag
            )
            keys.append(key)

            value_begin = (num_layers + layer_idx) * kv_size
            value = builder.create_slice_node(
                projected, [value_begin], [value_begin + kv_size], [1], [3]
            )
            values.append(
                builder.create_slice_concat_node(
                    value,
                    axis=1,
                    split_axis=3,
                    split_block=num_heads,
                    split_repeat=1,
                )
            )

        packed = builder.create_concat_node(keys + values, 1)
        if self.cfg.pipeline_cfg.quantize_kv_cache:
            scale = builder.create_dynamic_quant_scale_node(
                packed, per_token_quant=True
            )
            outputs = [builder.create_dynamic_quant_node(packed, scale), scale]
        else:
            outputs = [packed]
        if len(outputs) > 1:
            builder.create_tuple_node(outputs)
        mla_node = builder.finish_subnet("MLA_0")
        if activation_type(quantizable) != ScalarType.float32:
            self._cast_bf16_outputs_to_fp32(builder, mla_node)
        return builder.finish(self.model_name)

    def get_mla_input_tessellate_params(self):
        return {}

    def get_mla_output_tessellate_params(self):
        heads = (
            2
            * self.cfg.lm_cfg.num_hidden_layers
            * self.cfg.lm_cfg.attn_cfg.num_key_value_heads
        )
        packed = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC16,
            dram_shape=(1, heads, self.cfg.pipeline_cfg.max_num_tokens, self._head_dim),
        )
        if not self.cfg.pipeline_cfg.quantize_kv_cache:
            return {0: packed}
        scale = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC16,
            dram_shape=(1, heads, self.cfg.pipeline_cfg.max_num_tokens, 1),
        )
        return {0: packed, 1: scale}
