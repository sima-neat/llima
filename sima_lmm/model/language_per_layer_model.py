from dataclasses import dataclass

import numpy as np

from afe.apis.defines import gen2_target
from afe.ir.defines import Status
from afe.ir.serializer import save_awesomenet
from afe.ir.tensor_type import TensorType, ScalarType

from sima_lmm.model.base import TensorTessellateParameters, LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import SimaBuilder, build_conv_from_dense_with_lora, activation_type


@dataclass
class LanguagePerLayerModel(LanguagePartBaseModel):
    """Gemma4 per-layer projection model.

    Computes per-layer residual inputs for all transformer layers in a single pass.

    IFM0: per-layer embedding staging buffer (1, L*H, 1, N) [NCHW]
        Rows gathered from embed_tokens_per_layer.weight by the CPU before inference.
    IFM1: input embeddings (1, hidden_size, 1, N) [NCHW]
    OFM:  per-layer inputs for all layers (1, H, 1, L*N) [NCHW]
        The W dimension is ordered layer-major, so the compiled NHWC layout is [L, N, H].

    L = num_hidden_layers, H = hidden_size_per_layer_input.
    """

    num_tokens: int

    def __post_init__(self):
        assert self.num_tokens >= 1
        assert self.cfg.lm_cfg.hidden_size_per_layer_input > 0, (
            "LanguagePerLayerModel requires hidden_size_per_layer_input > 0"
        )

    def gen_onnx_files(self):
        lm_base = self.hf_model.language_model_param_base_name
        L = self.cfg.lm_cfg.num_hidden_layers
        H = self.cfg.lm_cfg.hidden_size_per_layer_input

        self.create_onnx_builder()
        self._onnx_builder.create_input_node(
            "per_layer_emb_staging", (1, L * H, 1, self.num_tokens)
        )
        self._onnx_builder.create_input_node(
            "input", (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens)
        )

        output_node = self._build_onnx_per_layer_projection(
            lm_base, self._onnx_builder.input_nodes
        )
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_node),
            (1, H, 1, L * self.num_tokens),
        )

        self._onnx_builder.create_and_save_model()
        self._onnx_builder = None

    def gen_model_sdk_files_directly(
        self,
        layer_cfg: LayerConfiguration,
        log_level: int,
        quantizable: bool,
    ):
        del layer_cfg, log_level
        g = self._build_sima_nodes(
            self.hf_model.language_model_param_base_name,
            quantizable,
        )
        save_awesomenet(
            g,
            self.model_name + (".fp32" if quantizable else ""),
            str(self.sima_model_sdk_path),
        )

    def _build_onnx_per_layer_projection(
        self, lm_base: str, input_nodes: list[OnnxNode]
    ) -> OnnxNode:
        L = self.cfg.lm_cfg.num_hidden_layers

        proj = self._onnx_builder.build_conv_from_dense_with_lora(
            f"{lm_base}.per_layer_model_projection", input_nodes[1], None
        )
        proj = self._onnx_builder.build_op(
            f"{lm_base}.per_layer_proj_scale",
            [proj, self.cfg.lm_cfg.hidden_size ** -0.5],
            "Mul",
        )
        proj = self._onnx_builder.build_split_and_concat(
            f"{lm_base}.per_layer_proj_reshape", proj, L, split_axis=1, concat_axis=3
        )
        proj_normed = self._build_rms_norm(f"{lm_base}.per_layer_projection_norm", proj)

        emb = self._onnx_builder.build_split_and_concat(
            f"{lm_base}.per_layer_emb_reshape", input_nodes[0], L, split_axis=1, concat_axis=3
        )
        combined = self._onnx_builder.build_op(
            f"{lm_base}.per_layer_combine", [emb, proj_normed], "Add"
        )
        return self._onnx_builder.build_op(
            f"{lm_base}.per_layer_combine_scale", [combined, 2.0 ** -0.5], "Mul"
        )

    def _build_sima_nodes(self, lm_base: str, quantizable: bool):
        L = self.cfg.lm_cfg.num_hidden_layers
        H = self.cfg.lm_cfg.hidden_size_per_layer_input
        builder = SimaBuilder(Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target)

        model_input_staging = builder.create_placeholder_node(
            "per_layer_emb_staging",
            TensorType(activation_type(quantizable), (1, 1, self.num_tokens, L * H)),
        )
        model_input_input = builder.create_placeholder_node(
            "input",
            TensorType(
                activation_type(quantizable),
                (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size),
            ),
        )

        builder.begin_subnet([model_input_staging, model_input_input])
        mla_input_staging = builder.create_placeholder_node(
            "MLA_0/per_layer_emb_staging",
            TensorType(activation_type(quantizable), (1, 1, self.num_tokens, L * H)),
        )
        mla_input_input = builder.create_placeholder_node(
            "MLA_0/input",
            TensorType(
                activation_type(quantizable),
                (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size),
            ),
        )

        proj = build_conv_from_dense_with_lora(
            builder,
            self.get_hf_param,
            self.check_hf_param,
            f"{lm_base}.per_layer_model_projection",
            mla_input_input,
            None,
        )
        proj = builder.create_mul_node(
            proj,
            builder.create_constant_node(
                np.array([self.cfg.lm_cfg.hidden_size ** -0.5], dtype=np.float32)
            ),
        )
        proj = builder.create_slice_concat_node(
            proj,
            axis=2,
            split_axis=3,
            split_block=L,
            split_repeat=1,
        )
        proj_normed = self._build_sima_rms_norm(
            builder,
            f"{lm_base}.per_layer_projection_norm",
            proj,
        )

        emb = builder.create_slice_concat_node(
            mla_input_staging,
            axis=2,
            split_axis=3,
            split_block=L,
            split_repeat=1,
        )
        combined = builder.create_add_node(emb, proj_normed)
        _ = builder.create_mul_node(
            combined,
            builder.create_constant_node(np.array([2.0 ** -0.5], dtype=np.float32)),
        )

        mla_node = builder.finish_subnet("MLA_0")
        if activation_type(quantizable) != ScalarType.float32:
            builder.create_cast_node(mla_node, ScalarType.float32)
        return builder.finish(self.model_name)

    def get_mla_input_tessellate_params(self) -> dict:
        return {}

    def get_mla_output_tessellate_params(self) -> dict:
        return {}
