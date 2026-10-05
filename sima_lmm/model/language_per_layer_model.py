from dataclasses import dataclass

import numpy as np

from afe.ir.tensor_type import TensorType, ScalarType

from sima_lmm.model.base import LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.model.model_graph import ModelGraph, save_model_graph
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import activation_dtype


@dataclass
class LanguagePerLayerModel(LanguagePartBaseModel):
    """Gemma4 per-layer projection model.

    Computes per-layer residual inputs for all transformer layers in a single pass.

    Without embedding quantization, the IFMs are the per-layer embedding staging buffer and normal
    input embeddings. With quantization, each embedding IFM is immediately followed by its scale.
    Rows are gathered from embed_tokens_per_layer.weight by the CPU before inference.
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
        save_model_graph(self, g, quantizable)

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
        staging_shape = (1, 1, self.num_tokens, L * H)
        input_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        scale_shape = (1, 1, self.num_tokens, 1)
        input_specs = {"per_layer_emb_staging": staging_shape}
        if self.cfg.pipeline_cfg.quantize_embeddings:
            input_specs["per_layer_emb_staging"] = TensorType(ScalarType.int8, staging_shape)
            input_specs["per_layer_emb_staging_scale"] = scale_shape
        input_specs["input"] = (
            TensorType(ScalarType.int8, input_shape)
            if self.uses_quantized_input_embeddings else input_shape
        )
        if self.cfg.pipeline_cfg.quantize_embeddings:
            input_specs["input_scale"] = scale_shape
        graph = ModelGraph(self, input_specs, quantizable)
        builder = graph.raw
        mla_input_staging = graph.inputs["per_layer_emb_staging"]
        if self.cfg.pipeline_cfg.quantize_embeddings:
            mla_input_staging_scale = graph.inputs["per_layer_emb_staging_scale"]
        mla_input_input = graph.inputs["input"]
        if self.cfg.pipeline_cfg.quantize_embeddings:
            mla_input_input_scale = graph.inputs["input_scale"]

        if self.uses_quantized_input_embeddings:
            projection_input = graph.dequantize(mla_input_input, mla_input_input_scale)
        else:
            projection_input = mla_input_input
        proj = graph.linear(f"{lm_base}.per_layer_model_projection", projection_input, lora_rank=None)
        proj = builder.create_mul_node(
            proj,
            graph.constant(
                np.array(
                    [self.cfg.lm_cfg.hidden_size**-0.5],
                    dtype=activation_dtype(quantizable),
                )
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

        if self.cfg.pipeline_cfg.quantize_embeddings:
            staging = graph.dequantize(mla_input_staging, mla_input_staging_scale)
        else:
            staging = mla_input_staging
        emb = builder.create_slice_concat_node(
            staging,
            axis=2,
            split_axis=3,
            split_block=L,
            split_repeat=1,
        )
        combined = builder.create_add_node(emb, proj_normed)
        output = builder.create_mul_node(
            combined,
            graph.constant(np.array([2.0**-0.5], dtype=activation_dtype(quantizable))),
        )

        return graph.finish([output])
