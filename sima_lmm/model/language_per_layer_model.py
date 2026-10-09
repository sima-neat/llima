from dataclasses import dataclass

import numpy as np

from sima_lmm.model.base import LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.model.model_graph import ModelGraph


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

    def generate_graph(
        self,
        layer_cfg: LayerConfiguration,
        quantizable: bool,
    ):
        del layer_cfg
        lm_base = self.hf_model.language_model_param_base_name
        L = self.cfg.lm_cfg.num_hidden_layers
        H = self.cfg.lm_cfg.hidden_size_per_layer_input
        staging_shape = (1, 1, self.num_tokens, L * H)
        input_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        scale_shape = (1, 1, self.num_tokens, 1)
        input_specs = {"per_layer_emb_staging": staging_shape}
        input_dtypes = {}
        if self.cfg.pipeline_cfg.quantize_embeddings:
            input_dtypes["per_layer_emb_staging"] = np.int8
            input_specs["per_layer_emb_staging_scale"] = scale_shape
        input_specs["input"] = input_shape
        if self.uses_quantized_input_embeddings:
            input_dtypes["input"] = np.int8
        if self.cfg.pipeline_cfg.quantize_embeddings:
            input_specs["input_scale"] = scale_shape
        graph = ModelGraph(self, input_specs, quantizable, input_dtypes=input_dtypes)
        mla_input_staging = graph.inputs["per_layer_emb_staging"]
        if self.cfg.pipeline_cfg.quantize_embeddings:
            mla_input_staging_scale = graph.inputs["per_layer_emb_staging_scale"]
        mla_input_input = graph.inputs["input"]
        if self.cfg.pipeline_cfg.quantize_embeddings:
            mla_input_input_scale = graph.inputs["input_scale"]

        if self.uses_quantized_input_embeddings:
            projection_input = graph.dequant(mla_input_input, mla_input_input_scale)
        else:
            projection_input = mla_input_input
        proj = graph.linear(f"{lm_base}.per_layer_model_projection", projection_input, lora_rank=None)
        proj = graph.mul(
            proj,
            graph.constant([self.cfg.lm_cfg.hidden_size**-0.5]),
        )
        proj = graph.split_concat(
            proj,
            axis=2,
            split_axis=3,
            split_block=L,
            split_repeat=1,
        )
        proj_normed = graph.rms_norm(f"{lm_base}.per_layer_projection_norm", proj)

        if self.cfg.pipeline_cfg.quantize_embeddings:
            staging = graph.dequant(mla_input_staging, mla_input_staging_scale)
        else:
            staging = mla_input_staging
        emb = graph.split_concat(
            staging,
            axis=2,
            split_axis=3,
            split_block=L,
            split_repeat=1,
        )
        combined = graph.add(emb, proj_normed)
        output = graph.mul(
            combined,
            graph.constant([2.0**-0.5]),
        )

        graph.save([output])
