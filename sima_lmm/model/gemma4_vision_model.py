#########################################################
# Copyright (C) 2026 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################
from dataclasses import dataclass

import numpy as np

from afe.apis.defines import TensorDRAMLayout
from sima_lmm.model.base import BaseModel, TensorTessellateParameters
from sima_lmm.model.onnx_builder import OnnxNode


@dataclass
class Gemma4VisionLayerModel(BaseModel):
    """Gemma4 Vision model implementation.

    Architecture differences from SigLIP2/LFM2:
    - 2D learned position embeddings indexed by (x, y) patch coordinates
    - 2D RoPE on Q and K (head_dim split into x-half and y-half)
    - 4 RMSNorms per encoder layer (input, post-attn, pre-ffn, post-ffn)
    - Separate Q, K, V projections with per-QKV norms (q_norm, k_norm, weightless v_norm)
    - Attention scale = 1.0 (no head_dim scaling)
    - AveragePool spatial pooler followed by weightless RMSNorm + linear projection
    - Pixel scaling 2*(x - 0.5) absorbed into input_proj weights at graph-creation time
    """

    layer_idx: int
    num_layers: int
    include_embeddings: bool
    include_mm_proj: bool

    def gen_onnx_files(self):
        self.create_onnx_builder()

        patch_feature_size = 3 * self.cfg.vm_cfg.patch_size * self.cfg.vm_cfg.patch_size
        self._onnx_builder.create_input_node("input", (1, patch_feature_size, 1, self.cfg.vm_cfg.seq_len))

        output_nodes = self._build_onnx_nodes(self.hf_model.vision_model_param_base_name,
                                               self._onnx_builder.input_nodes)

        for node in output_nodes:
            self._onnx_builder.create_output_node(
                self._onnx_builder.get_node_output_name(node),
                (1, self.cfg.lm_cfg.hidden_size, 1, self.cfg.mm_cfg.mm_tokens_per_image)
            )

        self._onnx_builder.create_and_save_model()
        self._onnx_builder = None

    def _build_onnx_nodes(self, base_name: str, input_nodes: list[OnnxNode]) -> list[OnnxNode]:

        if isinstance(self.cfg.vm_cfg.image_size, list):
            image_h, image_w = self.cfg.vm_cfg.image_size
        else:
            image_h = image_w = self.cfg.vm_cfg.image_size

        grid_h = image_h // self.cfg.vm_cfg.patch_size
        grid_w = image_w // self.cfg.vm_cfg.patch_size

        # Precompute all constants needed for the encoder.
        (pos_embed_node,
         rope_cos_x_node, rope_sin_x_node,
         rope_cos_y_node, rope_sin_y_node) = self._precompute_constants(
            base_name, grid_h, grid_w
        )

        if self.include_embeddings:
            x = self._build_patch_embedder(base_name, input_nodes[0], pos_embed_node)
        else:
            x = input_nodes[0]

        for i in range(self.layer_idx, self.layer_idx + self.num_layers):
            layer_base = f"{base_name}.encoder.layers.{i}"
            x = self._build_encoder_layer(
                layer_base, x,
                rope_cos_x_node, rope_sin_x_node,
                rope_cos_y_node, rope_sin_y_node
            )

        if self.include_mm_proj:
            x = self._build_pooler(base_name, x, grid_h)
            x = self._build_multimodal_embedder(base_name, x)

        return [x]


    def _precompute_constants(
        self, base_name: str, grid_h: int, grid_w: int
    ) -> tuple[OnnxNode, OnnxNode, OnnxNode, OnnxNode, OnnxNode]:
        """Precompute position embeddings and 2D RoPE tables as ONNX initializers."""

        x_coords = np.tile(np.arange(grid_w, dtype=np.int64), grid_h)
        y_coords = np.repeat(np.arange(grid_h, dtype=np.int64), grid_w)

        pos_table = self._onnx_builder.get_param_func(
            f"{base_name}.patch_embedder.position_embedding_table"
        )
        pos_embed = pos_table[0][x_coords] + pos_table[1][y_coords]
        pos_embed = pos_embed.T[None, :, None, :].astype(np.float32)

        pos_embed_node = self._onnx_builder.create_initializer(
            f"{base_name}.pos_embed_static", pos_embed
        )

        rope_quarter_dim = self.cfg.vm_cfg.hidden_size // self.cfg.vm_cfg.num_attention_heads // 4
        inv_freq = 1.0 / (
            self.cfg.vm_cfg.rope_theta ** (np.arange(0, rope_quarter_dim, dtype=np.float32) / rope_quarter_dim)
        )

        cos_x = np.cos(np.outer(x_coords, inv_freq)).T.reshape(1, rope_quarter_dim, 1, self.cfg.vm_cfg.seq_len).astype(np.float32)
        sin_x = np.sin(np.outer(x_coords, inv_freq)).T.reshape(1, rope_quarter_dim, 1, self.cfg.vm_cfg.seq_len).astype(np.float32)
        cos_y = np.cos(np.outer(y_coords, inv_freq)).T.reshape(1, rope_quarter_dim, 1, self.cfg.vm_cfg.seq_len).astype(np.float32)
        sin_y = np.sin(np.outer(y_coords, inv_freq)).T.reshape(1, rope_quarter_dim, 1, self.cfg.vm_cfg.seq_len).astype(np.float32)

        rope_cos_x = self._onnx_builder.create_initializer(f"{base_name}.rope_cos_x", cos_x)
        rope_sin_x = self._onnx_builder.create_initializer(f"{base_name}.rope_sin_x", sin_x)
        rope_cos_y = self._onnx_builder.create_initializer(f"{base_name}.rope_cos_y", cos_y)
        rope_sin_y = self._onnx_builder.create_initializer(f"{base_name}.rope_sin_y", sin_y)

        return pos_embed_node, rope_cos_x, rope_sin_x, rope_cos_y, rope_sin_y

    def _build_patch_embedder(
        self, base_name: str, input_node: OnnxNode, pos_embed_node: OnnxNode
    ) -> OnnxNode:
        """Match HF Gemma4 patch embedder literally: 2 * (x - 0.5), proj, add position."""
        sub = self._onnx_builder.build_op(
            f"{base_name}.patch_embedder.sub_half",
            [input_node, 0.5],
            "Sub",
        )
        scaled = self._onnx_builder.build_op(
            f"{base_name}.patch_embedder.mul_two",
            [sub, 2.0],
            "Mul",
        )
        proj = self._onnx_builder.build_conv(
            f"{base_name}.patch_embedder.input_proj",
            scaled,
            is_fc=True,
        )
        return self._onnx_builder.build_op(
            f"{base_name}.patch_embedder.add_pos_embed",
            [proj, pos_embed_node],
            "Add"
        )
    
    def _build_encoder_layer(
        self, base_name: str, input_node: OnnxNode,
        rope_cos_x: OnnxNode, rope_sin_x: OnnxNode,
        rope_cos_y: OnnxNode, rope_sin_y: OnnxNode,
    ) -> OnnxNode:

        x = self._onnx_builder.build_rms_norm(
            f"{base_name}.input_layernorm", input_node, self.cfg.vm_cfg.layer_norm_eps, 0.0
        )
        x = self._build_attention(base_name, x, rope_cos_x, rope_sin_x, rope_cos_y, rope_sin_y)
        x = self._onnx_builder.build_rms_norm(
            f"{base_name}.post_attention_layernorm", x, self.cfg.vm_cfg.layer_norm_eps, 0.0
        )
        x = self._onnx_builder.build_op(f"{base_name}.add1", [input_node, x], "Add")

        residual = x
        x = self._onnx_builder.build_rms_norm(
            f"{base_name}.pre_feedforward_layernorm", x, self.cfg.vm_cfg.layer_norm_eps, 0.0
        )
        x = self._build_mlp(base_name, x)
        x = self._onnx_builder.build_rms_norm(
            f"{base_name}.post_feedforward_layernorm", x, self.cfg.vm_cfg.layer_norm_eps, 0.0
        )
        x = self._onnx_builder.build_op(f"{base_name}.add2", [residual, x], "Add")

        return x

    def _build_attention(
        self, base_name: str, input_node: OnnxNode,
        rope_cos_x: OnnxNode, rope_sin_x: OnnxNode,
        rope_cos_y: OnnxNode, rope_sin_y: OnnxNode,
    ) -> OnnxNode:
        attn_base = f"{base_name}.self_attn"
        head_dim = self.cfg.vm_cfg.hidden_size // self.cfg.vm_cfg.num_attention_heads

        q = self._build_enc_conv(f"{attn_base}.q_proj", input_node)
        k = self._build_enc_conv(f"{attn_base}.k_proj", input_node)
        v = self._build_enc_conv(f"{attn_base}.v_proj", input_node)

        q = self._onnx_builder.build_split_and_concat(
            f"{attn_base}.q_split_concat", q,
            num_splits=self.cfg.vm_cfg.num_attention_heads, split_axis=1, concat_axis=2
        )
        k = self._onnx_builder.build_split_and_concat(
            f"{attn_base}.k_split_concat", k,
            num_splits=self.cfg.vm_cfg.num_attention_heads, split_axis=1, concat_axis=2
        )
        v = self._onnx_builder.build_split_and_concat(
            f"{attn_base}.v_split_concat", v,
            num_splits=self.cfg.vm_cfg.num_attention_heads, split_axis=1, concat_axis=2
        )

        eps = self.cfg.vm_cfg.layer_norm_eps
        q = self._onnx_builder.build_rms_norm(f"{attn_base}.q_norm", q, eps, 0.0)
        k = self._onnx_builder.build_rms_norm(f"{attn_base}.k_norm", k, eps, 0.0)
        v = self._onnx_builder.build_rms_norm(
            f"{attn_base}.v_norm", v, float(self.cfg.vm_cfg.layer_norm_eps),
            weightless=True, num_channels=head_dim,
        )

        q = self._apply_2d_rope(f"{attn_base}.q_rope", q, rope_cos_x, rope_sin_x, rope_cos_y, rope_sin_y)
        k = self._apply_2d_rope(f"{attn_base}.k_rope", k, rope_cos_x, rope_sin_x, rope_cos_y, rope_sin_y)

        attn_scores = self._onnx_builder.build_op(
            f"{attn_base}.attn_scores", [q, k], "Einsum", equation="nchw,nchq->nqhw"
        )
        attn_probs = self._onnx_builder.build_op(
            f"{attn_base}.softmax", [attn_scores], "Softmax", axis=1
        )
        context = self._onnx_builder.build_op(
            f"{attn_base}.context", [attn_probs, v], "Einsum", equation="nchw,nqhc->nqhw"
        )

        x = self._onnx_builder.build_split_and_concat(
            f"{attn_base}.merge_heads_split_concat", context,
            num_splits=self.cfg.vm_cfg.num_attention_heads, split_axis=2, concat_axis=1
        )
        x = self._build_enc_conv(f"{attn_base}.o_proj", x)
        return x


    def _apply_2d_rope(
        self, base_name: str, x: OnnxNode,
        cos_x: OnnxNode, sin_x: OnnxNode,
        cos_y: OnnxNode, sin_y: OnnxNode,
    ) -> OnnxNode:
        """Apply 2D RoPE via a single 4-way split and 4-way concat."""
        split_names = [f"{base_name}.split_out_{i}" for i in range(4)]
        split_node = self._onnx_builder.build_op(
            f"{base_name}.split", [x], "Split", axis=1, output_names=split_names
        )
        x_xr, x_xi, x_yr, x_yi = [[split_node, i] for i in range(4)]

        mul_rr_x = self._onnx_builder.build_op(f"{base_name}.x_rr", [x_xr, cos_x], "Mul")
        mul_ii_x = self._onnx_builder.build_op(f"{base_name}.x_ii", [x_xi, sin_x], "Mul")
        real_x   = self._onnx_builder.build_op(f"{base_name}.x_real", [mul_rr_x, mul_ii_x], "Sub")
        mul_ri_x = self._onnx_builder.build_op(f"{base_name}.x_ri", [x_xr, sin_x], "Mul")
        mul_ir_x = self._onnx_builder.build_op(f"{base_name}.x_ir", [x_xi, cos_x], "Mul")
        imag_x   = self._onnx_builder.build_op(f"{base_name}.x_imag", [mul_ri_x, mul_ir_x], "Add")

        mul_rr_y = self._onnx_builder.build_op(f"{base_name}.y_rr", [x_yr, cos_y], "Mul")
        mul_ii_y = self._onnx_builder.build_op(f"{base_name}.y_ii", [x_yi, sin_y], "Mul")
        real_y   = self._onnx_builder.build_op(f"{base_name}.y_real", [mul_rr_y, mul_ii_y], "Sub")
        mul_ri_y = self._onnx_builder.build_op(f"{base_name}.y_ri", [x_yr, sin_y], "Mul")
        mul_ir_y = self._onnx_builder.build_op(f"{base_name}.y_ir", [x_yi, cos_y], "Mul")
        imag_y   = self._onnx_builder.build_op(f"{base_name}.y_imag", [mul_ri_y, mul_ir_y], "Add")

        return self._onnx_builder.build_op(
            f"{base_name}.concat", [real_x, imag_x, real_y, imag_y], "Concat", axis=1
        )

    def _build_mlp(self, base_name: str, input_node: OnnxNode) -> OnnxNode:
        mlp_base = f"{base_name}.mlp"
        gate = self._build_enc_conv(f"{mlp_base}.gate_proj", input_node)
        up = self._build_enc_conv(f"{mlp_base}.up_proj", input_node)
        act = self._onnx_builder.build_activation(f"{mlp_base}.act", gate, self.cfg.vm_cfg.hidden_act)
        mul = self._onnx_builder.build_op(f"{mlp_base}.mul", [act, up], "Mul")
        return self._build_enc_conv(f"{mlp_base}.down_proj", mul)

 
    def _build_pooler(
        self, base_name: str, input_node: OnnxNode, grid_h: int) -> OnnxNode:
        """Convert 1D patch sequence to 2D grid, pool, then flatten back to 1D."""
        s = self.cfg.vm_cfg.spatial_merge_size

        x = self._onnx_builder.build_split_and_concat(
            f"{base_name}.pooler_reshape_2d",
            input_node,
            num_splits=grid_h,
            split_axis=3,
            concat_axis=2,
        )

        x = self._onnx_builder.build_op(
            f"{base_name}.pooler_avgpool", [x], "AveragePool",
            kernel_shape=[s, s], strides=[s, s]
        )

        scale_node = self._onnx_builder.create_initializer(
            f"{base_name}.pooler_scale_val", float(np.sqrt(self.cfg.vm_cfg.hidden_size))
        )
        x = self._onnx_builder.build_op(f"{base_name}.pooler_scale", [x, scale_node], "Mul")

        x = self._onnx_builder.build_split_and_concat(
            f"{base_name}.pooler_reshape_1d",
            x,
            num_splits=grid_h // s,
            split_axis=2,
            concat_axis=3,
        )
        return x

    def _build_multimodal_embedder(self, base_name: str, input_node: OnnxNode) -> OnnxNode:
        """Weightless RMSNorm + linear projection to LM hidden size."""
        x = self._onnx_builder.build_rms_norm(
            f"{base_name}.embed_vision_norm", input_node, float(self.cfg.vm_cfg.layer_norm_eps),
            weightless=True, num_channels=self.cfg.vm_cfg.hidden_size,
        )
        x = self._onnx_builder.build_conv(
            f"model.embed_vision.embedding_projection", x, is_fc=True
        )
        return x

    def _build_enc_conv(self, base_name: str, input_node: OnnxNode, **kwargs) -> OnnxNode:
        """build_conv for encoder-layer projections with optional activation clipping.

        Weights live at .linear.weight. If the checkpoint contains finite
        input_min/input_max or output_min/output_max scalars (activation-quantization
        metadata), Clip nodes are inserted before and after the Conv respectively.
        """
        x = self._maybe_clip(base_name, input_node, "input")
        x = self._onnx_builder.build_conv(
            base_name, x, is_fc=True,
            src_weight_name=f"{base_name}.linear.weight",
            **kwargs
        )
        return self._maybe_clip(base_name, x, "output")

    def _maybe_clip(self, base_name: str, input_node: OnnxNode, side: str) -> OnnxNode:
        """Insert a Clip node if finite {side}_min / {side}_max scalars exist in the checkpoint."""
        min_name = f"{base_name}.{side}_min"
        max_name = f"{base_name}.{side}_max"
        if not (self._onnx_builder.check_param_func(min_name)
                and self._onnx_builder.check_param_func(max_name)):
            return input_node
        clip_min = float(self._onnx_builder.get_param_func(min_name))
        clip_max = float(self._onnx_builder.get_param_func(max_name))
        if not (np.isfinite(clip_min) and np.isfinite(clip_max)):
            return input_node
        min_node = self._onnx_builder.create_initializer(min_name, clip_min)
        max_node = self._onnx_builder.create_initializer(max_name, clip_max)
        return self._onnx_builder.build_op(
            f"{base_name}.{side}_clip", [input_node, min_node, max_node], "Clip"
        )

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        input_tessellate_params = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC,
            persistent_mem_name="input",
            dram_shape=None,
        )
        return {0: input_tessellate_params}

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        return {}
