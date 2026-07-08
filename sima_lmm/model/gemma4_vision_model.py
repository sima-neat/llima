from dataclasses import dataclass

import numpy as np

from afe.apis.defines import TensorDRAMLayout, gen2_target
from afe.backends.backends import Backend
from afe.ir.attributes import ClipAttrs
from afe.ir.build_node import NodeOrHandle
from afe.ir.defines import Status, get_expected_tensor_value
from afe.ir.serializer import save_awesomenet
from afe.ir.tensor_type import ScalarType, TensorType
from sima_lmm.model.base import BaseModel, LayerConfiguration, TensorTessellateParameters
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import (
    SimaBuilder, activation_dtype, activation_type, build_activation, build_conv,
    create_channel_slice, load_tensor_from_source
)


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

    def gen_model_sdk_files_directly(
        self,
        layer_cfg: LayerConfiguration,
        log_level: int,
        quantizable: bool,
    ):
        g = self._build_sima_nodes(self.hf_model.vision_model_param_base_name, quantizable)
        save_awesomenet(g, self.model_name + (".fp32" if quantizable else ""), str(self.sima_model_sdk_path))

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
        pos_embed, cos_x, sin_x, cos_y, sin_y = self._calc_static_constants(base_name, grid_h, grid_w)

        pos_embed_node = self._onnx_builder.create_initializer(
            f"{base_name}.pos_embed_static", pos_embed
        )
        rope_cos_x = self._onnx_builder.create_initializer(f"{base_name}.rope_cos_x", cos_x)
        rope_sin_x = self._onnx_builder.create_initializer(f"{base_name}.rope_sin_x", sin_x)
        rope_cos_y = self._onnx_builder.create_initializer(f"{base_name}.rope_cos_y", cos_y)
        rope_sin_y = self._onnx_builder.create_initializer(f"{base_name}.rope_sin_y", sin_y)

        return pos_embed_node, rope_cos_x, rope_sin_x, rope_cos_y, rope_sin_y

    def _calc_static_constants(
        self, base_name: str, grid_h: int, grid_w: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_coords = np.tile(np.arange(grid_w, dtype=np.int64), grid_h)
        y_coords = np.repeat(np.arange(grid_h, dtype=np.int64), grid_w)

        pos_table = self.get_hf_param(
            f"{base_name}.patch_embedder.position_embedding_table"
        )
        pos_embed = pos_table[0][x_coords] + pos_table[1][y_coords]
        pos_embed = pos_embed.T[None, :, None, :].astype(np.float32)

        rope_quarter_dim = self.cfg.vm_cfg.hidden_size // self.cfg.vm_cfg.num_attention_heads // 4
        inv_freq = 1.0 / (
            self.cfg.vm_cfg.rope_theta ** (np.arange(0, rope_quarter_dim, dtype=np.float32) / rope_quarter_dim)
        )

        cos_x = np.cos(np.outer(x_coords, inv_freq)).T.reshape(1, rope_quarter_dim, 1, self.cfg.vm_cfg.seq_len).astype(np.float32)
        sin_x = np.sin(np.outer(x_coords, inv_freq)).T.reshape(1, rope_quarter_dim, 1, self.cfg.vm_cfg.seq_len).astype(np.float32)
        cos_y = np.cos(np.outer(y_coords, inv_freq)).T.reshape(1, rope_quarter_dim, 1, self.cfg.vm_cfg.seq_len).astype(np.float32)
        sin_y = np.sin(np.outer(y_coords, inv_freq)).T.reshape(1, rope_quarter_dim, 1, self.cfg.vm_cfg.seq_len).astype(np.float32)

        return pos_embed, cos_x, sin_x, cos_y, sin_y

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

    def _build_sima_nodes(self, base_name: str, quantizable: bool):
        patch_feature_size = 3 * self.cfg.vm_cfg.patch_size * self.cfg.vm_cfg.patch_size
        if self.include_embeddings:
            input_shape = (1, 1, self.cfg.vm_cfg.seq_len, patch_feature_size)
        else:
            input_shape = (1, 1, self.cfg.vm_cfg.seq_len, self.cfg.vm_cfg.hidden_size)

        builder = SimaBuilder(Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target)
        model_input = builder.create_placeholder_node(
            "input", TensorType(activation_type(quantizable), input_shape)
        )
        builder.begin_subnet([model_input])
        mla_input = builder.create_placeholder_node(
            "MLA_0/input", TensorType(activation_type(quantizable), input_shape)
        )

        self._build_sima_vision_model(builder, base_name, mla_input, quantizable)
        mla_node = builder.finish_subnet("MLA_0")
        if activation_type(quantizable) != ScalarType.float32:
            _ = builder.create_cast_node(mla_node, ScalarType.float32, backend=Backend.EV)
        return builder.finish(self.model_name)

    def _build_sima_vision_model(
        self, builder: SimaBuilder, base_name: str, input_node: NodeOrHandle, quantizable: bool
    ) -> NodeOrHandle:
        if isinstance(self.cfg.vm_cfg.image_size, list):
            image_h, image_w = self.cfg.vm_cfg.image_size
        else:
            image_h = image_w = self.cfg.vm_cfg.image_size

        grid_h = image_h // self.cfg.vm_cfg.patch_size
        grid_w = image_w // self.cfg.vm_cfg.patch_size
        pos_embed, rope_cos_x, rope_sin_x, rope_cos_y, rope_sin_y = self._precompute_sima_constants(
            builder, base_name, grid_h, grid_w, quantizable
        )

        if self.include_embeddings:
            x = self._build_sima_patch_embedder(builder, base_name, input_node, pos_embed, quantizable)
        else:
            x = input_node

        for i in range(self.layer_idx, self.layer_idx + self.num_layers):
            x = self._build_sima_encoder_layer(
                builder, f"{base_name}.encoder.layers.{i}", x,
                rope_cos_x, rope_sin_x, rope_cos_y, rope_sin_y, quantizable
            )

        if self.include_mm_proj:
            x = self._build_sima_pooler(builder, base_name, x, grid_h, quantizable)
            x = self._build_sima_multimodal_embedder(builder, base_name, x)
        return x

    def _precompute_sima_constants(
        self, builder: SimaBuilder, base_name: str, grid_h: int, grid_w: int, quantizable: bool
    ) -> tuple[NodeOrHandle, NodeOrHandle, NodeOrHandle, NodeOrHandle, NodeOrHandle]:
        constants = self._calc_static_constants(base_name, grid_h, grid_w)
        dtype = activation_dtype(quantizable)
        return tuple(
            builder.create_constant_node(c.transpose(0, 2, 3, 1).astype(dtype))
            for c in constants
        )

    def _build_sima_patch_embedder(
        self, builder: SimaBuilder, base_name: str, input_node: NodeOrHandle,
        pos_embed_node: NodeOrHandle, quantizable: bool
    ) -> NodeOrHandle:
        dtype = activation_dtype(quantizable)
        sub = builder.create_subtract_node(
            input_node, builder.create_constant_node(np.array([0.5], dtype=dtype))
        )
        scaled = builder.create_mul_node(
            sub, builder.create_constant_node(np.array([2.0], dtype=dtype))
        )
        proj = build_conv(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.patch_embedder.input_proj", scaled,
            is_fc=True,
        )
        return builder.create_add_node(proj, pos_embed_node)

    def _build_sima_encoder_layer(
        self,
        builder: SimaBuilder,
        base_name: str,
        input_node: NodeOrHandle,
        rope_cos_x: NodeOrHandle,
        rope_sin_x: NodeOrHandle,
        rope_cos_y: NodeOrHandle,
        rope_sin_y: NodeOrHandle,
        quantizable: bool,
    ) -> NodeOrHandle:
        eps = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        w = load_tensor_from_source(f"{base_name}.input_layernorm.weight", self.get_hf_param, self.check_hf_param)
        x = builder.create_rms_norm_node(input_node, eps, w)
        x = self._build_sima_attention(builder, base_name, x, rope_cos_x, rope_sin_x, rope_cos_y, rope_sin_y)
        w = load_tensor_from_source(f"{base_name}.post_attention_layernorm.weight", self.get_hf_param, self.check_hf_param)
        x = builder.create_rms_norm_node(x, eps, w)
        x = builder.create_add_node(input_node, x)

        residual = x
        w = load_tensor_from_source(f"{base_name}.pre_feedforward_layernorm.weight", self.get_hf_param, self.check_hf_param)
        x = builder.create_rms_norm_node(x, eps, w)
        x = self._build_sima_mlp(builder, base_name, x, quantizable)
        w = load_tensor_from_source(f"{base_name}.post_feedforward_layernorm.weight", self.get_hf_param, self.check_hf_param)
        x = builder.create_rms_norm_node(x, eps, w)
        return builder.create_add_node(residual, x)

    def _build_sima_attention(
        self,
        builder: SimaBuilder,
        base_name: str,
        input_node: NodeOrHandle,
        rope_cos_x: NodeOrHandle,
        rope_sin_x: NodeOrHandle,
        rope_cos_y: NodeOrHandle,
        rope_sin_y: NodeOrHandle,
    ) -> NodeOrHandle:
        attn_base = f"{base_name}.self_attn"
        num_heads = self.cfg.vm_cfg.num_attention_heads
        head_dim = self.cfg.vm_cfg.hidden_size // num_heads

        q_base = f"{attn_base}.q_proj"
        k_base = f"{attn_base}.k_proj"
        v_base = f"{attn_base}.v_proj"
        qkv_bounds = [
            self._get_sima_clip_bounds(q_base, "input"),
            self._get_sima_clip_bounds(k_base, "input"),
            self._get_sima_clip_bounds(v_base, "input"),
        ]
        if qkv_bounds[0] is not None and all(bounds == qkv_bounds[0] for bounds in qkv_bounds):
            shared_input = builder.create_clip_node(input_node, qkv_bounds[0][0], qkv_bounds[0][1])
            q = self._build_sima_enc_conv(builder, q_base, shared_input, include_input_clip=False)
            k = self._build_sima_enc_conv(builder, k_base, shared_input, include_input_clip=False)
            v = self._build_sima_enc_conv(builder, v_base, shared_input, include_input_clip=False)
        else:
            q = self._build_sima_enc_conv(builder, q_base, input_node)
            k = self._build_sima_enc_conv(builder, k_base, input_node)
            v = self._build_sima_enc_conv(builder, v_base, input_node)

        q = builder.create_slice_concat_node(q, axis=1, split_axis=3, split_block=num_heads, split_repeat=1)
        k = builder.create_slice_concat_node(k, axis=1, split_axis=3, split_block=num_heads, split_repeat=1)
        v = builder.create_slice_concat_node(v, axis=1, split_axis=3, split_block=num_heads, split_repeat=1)

        eps = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        w = load_tensor_from_source(f"{attn_base}.q_norm.weight", self.get_hf_param, self.check_hf_param)
        q = builder.create_rms_norm_node(q, eps, w)
        w = load_tensor_from_source(f"{attn_base}.k_norm.weight", self.get_hf_param, self.check_hf_param)
        k = builder.create_rms_norm_node(k, eps, w)
        v = builder.create_rms_norm_node(v, eps, np.ones(head_dim, dtype=np.float32))

        q = self._build_sima_apply_2d_rope(builder, q, rope_cos_x, rope_sin_x, rope_cos_y, rope_sin_y)
        k = self._build_sima_apply_2d_rope(builder, k, rope_cos_x, rope_sin_x, rope_cos_y, rope_sin_y)

        scores = builder.create_einsum_node(q, k, equation="nhwc,nhqc->nhwq", layout="NHWC")
        probs = builder.create_softmax_node(scores, axis=3)
        context = builder.create_einsum_node(probs, v, equation="nhwc,nhcq->nhwq", layout="NHWC")
        merged = builder.create_slice_concat_node(context, axis=3, split_axis=1, split_block=num_heads, split_repeat=1)
        return self._build_sima_enc_conv(builder, f"{attn_base}.o_proj", merged)

    def _build_sima_apply_2d_rope(
        self,
        builder: SimaBuilder,
        x: NodeOrHandle,
        cos_x: NodeOrHandle,
        sin_x: NodeOrHandle,
        cos_y: NodeOrHandle,
        sin_y: NodeOrHandle,
    ) -> NodeOrHandle:
        quarter_dim = self.cfg.vm_cfg.hidden_size // self.cfg.vm_cfg.num_attention_heads // 4
        x_xr = create_channel_slice(builder, x, 0, quarter_dim)
        x_xi = create_channel_slice(builder, x, quarter_dim, 2 * quarter_dim)
        x_yr = create_channel_slice(builder, x, 2 * quarter_dim, 3 * quarter_dim)
        x_yi = create_channel_slice(builder, x, 3 * quarter_dim, 4 * quarter_dim)

        real_x = builder.create_subtract_node(
            builder.create_mul_node(x_xr, cos_x),
            builder.create_mul_node(x_xi, sin_x),
        )
        imag_x = builder.create_add_node(
            builder.create_mul_node(x_xr, sin_x),
            builder.create_mul_node(x_xi, cos_x),
        )
        real_y = builder.create_subtract_node(
            builder.create_mul_node(x_yr, cos_y),
            builder.create_mul_node(x_yi, sin_y),
        )
        imag_y = builder.create_add_node(
            builder.create_mul_node(x_yr, sin_y),
            builder.create_mul_node(x_yi, cos_y),
        )
        return builder.create_concat_node([real_x, imag_x, real_y, imag_y], axis=3)

    def _build_sima_mlp(
        self, builder: SimaBuilder, base_name: str, input_node: NodeOrHandle, quantizable: bool
    ) -> NodeOrHandle:
        mlp_base = f"{base_name}.mlp"
        gate_base = f"{mlp_base}.gate_proj"
        up_base = f"{mlp_base}.up_proj"
        gate_bounds = self._get_sima_clip_bounds(gate_base, "input")
        up_bounds = self._get_sima_clip_bounds(up_base, "input")
        if gate_bounds is not None and gate_bounds == up_bounds:
            shared_input = builder.create_clip_node(input_node, gate_bounds[0], gate_bounds[1])
            gate = self._build_sima_enc_conv(builder, gate_base, shared_input, include_input_clip=False)
            up = self._build_sima_enc_conv(builder, up_base, shared_input, include_input_clip=False)
        else:
            gate = self._build_sima_enc_conv(builder, gate_base, input_node)
            up = self._build_sima_enc_conv(builder, up_base, input_node)
        act = build_activation(builder, gate, self.cfg.vm_cfg.hidden_act, quantizable)
        mul = builder.create_mul_node(act, up)
        return self._build_sima_enc_conv(builder, f"{mlp_base}.down_proj", mul)

    def _build_sima_pooler(
        self, builder: SimaBuilder, base_name: str, input_node: NodeOrHandle,
        grid_h: int, quantizable: bool
    ) -> NodeOrHandle:
        s = self.cfg.vm_cfg.spatial_merge_size
        x = builder.create_slice_concat_node(
            input_node, axis=1, split_axis=2, split_block=grid_h, split_repeat=1
        )
        x = builder.create_avgpool2d_node(x, kernel_shape=(s, s), strides=(s, s))
        x = builder.create_mul_node(
            x,
            builder.create_constant_node(
                np.array([float(np.sqrt(self.cfg.vm_cfg.hidden_size))], dtype=activation_dtype(quantizable))
            ),
        )
        return builder.create_slice_concat_node(
            x, axis=2, split_axis=1, split_block=grid_h // s, split_repeat=1
        )

    def _build_sima_multimodal_embedder(
        self, builder: SimaBuilder, base_name: str, input_node: NodeOrHandle
    ) -> NodeOrHandle:
        x = builder.create_rms_norm_node(
            input_node,
            float(np.float32(self.cfg.vm_cfg.layer_norm_eps)),
            np.ones(self.cfg.vm_cfg.hidden_size, dtype=np.float32),
        )
        return build_conv(
            builder, self.get_hf_param, self.check_hf_param,
            "model.embed_vision.embedding_projection", x,
            is_fc=True,
        )

    def _build_sima_enc_conv(
        self,
        builder: SimaBuilder,
        base_name: str,
        input_node: NodeOrHandle,
        include_input_clip: bool = True,
    ) -> NodeOrHandle:
        x = input_node
        if include_input_clip:
            x = self._build_sima_maybe_clip(builder, base_name, x, "input")

        activation = None
        out_bounds = self._get_sima_clip_bounds(base_name, "output")
        if out_bounds is not None:
            ifm_type = get_expected_tensor_value(x.get_type().output)
            params = self.get_hf_param(f"{base_name}.linear.weight")
            weight = params[1] if isinstance(params, tuple) else params
            activation = ClipAttrs(
                a_min=out_bounds[0],
                a_max=out_bounds[1],
                shape=(*ifm_type.shape[:-1], weight.shape[0]),
                scalar_type=ifm_type.scalar,
            )

        x = build_conv(
            builder, self.get_hf_param, self.check_hf_param,
            base_name, x,
            is_fc=True,
            src_weight_name=f"{base_name}.linear.weight",
            activation=activation,
        )
        return x

    def _build_sima_maybe_clip(
        self, builder: SimaBuilder, base_name: str, input_node: NodeOrHandle, side: str
    ) -> NodeOrHandle:
        bounds = self._get_sima_clip_bounds(base_name, side)
        if bounds is None:
            return input_node
        return builder.create_clip_node(input_node, bounds[0], bounds[1])

    def _get_sima_clip_bounds(self, base_name: str, side: str) -> tuple[float, float] | None:
        min_name = f"{base_name}.{side}_min"
        max_name = f"{base_name}.{side}_max"
        if not (self.check_hf_param(min_name) and self.check_hf_param(max_name)):
            return None
        clip_min = float(self.get_hf_param(min_name))
        clip_max = float(self.get_hf_param(max_name))
        if not (np.isfinite(clip_min) and np.isfinite(clip_max)):
            return None
        return clip_min, clip_max
