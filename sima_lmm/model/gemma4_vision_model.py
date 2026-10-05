from dataclasses import dataclass

import numpy as np

from afe.apis.defines import TensorDRAMLayout
from afe.ir.attributes import ClipAttrs
from afe.ir.build_node import NodeOrHandle
from afe.ir.defines import get_expected_tensor_value
from sima_lmm.model.base import BaseModel, LayerConfiguration, TensorTessellateParameters
from sima_lmm.model.model_graph import ModelGraph, activation_dtype


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
    include_embeddings: bool
    include_mm_proj: bool

    def generate_graph(
        self,
        layer_cfg: LayerConfiguration,
        quantizable: bool,
    ):
        base_name = self.hf_model.vision_model_param_base_name
        patch_feature_size = 3 * self.cfg.vm_cfg.patch_size * self.cfg.vm_cfg.patch_size
        if self.include_embeddings:
            input_shape = (1, 1, self.cfg.vm_cfg.seq_len, patch_feature_size)
        else:
            input_shape = (1, 1, self.cfg.vm_cfg.seq_len, self.cfg.vm_cfg.hidden_size)

        graph = ModelGraph(self, {"input": input_shape}, quantizable)
        vision_output = self._build_vision_model(
            graph, base_name, graph.inputs["input"], quantizable
        )
        outputs = [vision_output]
        if self.include_mm_proj and self.cfg.pipeline_cfg.quantize_embeddings:
            outputs = list(graph.quant(vision_output))
        graph.save(outputs)

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

    def _build_vision_model(
        self, graph: ModelGraph, base_name: str, input_node: NodeOrHandle, quantizable: bool
    ) -> NodeOrHandle:
        if isinstance(self.cfg.vm_cfg.image_size, list):
            image_h, image_w = self.cfg.vm_cfg.image_size
        else:
            image_h = image_w = self.cfg.vm_cfg.image_size

        grid_h = image_h // self.cfg.vm_cfg.patch_size
        grid_w = image_w // self.cfg.vm_cfg.patch_size
        pos_embed, rope_cos_x, rope_sin_x, rope_cos_y, rope_sin_y = self._precompute_constants(
            graph, base_name, grid_h, grid_w, quantizable
        )

        if self.include_embeddings:
            x = self._build_patch_embedder(graph, base_name, input_node, pos_embed, quantizable)
        else:
            x = input_node

        x = self._build_encoder_layer(
            graph, f"{base_name}.encoder.layers.{self.layer_idx}", x,
            rope_cos_x, rope_sin_x, rope_cos_y, rope_sin_y
        )

        if self.include_mm_proj:
            x = self._build_pooler(graph, base_name, x, grid_h, quantizable)
            x = self._build_multimodal_embedder(graph, base_name, x)
        return x

    def _precompute_constants(
        self, graph: ModelGraph, base_name: str, grid_h: int, grid_w: int, quantizable: bool
    ) -> tuple[NodeOrHandle, NodeOrHandle, NodeOrHandle, NodeOrHandle, NodeOrHandle]:
        constants = self._calc_static_constants(base_name, grid_h, grid_w)
        dtype = activation_dtype(quantizable)
        return tuple(
            graph.constant(c.transpose(0, 2, 3, 1).astype(dtype))
            for c in constants
        )

    def _build_patch_embedder(
        self, graph: ModelGraph, base_name: str, input_node: NodeOrHandle,
        pos_embed_node: NodeOrHandle, quantizable: bool
    ) -> NodeOrHandle:
        dtype = activation_dtype(quantizable)
        sub = graph.sub(
            input_node, graph.constant(np.array([0.5], dtype=dtype))
        )
        scaled = graph.mul(
            sub, graph.constant(np.array([2.0], dtype=dtype))
        )
        proj = graph.linear(f"{base_name}.patch_embedder.input_proj", scaled)
        return graph.add(proj, pos_embed_node)

    def _build_encoder_layer(
        self,
        graph: ModelGraph,
        base_name: str,
        input_node: NodeOrHandle,
        rope_cos_x: NodeOrHandle,
        rope_sin_x: NodeOrHandle,
        rope_cos_y: NodeOrHandle,
        rope_sin_y: NodeOrHandle,
    ) -> NodeOrHandle:
        eps = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        x = graph.rms_norm(f"{base_name}.input_layernorm", input_node, epsilon=eps)
        x = self._build_attention(graph, base_name, x, rope_cos_x, rope_sin_x, rope_cos_y, rope_sin_y)
        x = graph.rms_norm(f"{base_name}.post_attention_layernorm", x, epsilon=eps)
        x = graph.add(input_node, x)

        residual = x
        x = graph.rms_norm(f"{base_name}.pre_feedforward_layernorm", x, epsilon=eps)
        x = self._build_mlp(graph, base_name, x)
        x = graph.rms_norm(f"{base_name}.post_feedforward_layernorm", x, epsilon=eps)
        return graph.add(residual, x)

    def _build_attention(
        self,
        graph: ModelGraph,
        base_name: str,
        input_node: NodeOrHandle,
        rope_cos_x: NodeOrHandle,
        rope_sin_x: NodeOrHandle,
        rope_cos_y: NodeOrHandle,
        rope_sin_y: NodeOrHandle,
    ) -> NodeOrHandle:
        attn_base = f"{base_name}.self_attn"
        num_heads = self.cfg.vm_cfg.num_attention_heads

        q_base = f"{attn_base}.q_proj"
        k_base = f"{attn_base}.k_proj"
        v_base = f"{attn_base}.v_proj"
        qkv_bounds = [
            self._get_clip_bounds(q_base, "input"),
            self._get_clip_bounds(k_base, "input"),
            self._get_clip_bounds(v_base, "input"),
        ]
        if qkv_bounds[0] is not None and all(bounds == qkv_bounds[0] for bounds in qkv_bounds):
            shared_input = graph.clip(input_node, qkv_bounds[0][0], qkv_bounds[0][1])
            q = self._build_enc_conv(graph, q_base, shared_input, include_input_clip=False)
            k = self._build_enc_conv(graph, k_base, shared_input, include_input_clip=False)
            v = self._build_enc_conv(graph, v_base, shared_input, include_input_clip=False)
        else:
            q = self._build_enc_conv(graph, q_base, input_node)
            k = self._build_enc_conv(graph, k_base, input_node)
            v = self._build_enc_conv(graph, v_base, input_node)

        q = graph.split_heads(q, num_heads)
        k = graph.split_heads(k, num_heads)
        v = graph.split_heads(v, num_heads)

        eps = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        q = graph.rms_norm(f"{attn_base}.q_norm", q, epsilon=eps)
        k = graph.rms_norm(f"{attn_base}.k_norm", k, epsilon=eps)
        v = graph.rms_norm(None, v, epsilon=eps)

        q = graph.rope2d(q, rope_cos_x, rope_sin_x, rope_cos_y, rope_sin_y)
        k = graph.rope2d(k, rope_cos_x, rope_sin_x, rope_cos_y, rope_sin_y)
        context = graph.attention(q, k, v)
        return self._build_enc_conv(graph, f"{attn_base}.o_proj", graph.merge_heads(context))

    def _build_mlp(
        self, graph: ModelGraph, base_name: str, input_node: NodeOrHandle
    ) -> NodeOrHandle:
        mlp_base = f"{base_name}.mlp"
        gate_base = f"{mlp_base}.gate_proj"
        up_base = f"{mlp_base}.up_proj"
        gate_bounds = self._get_clip_bounds(gate_base, "input")
        up_bounds = self._get_clip_bounds(up_base, "input")
        if gate_bounds is not None and gate_bounds == up_bounds:
            shared_input = graph.clip(input_node, gate_bounds[0], gate_bounds[1])
            gate = self._build_enc_conv(graph, gate_base, shared_input, include_input_clip=False)
            up = self._build_enc_conv(graph, up_base, shared_input, include_input_clip=False)
        else:
            gate = self._build_enc_conv(graph, gate_base, input_node)
            up = self._build_enc_conv(graph, up_base, input_node)
        act = graph.activation(gate, self.cfg.vm_cfg.hidden_act)
        mul = graph.mul(act, up)
        return self._build_enc_conv(graph, f"{mlp_base}.down_proj", mul)

    def _build_pooler(
        self, graph: ModelGraph, base_name: str, input_node: NodeOrHandle,
        grid_h: int, quantizable: bool
    ) -> NodeOrHandle:
        s = self.cfg.vm_cfg.spatial_merge_size
        x = graph.split_concat(
            input_node, axis=1, split_axis=2, split_block=grid_h, split_repeat=1
        )
        x = graph.avgpool2d(x, kernel_shape=(s, s), strides=(s, s))
        x = graph.mul(
            x,
            graph.constant(
                np.array([float(np.sqrt(self.cfg.vm_cfg.hidden_size))], dtype=activation_dtype(quantizable))
            ),
        )
        return graph.split_concat(
            x, axis=2, split_axis=1, split_block=grid_h // s, split_repeat=1
        )

    def _build_multimodal_embedder(
        self, graph: ModelGraph, base_name: str, input_node: NodeOrHandle
    ) -> NodeOrHandle:
        x = graph.rms_norm(None, input_node, epsilon=self.cfg.vm_cfg.layer_norm_eps)
        return graph.linear("model.embed_vision.embedding_projection", x)

    def _build_enc_conv(
        self,
        graph: ModelGraph,
        base_name: str,
        input_node: NodeOrHandle,
        include_input_clip: bool = True,
    ) -> NodeOrHandle:
        x = input_node
        if include_input_clip:
            x = self._build_maybe_clip(graph, base_name, x, "input")

        activation = None
        out_bounds = self._get_clip_bounds(base_name, "output")
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

        x = graph.linear(
            base_name,
            x,
            src_weight_name=f"{base_name}.linear.weight",
            activation=activation,
        )
        return x

    def _build_maybe_clip(
        self, graph: ModelGraph, base_name: str, input_node: NodeOrHandle, side: str
    ) -> NodeOrHandle:
        bounds = self._get_clip_bounds(base_name, side)
        if bounds is None:
            return input_node
        return graph.clip(input_node, bounds[0], bounds[1])

    def _get_clip_bounds(self, base_name: str, side: str) -> tuple[float, float] | None:
        min_name = f"{base_name}.{side}_min"
        max_name = f"{base_name}.{side}_max"
        if not (self.check_hf_param(min_name) and self.check_hf_param(max_name)):
            return None
        clip_min = float(self.get_hf_param(min_name))
        clip_max = float(self.get_hf_param(max_name))
        if not (np.isfinite(clip_min) and np.isfinite(clip_max)):
            return None
        return clip_min, clip_max
