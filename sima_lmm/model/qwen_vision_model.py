from dataclasses import dataclass

import numpy as np

from afe.apis.defines import TensorDRAMLayout
from afe.ir.build_node import NodeOrHandle
from sima_lmm.model.base import BaseModel, TensorTessellateParameters, LayerConfiguration
from sima_lmm.model.model_graph import ModelGraph, activation_dtype
from sima_lmm.config.vlm_config import VlmArchType

@dataclass
class QwenVisionLayerModel(BaseModel):
    """Qwen Vision model implementation (Qwen2.5-VL and Qwen3-VL).

    Differs from StandardVisionLayerModel (CLIP/SigLIP/LFM2) in:
    - 3D Conv patch embedding (vs 2D Conv)
    - RoPE position embeddings (vs additive embeddings)
    - Windowed/global attention with RoPE on Q/K (Qwen2.5) or full attention with RoPE (Qwen3)
    - RMSNorm (Qwen2.5) or LayerNorm (Qwen3) normalization
    - Strided Conv spatial merging (vs linear projectors/pooling)
    - Multiple outputs for Qwen3-VL deepstack mergers
    """

    layer_idx: int
    include_embeddings: bool
    include_mm_proj: bool


    # ------------------------------------------------------------------------
    # Qwen 3 logic
    # ------------------------------------------------------------------------

    def _calc_qwen3_position_embeddings_array(self, base_name: str) -> np.ndarray:
        """
        Calculates Qwen3-VL position embeddings by interpolating pretrained weights to target image size.

        Uses bilinear interpolation to resize the pretrained position embedding grid,
        then reorders the patches to match the spatial merge pattern used during inference.
        """
        pos_weight = self.get_hf_param(f"{base_name}.pos_embed.weight")
        hidden_size = pos_weight.shape[1]
        grid_size = int(np.sqrt(pos_weight.shape[0]))

        if isinstance(self.cfg.vm_cfg.image_size, list):
            image_h, image_w = self.cfg.vm_cfg.image_size
        else:
            image_h = image_w = self.cfg.vm_cfg.image_size

        grid_h = image_h // self.cfg.vm_cfg.patch_size
        grid_w = image_w // self.cfg.vm_cfg.patch_size
        merge_size = self.cfg.vm_cfg.spatial_merge_size

        h_lin = np.linspace(0, grid_size - 1, grid_h, dtype=np.float32)
        w_lin = np.linspace(0, grid_size - 1, grid_w, dtype=np.float32)

        h_floor = np.floor(h_lin).astype(np.int64)
        w_floor = np.floor(w_lin).astype(np.int64)
        h_ceil = np.clip(h_floor + 1, 0, grid_size - 1)
        w_ceil = np.clip(w_floor + 1, 0, grid_size - 1)

        dh = h_lin - h_floor
        dw = w_lin - w_floor

        base_h = h_floor[:, None] * grid_size
        base_h_ceil = h_ceil[:, None] * grid_size

        indices = [
            (base_h + w_floor).reshape(-1),
            (base_h + w_ceil).reshape(-1),
            (base_h_ceil + w_floor).reshape(-1),
            (base_h_ceil + w_ceil).reshape(-1),
        ]

        weights = [
            ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1),
            ((1 - dh)[:, None] * dw[None, :]).reshape(-1),
            (dh[:, None] * (1 - dw)[None, :]).reshape(-1),
            (dh[:, None] * dw[None, :]).reshape(-1),
        ]

        gathered = [
            pos_weight[idx] * weight[:, None]
            for idx, weight in zip(indices, weights)
        ]
        pos_embed = np.sum(np.stack(gathered, axis=0), axis=0)

        pos_embed = pos_embed.reshape(grid_h, grid_w, hidden_size)
        pos_embed = np.repeat(pos_embed[np.newaxis, ...], 1, axis=0)
        pos_embed = pos_embed.reshape(grid_h * grid_w, hidden_size)

        # Match PyTorch ordering used in the reference implementation.
        pos_embed = pos_embed.reshape(
            1,
            grid_h // merge_size,
            merge_size,
            grid_w // merge_size,
            merge_size,
            hidden_size,
        )
        pos_embed = np.transpose(pos_embed, (0, 1, 3, 2, 4, 5)).reshape(-1, hidden_size)

        pos_embed = pos_embed.astype(np.float32)
        pos_embed = pos_embed.T.reshape(1, hidden_size, 1, -1)
        return pos_embed

    def _calc_qwen3_rotary_tables(self) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(self.cfg.vm_cfg.image_size, list):
            image_h, image_w = self.cfg.vm_cfg.image_size
        else:
            image_h = image_w = self.cfg.vm_cfg.image_size

        grid_h = image_h // self.cfg.vm_cfg.patch_size
        grid_w = image_w // self.cfg.vm_cfg.patch_size

        merge_size = self.cfg.vm_cfg.spatial_merge_size
        merged_h = grid_h // merge_size
        merged_w = grid_w // merge_size

        hidden_size = self.cfg.vm_cfg.hidden_size
        num_heads = self.cfg.vm_cfg.num_attention_heads
        head_dim = hidden_size // num_heads
        half_dim = head_dim // 2

        inv_freq = 1.0 / (10000.0 ** (np.arange(0, half_dim, 2, dtype=np.float32) / half_dim))
        max_hw = max(grid_h, grid_w)
        freq_table = np.outer(inv_freq, np.arange(max_hw, dtype=np.float32))

        block_rows = np.arange(merged_h)[:, None, None, None]
        block_cols = np.arange(merged_w)[None, :, None, None]
        intra_row = np.arange(merge_size)[None, None, :, None]
        intra_col = np.arange(merge_size)[None, None, None, :]

        row_idx = (block_rows * merge_size + intra_row)
        row_idx = row_idx + np.zeros((merged_h, merged_w, merge_size, merge_size), dtype=np.int64)
        col_idx = (block_cols * merge_size + intra_col)
        col_idx = col_idx + np.zeros((merged_h, merged_w, merge_size, merge_size), dtype=np.int64)

        coords = np.stack([row_idx.reshape(-1), col_idx.reshape(-1)], axis=-1)
        rotary = np.concatenate([freq_table[:, coords[:, 0]], freq_table[:, coords[:, 1]]], axis=0)

        cos = np.cos(rotary).astype(np.float32).reshape(1, half_dim, 1, -1)
        sin = np.sin(rotary).astype(np.float32).reshape(1, half_dim, 1, -1)
        return cos, sin

    # ------------------------------------------------------------------------
    # Qwen 2.5 logic
    # ------------------------------------------------------------------------

    def _calc_qwen2_vision_rope_tables(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the permuted 2D RoPE tables and returns them as
        raw (SeqLen, HeadDim / 2) NumPy arrays.
        """
        head_dim = self.cfg.vm_cfg.hidden_size // self.cfg.vm_cfg.num_attention_heads
        rope_dim = head_dim // 2
        if isinstance(self.cfg.vm_cfg.num_patches, list):
            grid_h, grid_w = self.cfg.vm_cfg.num_patches[0], self.cfg.vm_cfg.num_patches[1]
        else:
            grid_h = grid_w = self.cfg.vm_cfg.num_patches
        spatial_merge_size = self.cfg.vm_cfg.spatial_merge_size

        h_grid_2d = np.broadcast_to(np.arange(grid_h).reshape(-1, 1), (grid_h, grid_w))

        w_grid_2d = np.broadcast_to(np.arange(grid_w).reshape(1, -1), (grid_h, grid_w))

        h_blocks = h_grid_2d.reshape(
            grid_h // spatial_merge_size,
            spatial_merge_size,
            grid_w // spatial_merge_size,
            spatial_merge_size,
        )
        w_blocks = w_grid_2d.reshape(
            grid_h // spatial_merge_size,
            spatial_merge_size,
            grid_w // spatial_merge_size,
            spatial_merge_size,
        )

        hpos_ids = np.transpose(h_blocks, (0, 2, 1, 3)).flatten()
        wpos_ids = np.transpose(w_blocks, (0, 2, 1, 3)).flatten()

        inv_freq = 1.0 / (10000.0 ** (np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim))

        max_grid_size = max(grid_h, grid_w)
        seq = np.arange(max_grid_size, dtype=np.float32)
        freqs_full = np.outer(inv_freq, seq)

        h_emb = freqs_full[:, hpos_ids]
        w_emb = freqs_full[:, wpos_ids]

        rotary_pos_emb_np = np.concatenate([h_emb, w_emb], axis=0)

        cos_table_np = np.cos(rotary_pos_emb_np).astype(np.float32)
        sin_table_np = np.sin(rotary_pos_emb_np).astype(np.float32)

        return cos_table_np, sin_table_np

    def _reshape_qwen_patch_embed_kernel(self, weight: np.ndarray) -> np.ndarray:
        """
        Converts the 5D Conv3D kernel from Qwen's patch_embed.proj
        into a 4D kernel for a 1x1 convolution (linear projection).
        """
        out_features = self.cfg.vm_cfg.hidden_size
        in_features = 3 * self.cfg.vm_cfg.temporal_patch_size * (self.cfg.vm_cfg.patch_size ** 2)
        flattened_weight = weight.reshape(out_features, in_features)
        return flattened_weight.reshape(out_features, in_features, 1, 1)

    def _reshape_qwen_patch_embed_scales(self, scales: np.ndarray) -> np.ndarray:
        """Preserve grouped scales while the matching kernel flattens input dimensions."""
        if scales.ndim != 2 or scales.shape[0] != self.cfg.vm_cfg.hidden_size:
            raise ValueError(
                "Qwen patch-embedding scales must have shape "
                f"({self.cfg.vm_cfg.hidden_size}, num_input_blocks), got {scales.shape}"
            )
        return scales

    def _reshape_merger_kernel(self, weight_linear: np.ndarray) -> np.ndarray:
        """
        Converts nn.Linear weight into a strided Conv kernel for patch merging.
        """
        C_mid, _ = weight_linear.shape
        C_in = self.cfg.vm_cfg.hidden_size
        factor = self.cfg.vm_cfg.spatial_merge_size ** 2
        kernel_reshaped = weight_linear.reshape(C_mid, factor, C_in)
        kernel_transposed = kernel_reshaped.transpose(0, 2, 1)
        return kernel_transposed.reshape(C_mid, C_in, 1, factor)

    def generate_graph(
        self,
        layer_cfg: LayerConfiguration,
        quantizable: bool,
    ):
        base_name = self.hf_model.vision_model_param_base_name
        patch_feature_size = (
            3 * self.cfg.vm_cfg.temporal_patch_size * (self.cfg.vm_cfg.patch_size ** 2)
        )
        input_size = (
            patch_feature_size if self.include_embeddings else self.cfg.vm_cfg.hidden_size
        )
        input_shape = (1, 1, self.cfg.vm_cfg.seq_len, input_size)

        graph = ModelGraph(self, {"input": input_shape}, quantizable)
        mla_input = graph.inputs["input"]

        if self.cfg.model_type in (VlmArchType.VLM_QWEN3_VL, VlmArchType.VLM_QWEN3_5_VL):
            output_nodes = self._build_qwen3_vision_model(graph, base_name, mla_input, quantizable)
        else:
            output_nodes = [
                self._build_qwen2_vision_model(graph, base_name, mla_input, quantizable)
            ]

        if self.include_mm_proj and self.cfg.pipeline_cfg.quantize_embeddings:
            quantized_vision_output, vision_scale = graph.quant(output_nodes[0])
            output_nodes = [quantized_vision_output, vision_scale, *output_nodes[1:]]
        graph.save(output_nodes)

    def _build_qwen3_vision_model(
        self, graph: ModelGraph, base_name: str, input_node: NodeOrHandle, quantizable: bool
    ) -> list[NodeOrHandle]:
        cos_table, sin_table = self._prepare_qwen3_rotary_tables(
            graph, base_name, quantizable
        )
        hidden_states = input_node
        if self.include_embeddings:
            pos_embed = self._prepare_qwen3_position_embedding(
                graph, base_name, quantizable
            )
            hidden_states = graph.conv(
                f"{base_name}.patch_embed.proj",
                hidden_states,
                weight_process_func=self._reshape_qwen_patch_embed_kernel,
                scale_process_func=self._reshape_qwen_patch_embed_scales,
                src_bias_name=f"{base_name}.patch_embed.proj.bias",
            )
            hidden_states = graph.add(hidden_states, pos_embed)

        layer_base = f"{base_name}.blocks.{self.layer_idx}"
        hidden_states = self._build_qwen3_vision_block(
            graph, layer_base, hidden_states, cos_table, sin_table
        )
        deepstack_outputs: list[NodeOrHandle] = []
        if self.layer_idx in self.cfg.vm_cfg.deepstack_visual_indexes:
            ds_idx = self.cfg.vm_cfg.deepstack_visual_indexes.index(self.layer_idx)
            ds_base = f"{base_name}.deepstack_merger_list.{ds_idx}"
            deepstack_outputs.append(
                self._build_qwen3_deepstack_merger(
                    graph, ds_base, hidden_states
                )
            )

        primary_output = (
            self._build_qwen3_merger(
                graph, f"{base_name}.merger", hidden_states
            )
            if self.include_mm_proj
            else hidden_states
        )
        return [primary_output, *deepstack_outputs]

    def _build_qwen2_vision_model(
        self, graph: ModelGraph, base_name: str, input_node: NodeOrHandle, quantizable: bool
    ) -> NodeOrHandle:
        hidden_states = input_node
        if self.include_embeddings:
            hidden_states = graph.conv(
                f"{base_name}.patch_embed.proj",
                hidden_states,
                weight_process_func=self._reshape_qwen_patch_embed_kernel,
                scale_process_func=self._reshape_qwen_patch_embed_scales,
            )
        cos_table, sin_table, global_mask, windowed_mask = self._prepare_qwen2_static_inputs(graph, quantizable)

        layer_base = f"{base_name}.blocks.{self.layer_idx}"
        mask = (
            global_mask
            if self.layer_idx in self.cfg.vm_cfg.fullatt_block_indexes
            else windowed_mask
        )
        hidden_states = self._build_qwen2_vision_block(
            graph, layer_base, hidden_states, mask, cos_table, sin_table
        )

        if self.include_mm_proj:
            return self._build_qwen2_merger(graph, base_name, hidden_states)
        return hidden_states

    def _build_qwen3_vision_block(
        self,
        graph: ModelGraph,
        base_name: str,
        input_node: NodeOrHandle,
        cos_table: NodeOrHandle,
        sin_table: NodeOrHandle,
    ) -> NodeOrHandle:
        epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        norm1 = graph.layer_norm(f"{base_name}.norm1", input_node, axis=-1, epsilon=epsilon)
        attn = self._build_qwen_attention(
            graph, f"{base_name}.attn", norm1, cos_table, sin_table,
        )
        add1 = graph.add(input_node, attn)
        norm2 = graph.layer_norm(f"{base_name}.norm2", add1, axis=-1, epsilon=epsilon)
        mlp = graph.mlp(
            f"{base_name}.mlp", norm2, self.cfg.vm_cfg.hidden_act,
            projections=("linear_fc1", "linear_fc2"),
        )
        return graph.add(add1, mlp)

    def _build_qwen2_vision_block(
        self,
        graph: ModelGraph,
        base_name: str,
        input_node: NodeOrHandle,
        attention_mask: NodeOrHandle,
        cos_table: NodeOrHandle,
        sin_table: NodeOrHandle,
    ) -> NodeOrHandle:
        epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        norm1 = graph.rms_norm(f"{base_name}.norm1", input_node, epsilon=epsilon)
        attn = self._build_qwen_attention(
            graph, f"{base_name}.attn", norm1, cos_table, sin_table,
            attention_mask=attention_mask,
        )
        add1 = graph.add(input_node, attn)
        norm2 = graph.rms_norm(f"{base_name}.norm2", add1, epsilon=epsilon)
        mlp = graph.mlp(
            f"{base_name}.mlp", norm2, self.cfg.vm_cfg.hidden_act,
            projections=("gate_proj", "up_proj", "down_proj"),
        )
        return graph.add(add1, mlp)

    def _build_qwen_attention(
        self,
        graph: ModelGraph,
        base_name: str,
        input_node: NodeOrHandle,
        cos_table: NodeOrHandle,
        sin_table: NodeOrHandle,
        attention_mask: NodeOrHandle = None,
    ) -> NodeOrHandle:
        num_heads = self.cfg.vm_cfg.num_attention_heads
        hidden_size = self.cfg.vm_cfg.hidden_size
        head_dim = hidden_size // num_heads

        qkv = graph.linear(f"{base_name}.qkv", input_node)
        q = graph.slice(qkv, [0], [hidden_size], [1], [3])
        k = graph.slice(qkv, [hidden_size], [2 * hidden_size], [1], [3])
        v = graph.slice(qkv, [2 * hidden_size], [3 * hidden_size], [1], [3])

        q_heads = graph.split_heads(q, num_heads)
        k_heads = graph.split_heads(k, num_heads)
        v_heads = graph.split_heads(v, num_heads)
        q_rope = graph.rope(q_heads, cos_table, sin_table)
        k_rope = graph.rope(k_heads, cos_table, sin_table)
        context = graph.attention(
            q_rope, k_rope, v_heads, mask=attention_mask, score_scale=head_dim ** -0.5
        )
        return graph.linear(f"{base_name}.proj", graph.merge_heads(context))

    def _build_qwen3_merger(
        self, graph: ModelGraph, base_name: str, input_node: NodeOrHandle
    ) -> NodeOrHandle:
        epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        norm = graph.layer_norm(f"{base_name}.norm", input_node, axis=-1, epsilon=epsilon)
        factor = self.cfg.vm_cfg.spatial_merge_size ** 2
        fc1 = graph.conv(
            f"{base_name}.linear_fc1",
            norm,
            stride=(1, factor),
            weight_process_func=self._reshape_merger_kernel,
            src_bias_name=f"{base_name}.linear_fc1.bias",
        )
        act = graph.activation(fc1, self.cfg.mm_cfg.hidden_act)
        return graph.linear(f"{base_name}.linear_fc2", act)

    def _build_qwen2_merger(
        self, graph: ModelGraph, base_name: str, input_node: NodeOrHandle
    ) -> NodeOrHandle:
        epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        norm = graph.rms_norm(f"{base_name}.merger.ln_q", input_node, epsilon=epsilon)
        factor = self.cfg.vm_cfg.spatial_merge_size ** 2
        fc1 = graph.conv(
            f"{base_name}.merger.mlp.0",
            norm,
            stride=(1, factor),
            weight_process_func=self._reshape_merger_kernel,
            src_bias_name=f"{base_name}.merger.mlp.0.bias",
        )
        act = graph.activation(fc1, self.cfg.mm_cfg.hidden_act)
        return graph.linear(f"{base_name}.merger.mlp.2", act)

    def _build_qwen3_deepstack_merger(
        self, graph: ModelGraph, base_name: str, input_node: NodeOrHandle
    ) -> NodeOrHandle:
        factor = self.cfg.vm_cfg.spatial_merge_size ** 2
        grouped_seq = self.cfg.vm_cfg.seq_len // factor
        hidden = self.cfg.vm_cfg.hidden_size

        # NHWC (1, 1, seq, hidden) → (1, 1, grouped_seq, hidden*factor).
        # In NHWC, consecutive tokens are contiguous in memory, so a plain reshape
        # produces consecutive grouping — same semantics as PyTorch's `.view()`.
        reshaped = graph.reshape(
            input_node, [1, 1, grouped_seq, hidden * factor]
        )
        epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        norm = graph.layer_norm(f"{base_name}.norm", reshaped, axis=-1, epsilon=epsilon)
        fc1 = graph.linear(f"{base_name}.linear_fc1", norm)
        act = graph.activation(fc1, "gelu")
        return graph.linear(f"{base_name}.linear_fc2", act)

    def _prepare_qwen3_rotary_tables(
        self, graph: ModelGraph, base_name: str, quantizable: bool
    ) -> tuple[NodeOrHandle, NodeOrHandle]:
        dtype = activation_dtype(quantizable)
        cos_np, sin_np = self._calc_qwen3_rotary_tables()
        cos_node = graph.constant(cos_np.transpose(0, 2, 3, 1).astype(dtype))
        sin_node = graph.constant(sin_np.transpose(0, 2, 3, 1).astype(dtype))
        return cos_node, sin_node

    def _prepare_qwen3_position_embedding(
        self, graph: ModelGraph, base_name: str, quantizable: bool
    ) -> NodeOrHandle:
        pos_nchw = self._calc_qwen3_position_embeddings_array(base_name)
        pos_nhwc = pos_nchw.transpose(0, 2, 3, 1).astype(activation_dtype(quantizable))
        return graph.constant(pos_nhwc)

    def _prepare_qwen2_static_inputs(
        self, graph: ModelGraph, quantizable: bool
    ) -> tuple[NodeOrHandle, NodeOrHandle, NodeOrHandle, NodeOrHandle]:
        seq_len = self.cfg.vm_cfg.seq_len
        dtype = activation_dtype(quantizable)
        cos_np, sin_np = self._calc_qwen2_vision_rope_tables()
        half_dim = cos_np.shape[0]
        cos_node = graph.constant(
            cos_np.reshape(1, half_dim, 1, seq_len).transpose(0, 2, 3, 1).astype(dtype)
        )
        sin_node = graph.constant(
            sin_np.reshape(1, half_dim, 1, seq_len).transpose(0, 2, 3, 1).astype(dtype)
        )

        window_size_llm = (
            self.cfg.vm_cfg.window_size
            // self.cfg.vm_cfg.spatial_merge_size
            // self.cfg.vm_cfg.patch_size
        )
        window_size_patches = (window_size_llm ** 2) * (self.cfg.vm_cfg.spatial_merge_size ** 2)

        mask_shape_nchw = (1, seq_len, 1, seq_len)
        global_mask_np = np.zeros(mask_shape_nchw, dtype=dtype)
        global_mask = graph.constant(global_mask_np.transpose(0, 2, 3, 1))

        large_neg = np.array(np.finfo(np.float32).min, dtype=dtype)
        windowed_mask_np = np.zeros(mask_shape_nchw, dtype=dtype)
        for i in range(seq_len):
            for j in range(seq_len):
                if (i // window_size_patches) != (j // window_size_patches):
                    windowed_mask_np[0, j, 0, i] = large_neg
        windowed_mask = graph.constant(windowed_mask_np.transpose(0, 2, 3, 1))
        return cos_node, sin_node, global_mask, windowed_mask

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """
        Get the custom tessellate params for model's inputs on the MLA.
        """
        input_tessellate_params = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC,
            persistent_mem_name="input",
            dram_shape=None,
        )
        return {0: input_tessellate_params}

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """
        Get the custom tessellate params for model's output on the MLA.
        """
        # Use default tessellate params.
        return {}
