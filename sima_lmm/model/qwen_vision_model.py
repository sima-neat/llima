import logging
import sys
from dataclasses import dataclass

import numpy as np

from afe.apis.defines import TensorDRAMLayout, gen2_target
from afe.backends.backends import Backend
from afe.ir.serializer import save_awesomenet
from afe.ir.defines import Status, TensorValue, TupleValue
from afe.ir.tensor_type import TensorType, ScalarType
from afe.ir.build_node import NodeOrHandle
from sima_lmm.model.base import BaseModel, TensorTessellateParameters, LayerConfiguration
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import (
    SimaBuilder, build_conv, build_activation, activation_type,
    activation_dtype, build_two_stage_layer_norm, create_channel_slice,
    load_tensor_from_source
)
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
    num_layers: int
    include_embeddings: bool
    include_mm_proj: bool

    def gen_onnx_files(self):
        base_name = "vision_model"
        self.create_onnx_builder()
        
        # Create input nodes
        # Qwen2/2.5-VL and Qwen3-VL use the same input shape logic
        patch_feature_size = 3 * self.cfg.vm_cfg.temporal_patch_size * (self.cfg.vm_cfg.patch_size ** 2)
        self._onnx_builder.create_input_node(
            "input", (1, patch_feature_size, 1, self.cfg.vm_cfg.seq_len)
        )

        output_nodes = self._build_onnx_nodes(base_name, self._onnx_builder.input_nodes)
        
        # Create output nodes (handles both single output for Qwen2.5-VL and multiple for Qwen3-VL)
        for node in output_nodes:
            self._onnx_builder.create_output_node(
                self._onnx_builder.get_node_output_name(node),
                (1, self.cfg.lm_cfg.hidden_size, 1, self.cfg.mm_cfg.mm_tokens_per_image)
            )

        self._onnx_builder.create_and_save_model()
        self._onnx_builder = None

    def _build_onnx_nodes(self, base_name: str, input_nodes: list[OnnxNode]) -> list[OnnxNode]:
        if self.cfg.model_type in (VlmArchType.VLM_QWEN3_VL, VlmArchType.VLM_QWEN3_5_VL):
            vision_output = self._build_qwen3_vision_model(
                self.hf_model.vision_model_param_base_name, input_nodes
            )
            return vision_output 
        else: # Qwen 2.5-VL
            vision_output = self._build_qwen2_vision_model(
                self.hf_model.vision_model_param_base_name, input_nodes
            )
            return [vision_output]

    # ------------------------------------------------------------------------
    # Qwen 3 logic
    # ------------------------------------------------------------------------

    def _build_qwen3_vision_model(self, base_name: str, input_nodes: list[OnnxNode]) -> list[OnnxNode]:
        """
        Builds the complete, static Qwen3-VL vision model (encoder + merger).
        Returns a list of output nodes: [final_merger_output, deepstack_output_1, ...]
        """
        cos_table_node, sin_table_node = self._prepare_qwen3_rotary_tables(base_name)
        pos_embed_node = self._prepare_qwen3_position_embedding(base_name)

        encoder_input = self._onnx_builder.build_conv(
            f"{base_name}.patch_embed.proj",
            input_nodes[0],
            is_fc=False,
            src_bias_name=f"{base_name}.patch_embed.proj.bias",
            weight_process_func=self._reshape_qwen_patch_embed_kernel
        )

        encoder_input = self._onnx_builder.build_op(
            f"{base_name}.add_position_embedding",
            [encoder_input, pos_embed_node],
            "Add"
        )

        deepstack_outputs: list[OnnxNode] = []
        hidden_states = encoder_input
        for layer_idx in range(self.num_layers):
            layer_base = f"{base_name}.blocks.{layer_idx}"
            hidden_states = self._build_qwen3_vision_block(
                layer_base,
                hidden_states,
                cos_table_node,
                sin_table_node
            )
            if layer_idx in getattr(self.cfg.vm_cfg, "deepstack_visual_indexes", []):
                ds_index = self.cfg.vm_cfg.deepstack_visual_indexes.index(layer_idx)
                ds_base = f"{base_name}.deepstack_merger_list.{ds_index}"
                deepstack_outputs.append(
                    self._build_qwen3_deepstack_merger(ds_base, hidden_states)
                )

        final_output = self._build_qwen3_merger(f"{base_name}.merger", hidden_states)

        return [final_output, *deepstack_outputs]

    def _build_qwen3_vision_block(
        self,
        base_name: str,
        input_node: OnnxNode,
        cos_table: OnnxNode,
        sin_table: OnnxNode,
    ) -> OnnxNode:
        norm1 = self._onnx_builder.build_layer_norm(
            f"{base_name}.norm1",
            input_node,
            self.cfg.vm_cfg.layer_norm_eps
        )
        attn_out = self._build_qwen_attention(
            f"{base_name}.attn",
            norm1,
            cos_table,
            sin_table
        )
        add1 = self._onnx_builder.build_op(
            f"{base_name}.add1",
            [input_node, attn_out],
            "Add"
        )
        norm2 = self._onnx_builder.build_layer_norm(
            f"{base_name}.norm2",
            add1,
            self.cfg.vm_cfg.layer_norm_eps
        )
        mlp_out = self._build_qwen3_mlp(f"{base_name}.mlp", norm2)
        add2 = self._onnx_builder.build_op(
            f"{base_name}.add2",
            [add1, mlp_out],
            "Add"
        )
        return add2

    def _build_qwen_attention(
        self,
        base_name: str,
        input_node: OnnxNode,
        cos_table: OnnxNode,
        sin_table: OnnxNode,
        attention_mask=None,
    ) -> OnnxNode:
        """
        Builds the shared Qwen vision attention block (Qwen2.5-VL and Qwen3-VL).
        Pass attention_mask for Qwen2.5-VL windowed attention; omit for Qwen3-VL.
        """
        num_heads = self.cfg.vm_cfg.num_attention_heads
        hidden_size = self.cfg.vm_cfg.hidden_size
        head_dim = hidden_size // num_heads
        scaling = head_dim ** -0.5

        qkv_proj = self._onnx_builder.build_conv(f"{base_name}.qkv", input_node)
        q_proj = self._onnx_builder.build_op(
            f"{base_name}.q_slice", [qkv_proj,
                np.array([0], dtype=np.int64),
                np.array([hidden_size], dtype=np.int64),
                np.array([1], dtype=np.int64)],
            "Slice"
        )
        k_proj = self._onnx_builder.build_op(
            f"{base_name}.k_slice", [qkv_proj,
                np.array([hidden_size], dtype=np.int64),
                np.array([2 * hidden_size], dtype=np.int64),
                np.array([1], dtype=np.int64)],
            "Slice"
        )
        v_proj = self._onnx_builder.build_op(
            f"{base_name}.v_slice", [qkv_proj,
                np.array([2 * hidden_size], dtype=np.int64),
                np.array([3 * hidden_size], dtype=np.int64),
                np.array([1], dtype=np.int64)],
            "Slice"
        )

        q_heads = self._onnx_builder.build_split_and_concat(
            f"{base_name}.q_heads", q_proj, num_heads, split_axis=1, concat_axis=2
        )
        k_heads = self._onnx_builder.build_split_and_concat(
            f"{base_name}.k_heads", k_proj, num_heads, split_axis=1, concat_axis=2
        )
        v_heads = self._onnx_builder.build_split_and_concat(
            f"{base_name}.v_heads", v_proj, num_heads, split_axis=1, concat_axis=2
        )

        q_rope = self._build_rotary_emb(f"{base_name}.q_rope", q_heads, cos_table, sin_table)
        k_rope = self._build_rotary_emb(f"{base_name}.k_rope", k_heads, cos_table, sin_table)

        attn_scores = self._onnx_builder.build_op(
            f"{base_name}.attn_scores", [q_rope, k_rope], "Einsum",
            equation="nchw,nchq->nqhw"
        )

        scaled_scores = self._onnx_builder.build_op(
            f"{base_name}.scaled_scores", [attn_scores, np.array(scaling, dtype=np.float32)], "Mul"
        )
        if attention_mask is not None:
            scaled_scores = self._onnx_builder.build_op(
                f"{base_name}.masked_scores", [scaled_scores, attention_mask], "Add"
            )
        softmax_scores = self._onnx_builder.build_op(
            f"{base_name}.softmax", [scaled_scores], "Softmax", axis=1
        )

        context_layer = self._onnx_builder.build_op(
            f"{base_name}.context_layer", [softmax_scores, v_heads], "Einsum",
            equation="nchw,nqhc->nqhw"
        )

        merged = self._onnx_builder.build_split_and_concat(
            f"{base_name}.merged", context_layer,
            num_splits=num_heads, split_axis=2, concat_axis=1
        )

        return self._onnx_builder.build_conv(f"{base_name}.proj", merged)

    def _build_qwen3_mlp(self, base_name: str, input_node: OnnxNode) -> OnnxNode:
        fc1 = self._onnx_builder.build_conv(
            f"{base_name}.linear_fc1",
            input_node,
        )
        act = self._onnx_builder.build_activation(
            f"{base_name}.act",
            fc1,
            self.cfg.vm_cfg.hidden_act
        )
        fc2 = self._onnx_builder.build_conv(
            f"{base_name}.linear_fc2",
            act,
        )
        return fc2

    def _build_qwen3_merger(self, base_name: str, input_node: OnnxNode) -> OnnxNode:
        """
        Builds the Qwen3-VL Patch Merger by fusing the spatial merge and first MLP layer.
        """
        norm_output = self._onnx_builder.build_layer_norm(
            f"{base_name}.norm",
            input_node,
            self.cfg.vm_cfg.layer_norm_eps
        )

        factor = self.cfg.vm_cfg.spatial_merge_size ** 2
        
        fc1 = self._onnx_builder.build_conv(
            f"{base_name}.linear_fc1",
            norm_output,
            is_fc=False, 
            strides=[1, factor], 
            weight_process_func=self._reshape_merger_kernel,
            src_bias_name=f"{base_name}.linear_fc1.bias"
        )

        act = self._onnx_builder.build_activation(
            f"{base_name}.act",
            fc1,
            self.cfg.mm_cfg.hidden_act
        )
        fc2 = self._onnx_builder.build_conv(
            f"{base_name}.linear_fc2",
            act,
        )

        return fc2

    def _build_qwen3_deepstack_merger(self, base_name: str, input_node: OnnxNode) -> OnnxNode:
        """
        Builds a Qwen3-VL deepstack merger that extracts intermediate vision features.
        """
        merge_factor = self.cfg.vm_cfg.spatial_merge_size * self.cfg.vm_cfg.spatial_merge_size
        grouped_seq = self.cfg.vm_cfg.seq_len // merge_factor

        transposed_to_nhwc = self._onnx_builder.build_op(
            f"{base_name}.to_nhwc",
            [input_node],
            "Transpose",
            perm=[0, 2, 3, 1]
        )

        merge_shape_np = np.array([1, 1, grouped_seq, self.cfg.vm_cfg.hidden_size * merge_factor], dtype=np.int64)
        merge_shape_node = self._onnx_builder.create_initializer(f"{base_name}.merge_shape", merge_shape_np)

        reshaped = self._onnx_builder.build_op(
            f"{base_name}.merge_flatten",
            [transposed_to_nhwc, merge_shape_node],
            "Reshape"
        )
        transposed_to_nchw = self._onnx_builder.build_op(
            f"{base_name}.to_nchw",
            [reshaped],
            "Transpose",
            perm=[0, 3, 1, 2]
        )

        norm_output = self._onnx_builder.build_layer_norm(
            f"{base_name}.norm",
            transposed_to_nchw,
            self.cfg.vm_cfg.layer_norm_eps
        )
        fc1 = self._onnx_builder.build_conv(f"{base_name}.linear_fc1", norm_output)
        act = self._onnx_builder.build_activation(f"{base_name}.act", fc1, "gelu")
        return self._onnx_builder.build_conv(f"{base_name}.linear_fc2", act)

    def _prepare_qwen3_rotary_tables(self, base_name: str) -> tuple[OnnxNode, OnnxNode]:
        cos_np, sin_np = self._calc_qwen3_rotary_tables()
        cos_table = self._onnx_builder.create_initializer(
            f"{base_name}.rotary.cos_table",
            cos_np
        )
        sin_table = self._onnx_builder.create_initializer(
            f"{base_name}.rotary.sin_table",
            sin_np
        )
        return cos_table, sin_table

    def _prepare_qwen3_position_embedding(self, base_name: str) -> OnnxNode:
        position_np = self._calc_qwen3_position_embeddings_array(base_name)
        return self._onnx_builder.create_initializer(
            f"{base_name}.pos_embed.static",
            position_np
        )

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

    def _build_qwen2_vision_model(self, base_name: str, input_nodes: list[OnnxNode]) -> OnnxNode:
        """
        Builds the complete, static Qwen 2.5-VL vision model (encoder + merger).
        """
        encoder_input = self._onnx_builder.build_conv(
            f"{base_name}.patch_embed.proj",
            input_nodes[0],
            is_fc=False,
            weight_process_func=self._reshape_qwen_patch_embed_kernel
        )
        (cos_table_node, sin_table_node, 
        global_mask_node, windowed_mask_node) = self._prepare_qwen2_static_inputs()

        for layer_idx in range(self.num_layers):
            layer_base_name = f"{base_name}.blocks.{layer_idx}"
            
            mask_to_use = (
                global_mask_node 
                if layer_idx in self.cfg.vm_cfg.fullatt_block_indexes 
                else windowed_mask_node
            )
            encoder_input = self._build_qwen2_vision_block(
                layer_base_name,
                encoder_input,
                mask_to_use,
                cos_table_node,
                sin_table_node
            )

        if self.include_mm_proj:
            final_output = self._build_qwen2_merger(base_name, encoder_input)
        else:
            final_output = encoder_input
        
        return final_output

    def _build_qwen2_merger(self, base_name: str, input_node: OnnxNode) -> OnnxNode:
        """
        Builds the Qwen 2.5-VL Patch Merger.
        """
        norm_out = self._onnx_builder.build_rms_norm(
            f"{base_name}.merger.ln_q",
            input_node,
            epsilon=self.cfg.vm_cfg.layer_norm_eps,
            weight_offset=0.0
        )

        factor = self.cfg.vm_cfg.spatial_merge_size ** 2
        mlp_fc1 = self._onnx_builder.build_conv(
            f"{base_name}.merger.mlp.0",
            norm_out,
            is_fc=False, 
            strides=[1, factor], 
            weight_process_func=self._reshape_merger_kernel,
            src_bias_name=f"{base_name}.merger.mlp.0.bias"
        )

        mlp_act = self._onnx_builder.build_activation(
            f"{base_name}.act", 
            mlp_fc1, 
            self.cfg.mm_cfg.hidden_act 
        )
        mlp_fc2 = self._onnx_builder.build_conv(
            f"{base_name}.merger.mlp.2", mlp_act
        )
        
        return mlp_fc2

    def _build_qwen2_vision_block(self, base_name: str, input_node: OnnxNode, 
                                attention_mask: OnnxNode, 
                                cos_table: OnnxNode, sin_table: OnnxNode) -> OnnxNode:
        """
        Builds one complete Qwen 2.5-VL Vision Transformer Block in NCHW layout.
        """
        
        norm1_out = self._onnx_builder.build_rms_norm(
            f"{base_name}.norm1",
            input_node,
            epsilon=self.cfg.vm_cfg.layer_norm_eps,
            weight_offset=0.0
        )

        attn_out = self._build_qwen_attention(
            f"{base_name}.attn",
            norm1_out,
            cos_table,
            sin_table,
            attention_mask=attention_mask,
        )

        add1 = self._onnx_builder.build_op(
            f"{base_name}.add1", [input_node, attn_out], "Add"
        )

        norm2_out = self._onnx_builder.build_rms_norm(
            f"{base_name}.norm2",
            add1,
            epsilon=self.cfg.vm_cfg.layer_norm_eps,
            weight_offset=0.0
        )

        mlp_out = self._build_qwen2_mlp(
            f"{base_name}.mlp",
            norm2_out
        )

        add2 = self._onnx_builder.build_op(
            f"{base_name}.add2", [add1, mlp_out], "Add"
        )

        return add2

    def _build_qwen2_mlp(self, base_name: str, input_node: OnnxNode) -> OnnxNode:
        gate_proj = self._onnx_builder.build_conv(f"{base_name}.gate_proj", input_node)
        act = self._onnx_builder.build_activation(
            f"{base_name}.act", gate_proj, self.cfg.vm_cfg.hidden_act 
        )
        up_proj = self._onnx_builder.build_conv(f"{base_name}.up_proj", input_node)
        mul2 = self._onnx_builder.build_op(f"{base_name}.mul2", [act, up_proj], "Mul")
        down_proj = self._onnx_builder.build_conv(f"{base_name}.down_proj", mul2)
        return down_proj


    def _build_rotary_emb(self, base_name: str, input_node: OnnxNode,
                               cos_table: OnnxNode, sin_table: OnnxNode) -> OnnxNode:
        """
        Applies RoPE for tensors in (1, head_dim, num_heads, seq) NCHW layout.
        Splits/concats on axis=1 (head_dim); cos/sin tables are (1, half_dim, 1, seq).
        """
        split_names = [f"{base_name}.split_out_{i}" for i in range(2)]
        split = self._onnx_builder.build_op(
            f"{base_name}.split", [input_node], "Split", axis=1, output_names=split_names
        )
        real_in, imag_in = [split, 0], [split, 1]

        mul_rr = self._onnx_builder.build_op(f"{base_name}.mul_rr", [real_in, cos_table], "Mul")
        mul_ii = self._onnx_builder.build_op(f"{base_name}.mul_ii", [imag_in, sin_table], "Mul")
        real_out = self._onnx_builder.build_op(f"{base_name}.real_out", [mul_rr, mul_ii], "Sub")

        mul_ri = self._onnx_builder.build_op(f"{base_name}.mul_ri", [real_in, sin_table], "Mul")
        mul_ir = self._onnx_builder.build_op(f"{base_name}.mul_ir", [imag_in, cos_table], "Mul")
        imag_out = self._onnx_builder.build_op(f"{base_name}.imag_out", [mul_ri, mul_ir], "Add")

        return self._onnx_builder.build_op(
            f"{base_name}.concat", [real_out, imag_out], "Concat", axis=1
        )

    def _prepare_qwen2_static_inputs(self) -> tuple[OnnxNode, OnnxNode, OnnxNode, OnnxNode]:
        """
        Pre-calculates and creates initializers for RoPE tables and attention masks.
        """
        base_name = self.hf_model.vision_model_param_base_name
        seq_len = self.cfg.vm_cfg.seq_len
        cos_np, sin_np = self._calc_qwen2_vision_rope_tables()
        
        cos_table_4d = cos_np.reshape(1, cos_np.shape[0], 1, -1)
        sin_table_4d = sin_np.reshape(1, cos_np.shape[0], 1, -1)

        cos_table_node = self._onnx_builder.create_initializer(f"{base_name}.cos_table", cos_table_4d)
        sin_table_node = self._onnx_builder.create_initializer(f"{base_name}.sin_table", sin_table_4d)

        window_size_llm = self.cfg.vm_cfg.window_size // self.cfg.vm_cfg.spatial_merge_size // self.cfg.vm_cfg.patch_size
        window_size_patches = (window_size_llm ** 2) * (self.cfg.vm_cfg.spatial_merge_size ** 2)
        # Mask shape matches einsum score layout (1, seq_k, num_heads, seq_q):
        mask_shape = (1, seq_len, 1, seq_len)

        global_mask_np = np.zeros(mask_shape, dtype=np.float32)
        global_mask_node = self._onnx_builder.create_initializer(f"{base_name}.global_mask", global_mask_np)

        large_neg_val = np.finfo(np.float32).min
        windowed_mask_np = np.zeros(mask_shape, dtype=np.float32)
        for i in range(seq_len):
            for j in range(seq_len):
                if (i // window_size_patches) != (j // window_size_patches):
                    windowed_mask_np[0, j, 0, i] = large_neg_val
        windowed_mask_node = self._onnx_builder.create_initializer(f"{base_name}.windowed_mask", windowed_mask_np)

        return cos_table_node, sin_table_node, global_mask_node, windowed_mask_node

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

    def gen_model_sdk_files_directly(
        self,
        layer_cfg: LayerConfiguration,
        log_level: int,
        quantizable: bool,
    ):
        base_name = self.hf_model.vision_model_param_base_name
        g = self._build_sima_nodes(self.hf_model.vision_model_param_base_name, quantizable)
        save_awesomenet(g, self.model_name + (".fp32" if quantizable else ""), str(self.sima_model_sdk_path))

    def _build_sima_nodes(self, base_name: str, quantizable: bool):
        patch_feature_size = (
            3 * self.cfg.vm_cfg.temporal_patch_size * (self.cfg.vm_cfg.patch_size ** 2)
        )
        input_shape = (1, 1, self.cfg.vm_cfg.seq_len, patch_feature_size)

        builder = SimaBuilder(Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target)
        model_input = builder.create_placeholder_node(
            "input", TensorType(activation_type(quantizable), input_shape)
        )
        builder.begin_subnet([model_input])
        mla_input = builder.create_placeholder_node(
            "MLA_0/input", TensorType(activation_type(quantizable), input_shape)
        )

        if self.cfg.model_type in (VlmArchType.VLM_QWEN3_VL, VlmArchType.VLM_QWEN3_5_VL):
            output_nodes = self._build_sima_qwen3_vision_model(builder, base_name, mla_input, quantizable)
            if len(output_nodes) > 1:
                builder.create_tuple_node(output_nodes)
        else:
            self._build_sima_qwen2_vision_model(builder, base_name, mla_input, quantizable)

        mla_node = builder.finish_subnet("MLA_0")
        if activation_type(quantizable) != ScalarType.float32:
            self._cast_sima_subnet_outputs_to_fp32(builder, mla_node)
        return builder.finish(self.model_name)

    def _cast_sima_subnet_outputs_to_fp32(
        self, builder: SimaBuilder, mla_node: NodeOrHandle
    ) -> None:
        match mla_node.get_type().output:
            case TensorValue(value=t):
                if t.scalar == ScalarType.bfloat16:
                    _ = builder.create_cast_node(mla_node, ScalarType.float32, backend=Backend.EV)
            case TupleValue():
                tuple_items = builder.create_tuple_get_item_nodes(mla_node)
                builder.create_tuple_node(
                    [
                        builder.create_cast_node(item, ScalarType.float32, backend=Backend.EV)
                        for item in tuple_items
                    ]
                )

    def _build_sima_qwen3_vision_model(
        self, builder: SimaBuilder, base_name: str, input_node: NodeOrHandle, quantizable: bool
    ) -> list[NodeOrHandle]:
        cos_table, sin_table = self._prepare_sima_qwen3_rotary_tables(builder, base_name, quantizable)
        pos_embed = self._prepare_sima_qwen3_position_embedding(builder, base_name, quantizable)

        hidden_states = build_conv(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.patch_embed.proj", input_node,
            is_fc=False, weight_process_func=self._reshape_qwen_patch_embed_kernel,
            src_bias_name=f"{base_name}.patch_embed.proj.bias",
        )
        hidden_states = builder.create_add_node(hidden_states, pos_embed)

        deepstack_outputs: list[NodeOrHandle] = []
        for layer_idx in range(self.num_layers):
            layer_base = f"{base_name}.blocks.{layer_idx}"
            hidden_states = self._build_sima_qwen3_vision_block(
                builder, layer_base, hidden_states, cos_table, sin_table, quantizable
            )
            if layer_idx in getattr(self.cfg.vm_cfg, "deepstack_visual_indexes", []):
                ds_idx = self.cfg.vm_cfg.deepstack_visual_indexes.index(layer_idx)
                ds_base = f"{base_name}.deepstack_merger_list.{ds_idx}"
                deepstack_outputs.append(
                    self._build_sima_qwen3_deepstack_merger(builder, ds_base, hidden_states, quantizable)
                )

        final_output = self._build_sima_qwen3_merger(
            builder, f"{base_name}.merger", hidden_states, quantizable
        )
        return [final_output, *deepstack_outputs]

    def _build_sima_qwen2_vision_model(
        self, builder: SimaBuilder, base_name: str, input_node: NodeOrHandle, quantizable: bool
    ) -> NodeOrHandle:
        hidden_states = build_conv(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.patch_embed.proj", input_node,
            is_fc=False, weight_process_func=self._reshape_qwen_patch_embed_kernel,
        )
        cos_table, sin_table, global_mask, windowed_mask = self._prepare_sima_qwen2_static_inputs(builder, quantizable)

        for layer_idx in range(self.num_layers):
            layer_base = f"{base_name}.blocks.{layer_idx}"
            mask = (
                global_mask
                if layer_idx in self.cfg.vm_cfg.fullatt_block_indexes
                else windowed_mask
            )
            hidden_states = self._build_sima_qwen2_vision_block(
                builder, layer_base, hidden_states, mask, cos_table, sin_table, quantizable
            )

        if self.include_mm_proj:
            return self._build_sima_qwen2_merger(builder, base_name, hidden_states, quantizable)
        return hidden_states

    def _build_sima_qwen3_vision_block(
        self,
        builder: SimaBuilder,
        base_name: str,
        input_node: NodeOrHandle,
        cos_table: NodeOrHandle,
        sin_table: NodeOrHandle,
        quantizable: bool,
    ) -> NodeOrHandle:
        epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        norm1 = build_two_stage_layer_norm(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.norm1", input_node, -1, epsilon,
        )
        attn = self._build_sima_qwen_attention(
            builder, f"{base_name}.attn", norm1, cos_table, sin_table,
            quantizable=quantizable,
        )
        add1 = builder.create_add_node(input_node, attn)
        norm2 = build_two_stage_layer_norm(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.norm2", add1, -1, epsilon,
        )
        mlp = self._build_sima_qwen3_mlp(builder, f"{base_name}.mlp", norm2, quantizable)
        return builder.create_add_node(add1, mlp)

    def _build_sima_qwen2_vision_block(
        self,
        builder: SimaBuilder,
        base_name: str,
        input_node: NodeOrHandle,
        attention_mask: NodeOrHandle,
        cos_table: NodeOrHandle,
        sin_table: NodeOrHandle,
        quantizable: bool,
    ) -> NodeOrHandle:
        epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        w1 = load_tensor_from_source(f"{base_name}.norm1.weight", self.get_hf_param, self.check_hf_param)
        norm1 = builder.create_rms_norm_node(input_node, epsilon, w1)
        attn = self._build_sima_qwen_attention(
            builder, f"{base_name}.attn", norm1, cos_table, sin_table,
            quantizable=quantizable, attention_mask=attention_mask,
        )
        add1 = builder.create_add_node(input_node, attn)
        w2 = load_tensor_from_source(f"{base_name}.norm2.weight", self.get_hf_param, self.check_hf_param)
        norm2 = builder.create_rms_norm_node(add1, epsilon, w2)
        mlp = self._build_sima_qwen2_mlp(builder, f"{base_name}.mlp", norm2, quantizable)
        return builder.create_add_node(add1, mlp)

    def _build_sima_qwen_attention(
        self,
        builder: SimaBuilder,
        base_name: str,
        input_node: NodeOrHandle,
        cos_table: NodeOrHandle,
        sin_table: NodeOrHandle,
        quantizable: bool,
        attention_mask: NodeOrHandle = None,
    ) -> NodeOrHandle:
        num_heads = self.cfg.vm_cfg.num_attention_heads
        hidden_size = self.cfg.vm_cfg.hidden_size
        head_dim = hidden_size // num_heads
        dtype = activation_dtype(quantizable)
        scaling = np.array([head_dim ** -0.5], dtype=dtype)

        qkv = build_conv(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.qkv", input_node
        )
        q = create_channel_slice(builder, qkv, 0, hidden_size)
        k = create_channel_slice(builder, qkv, hidden_size, 2 * hidden_size)
        v = create_channel_slice(builder, qkv, 2 * hidden_size, 3 * hidden_size)

        q_heads = builder.create_slice_concat_node(q, axis=1, split_axis=3, split_block=num_heads, split_repeat=1)
        k_heads = builder.create_slice_concat_node(k, axis=1, split_axis=3, split_block=num_heads, split_repeat=1)
        v_heads = builder.create_slice_concat_node(v, axis=1, split_axis=3, split_block=num_heads, split_repeat=1)

        q_rope = self._build_sima_rotary_emb(builder, f"{base_name}.q_rope", q_heads, cos_table, sin_table)
        k_rope = self._build_sima_rotary_emb(builder, f"{base_name}.k_rope", k_heads, cos_table, sin_table)

        scores = builder.create_einsum_node(q_rope, k_rope, equation="nhwc,nhqc->nhwq", layout="NHWC")
        scores = builder.create_mul_node(
            scores, builder.create_constant_node(scaling)
        )
        if attention_mask is not None:
            scores = builder.create_add_node(scores, attention_mask)

        probs = builder.create_softmax_node(scores, axis=3)
        context = builder.create_einsum_node(probs, v_heads, equation="nhwc,nhcq->nhwq", layout="NHWC")
        merged = builder.create_slice_concat_node(context, axis=3, split_axis=1, split_block=num_heads, split_repeat=1)

        return build_conv(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.proj", merged
        )

    def _build_sima_rotary_emb(
        self,
        builder: SimaBuilder,
        base_name: str,
        input_node: NodeOrHandle,
        cos_table: NodeOrHandle,
        sin_table: NodeOrHandle,
    ) -> NodeOrHandle:
        head_dim = self.cfg.vm_cfg.hidden_size // self.cfg.vm_cfg.num_attention_heads
        half_dim = head_dim // 2
        real_in = create_channel_slice(builder, input_node, 0, half_dim)
        imag_in = create_channel_slice(builder, input_node, half_dim, head_dim)
        real_out = builder.create_subtract_node(
            builder.create_mul_node(real_in, cos_table),
            builder.create_mul_node(imag_in, sin_table),
        )
        imag_out = builder.create_add_node(
            builder.create_mul_node(real_in, sin_table),
            builder.create_mul_node(imag_in, cos_table),
        )
        return builder.create_concat_node([real_out, imag_out], axis=3)

    def _build_sima_qwen3_mlp(
        self, builder: SimaBuilder, base_name: str, input_node: NodeOrHandle, quantizable: bool
    ) -> NodeOrHandle:
        fc1 = build_conv(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.linear_fc1", input_node
        )
        act = build_activation(builder, fc1, self.cfg.vm_cfg.hidden_act, quantizable)
        return build_conv(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.linear_fc2", act
        )

    def _build_sima_qwen2_mlp(
        self, builder: SimaBuilder, base_name: str, input_node: NodeOrHandle, quantizable: bool
    ) -> NodeOrHandle:
        gate = build_conv(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.gate_proj", input_node
        )
        act = build_activation(builder, gate, self.cfg.vm_cfg.hidden_act, quantizable)
        up = build_conv(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.up_proj", input_node
        )
        gated = builder.create_mul_node(act, up)
        return build_conv(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.down_proj", gated
        )

    def _build_sima_qwen3_merger(
        self, builder: SimaBuilder, base_name: str, input_node: NodeOrHandle, quantizable: bool
    ) -> NodeOrHandle:
        epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        norm = build_two_stage_layer_norm(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.norm", input_node, -1, epsilon,
        )
        factor = self.cfg.vm_cfg.spatial_merge_size ** 2
        fc1 = build_conv(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.linear_fc1", norm,
            is_fc=False, stride=(1, factor),
            weight_process_func=self._reshape_merger_kernel,
            scale_process_func=lambda x: x,
            src_bias_name=f"{base_name}.linear_fc1.bias",
        )
        act = build_activation(builder, fc1, self.cfg.mm_cfg.hidden_act, quantizable)
        return build_conv(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.linear_fc2", act
        )

    def _build_sima_qwen2_merger(
        self, builder: SimaBuilder, base_name: str, input_node: NodeOrHandle, quantizable: bool
    ) -> NodeOrHandle:
        epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        w = load_tensor_from_source(
            f"{base_name}.merger.ln_q.weight", self.get_hf_param, self.check_hf_param
        )
        norm = builder.create_rms_norm_node(input_node, epsilon, w)
        factor = self.cfg.vm_cfg.spatial_merge_size ** 2
        fc1 = build_conv(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.merger.mlp.0", norm,
            is_fc=False, stride=(1, factor),
            weight_process_func=self._reshape_merger_kernel,
            scale_process_func=lambda x: x,
            src_bias_name=f"{base_name}.merger.mlp.0.bias",
        )
        act = build_activation(builder, fc1, self.cfg.mm_cfg.hidden_act, quantizable)
        return build_conv(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.merger.mlp.2", act
        )

    def _build_sima_qwen3_deepstack_merger(
        self, builder: SimaBuilder, base_name: str, input_node: NodeOrHandle, quantizable: bool
    ) -> NodeOrHandle:
        factor = self.cfg.vm_cfg.spatial_merge_size ** 2
        grouped_seq = self.cfg.vm_cfg.seq_len // factor
        hidden = self.cfg.vm_cfg.hidden_size

        # NHWC (1, 1, seq, hidden) → (1, 1, grouped_seq, hidden*factor).
        # In NHWC, consecutive tokens are contiguous in memory, so a plain reshape
        # produces consecutive grouping — same semantics as PyTorch's `.view()`.
        reshaped = builder.create_reshape_node(
            input_node, [1, 1, grouped_seq, hidden * factor]
        )
        epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        norm = build_two_stage_layer_norm(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.norm", reshaped, -1, epsilon,
        )
        fc1 = build_conv(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.linear_fc1", norm
        )
        act = build_activation(builder, fc1, "gelu", quantizable)
        return build_conv(
            builder, self.get_hf_param, self.check_hf_param, f"{base_name}.linear_fc2", act
        )

    def _prepare_sima_qwen3_rotary_tables(
        self, builder: SimaBuilder, base_name: str, quantizable: bool
    ) -> tuple[NodeOrHandle, NodeOrHandle]:
        dtype = activation_dtype(quantizable)
        cos_np, sin_np = self._calc_qwen3_rotary_tables()
        cos_node = builder.create_constant_node(cos_np.transpose(0, 2, 3, 1).astype(dtype))
        sin_node = builder.create_constant_node(sin_np.transpose(0, 2, 3, 1).astype(dtype))
        return cos_node, sin_node

    def _prepare_sima_qwen3_position_embedding(
        self, builder: SimaBuilder, base_name: str, quantizable: bool
    ) -> NodeOrHandle:
        pos_nchw = self._calc_qwen3_position_embeddings_array(base_name)
        pos_nhwc = pos_nchw.transpose(0, 2, 3, 1).astype(activation_dtype(quantizable))
        return builder.create_constant_node(pos_nhwc)

    def _prepare_sima_qwen2_static_inputs(
        self, builder: SimaBuilder, quantizable: bool
    ) -> tuple[NodeOrHandle, NodeOrHandle, NodeOrHandle, NodeOrHandle]:
        seq_len = self.cfg.vm_cfg.seq_len
        dtype = activation_dtype(quantizable)
        cos_np, sin_np = self._calc_qwen2_vision_rope_tables()
        half_dim = cos_np.shape[0]
        cos_node = builder.create_constant_node(
            cos_np.reshape(1, half_dim, 1, seq_len).transpose(0, 2, 3, 1).astype(dtype)
        )
        sin_node = builder.create_constant_node(
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
        global_mask = builder.create_constant_node(global_mask_np.transpose(0, 2, 3, 1))

        large_neg = np.array(np.finfo(np.float32).min, dtype=dtype)
        windowed_mask_np = np.zeros(mask_shape_nchw, dtype=dtype)
        for i in range(seq_len):
            for j in range(seq_len):
                if (i // window_size_patches) != (j // window_size_patches):
                    windowed_mask_np[0, j, 0, i] = large_neg
        windowed_mask = builder.create_constant_node(windowed_mask_np.transpose(0, 2, 3, 1))
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
