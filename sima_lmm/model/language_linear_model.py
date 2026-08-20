import math
import numpy as np
from dataclasses import dataclass

from afe.apis.defines import gen2_target
from afe.backends.backends import Backend
from afe.ir.defines import Status
from afe.ir.serializer import save_awesomenet
from afe.ir.tensor_type import TensorType, ScalarType
from afe.ir.build_node import NodeOrHandle

from sima_lmm.model.base import (
    LayerConfiguration,
    LoraGenMode,
    TensorTessellateParameters,
)
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import (
    SimaBuilder,
    activation_dtype,
    activation_type,
    build_activation,
    build_conv,
    build_conv_from_dense_with_lora,
)


@dataclass
class LanguageLinearModel(LanguagePartBaseModel):
    """Fused Qwen3.5 Gated DeltaNet layer.

    Inputs are ONNX NCHW:
        - input: (1, hidden, 1, num_tokens)
        - linear_conv_state: (1, linear_conv_dim, 1, linear_conv_kernel_dim - 1)
        - linear_valid_mask group only: (1, 1, 1, num_tokens)
        - linear_delta_state: (1, value_head_dim, num_value_heads, key_head_dim)

    Outputs are ONNX NCHW:
        - hidden: (1, hidden, 1, num_tokens)
        - linear_conv_state_out: (1, linear_conv_dim, 1, num_tokens + kernel - 2)
        - linear_delta_state_out:  (1, value_head_dim, num_value_heads, key_head_dim)

    Direct SimaBuilder inputs are NHWC:
        - input: (1, 1, num_tokens, hidden)
        - input_scale layer 0 with quantized embeddings: (1, 1, num_tokens, 1)
        - linear_conv_state: (1, 1, linear_conv_kernel_dim - 1, linear_conv_dim)
        - linear_valid_mask group only: (1, 1, num_tokens, 1)
        - linear_delta_state: (1, num_value_heads, key_head_dim, value_head_dim)

    Direct SimaBuilder outputs are NHWC:
        - hidden: (1, 1, num_tokens, hidden)
        - linear_conv_state_out: (1, 1, num_tokens + kernel - 2, linear_conv_dim)
        - linear_delta_state_out: (1, num_value_heads, key_head_dim, value_head_dim)
    """

    num_tokens: int
    layer_idx: int

    def __post_init__(self):
        assert self.num_tokens >= 1
        assert self.num_tokens == 1 or self.num_tokens % 32 == 0, (
            "Qwen3.5 grouped linear_attention requires num_tokens divisible by 32."
        )
        assert 0 <= self.layer_idx < self.cfg.lm_cfg.num_hidden_layers
        assert self.cfg.lm_cfg.linear_attn_cfg is not None
        assert self.cfg.lm_cfg.linear_attn_cfg.conv_kernel_dim > 1
        assert self.layer_idx < self.cfg.lm_cfg.num_hidden_layers - 1, (
            "Qwen3.5 linear_attention is not expected on the final layer."
        )

    @property
    def enable_filter_sharing(self) -> bool:
        return self.cfg.pipeline_cfg.enable_filter_sharing

    @property
    def split_mlp(self) -> bool:
        return self.cfg.pipeline_cfg.split_mlp

    def gen_onnx_files(self):
        """Create the ONNX graph for one fused Qwen3.5 linear-attention layer.

        The graph exposes hidden input, convolution state, optional valid mask,
        and recurrent delta state as runtime inputs.
        """
        base_layer = f"{self.hf_model.language_model_param_base_name}.layers.{self.layer_idx}"

        self.create_onnx_builder()
        self._onnx_builder.create_input_node(
            "input", (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens)
        )
        self._onnx_builder.create_input_node(
            "linear_conv_state",
            (
                1,
                self.cfg.lm_cfg.linear_attn_cfg.conv_dim,
                1,
                self.cfg.lm_cfg.linear_attn_cfg.conv_kernel_dim - 1,
            ),
        )
        if self.num_tokens > 1:
            self._onnx_builder.create_input_node("linear_valid_mask", (1, 1, 1, self.num_tokens))
        self._onnx_builder.create_input_node(
            "linear_delta_state",
            (
                1,
                self.cfg.lm_cfg.linear_attn_cfg.value_head_dim,
                self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
                self.cfg.lm_cfg.linear_attn_cfg.key_head_dim,
            ),
        )

        output_nodes = self._build_onnx_nodes(base_layer, self._onnx_builder.input_nodes)
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[0]),
            (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens),
        )
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[1]),
            (
                1,
                self.cfg.lm_cfg.linear_attn_cfg.conv_dim,
                1,
                self.num_tokens + self.cfg.lm_cfg.linear_attn_cfg.conv_kernel_dim - 2,
            ),
        )
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[2]),
            (
                1,
                self.cfg.lm_cfg.linear_attn_cfg.value_head_dim,
                self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
                self.cfg.lm_cfg.linear_attn_cfg.key_head_dim,
            ),
        )
        self._onnx_builder.create_and_save_model()
        self._onnx_builder = None

    def gen_model_sdk_files_directly(
        self,
        layer_cfg: LayerConfiguration,
        log_level: int,
        quantizable: bool,
    ):
        base_layer = f"{self.hf_model.language_model_param_base_name}.layers.{self.layer_idx}"
        merged_lora = layer_cfg.get("lora", LoraGenMode.LORA_DISABLED) == LoraGenMode.LORA_MERGED
        graph = self._build_sima_nodes(base_layer, quantizable, merged_lora)
        save_awesomenet(
            graph,
            self.model_name + (".fp32" if quantizable else ""),
            str(self.sima_model_sdk_path),
        )

    def _sima_constant(
        self, builder: SimaBuilder, value: np.ndarray | float, quantizable: bool
    ) -> NodeOrHandle:
        dtype = activation_dtype(quantizable)
        return builder.create_constant_node(np.asarray(value, dtype=dtype))

    def _build_sima_static_triangular_sums(
        self, builder: SimaBuilder, g: NodeOrHandle, upper: bool, quantizable: bool
    ) -> NodeOrHandle:
        """Build NHWC prefix/suffix sums with one static triangular Einsum."""
        # The ONNX helper stores the mask as (sum_token, output_token). Here the
        # NHWC einsum stores it as (output_token, sum_token), so the triangle flips.
        mask_fn = np.tril if upper else np.triu
        mask = mask_fn(np.ones((self.num_tokens, self.num_tokens), dtype=np.float32))
        mask = np.broadcast_to(
            mask.reshape(1, 1, self.num_tokens, self.num_tokens),
            (
                1,
                self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
                self.num_tokens,
                self.num_tokens,
            ),
        ).copy()
        return builder.create_einsum_node(
            self._sima_constant(builder, mask, quantizable),
            g,
            equation="nhwc,nhcq->nhwq",
            layout="NHWC",
        )

    def _build_sima_global_interval_decay_mask(
        self, builder: SimaBuilder, g: NodeOrHandle, quantizable: bool
    ) -> NodeOrHandle:
        """Build NHWC pairwise decay in lower-triangular query/key orientation."""
        interval_end_mask = np.tril(
            np.ones((self.num_tokens, self.num_tokens), dtype=np.float32)
        )
        interval_start_mask = np.tril(
            np.ones((self.num_tokens, self.num_tokens), dtype=np.float32), k=-1
        )
        interval_start_mask = np.broadcast_to(
            interval_start_mask.reshape(1, 1, self.num_tokens, self.num_tokens),
            (
                1,
                self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
                self.num_tokens,
                self.num_tokens,
            ),
        ).copy()
        interval_end_mask = np.broadcast_to(
            interval_end_mask.reshape(1, 1, self.num_tokens, self.num_tokens),
            (
                1,
                self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
                self.num_tokens,
                self.num_tokens,
            ),
        ).copy()

        interval_start = self._sima_constant(builder, interval_start_mask, quantizable)
        interval_end = self._sima_constant(builder, interval_end_mask, quantizable)

        # Keep g on its native token axis: start_mask[t, i] multiplies g[t].
        masked_g = builder.create_mul_node(interval_start, g)
        interval_sum = builder.create_einsum_node(
            interval_end,
            masked_g,
            equation="nhwc,nhcq->nhwq",
            layout="NHWC",
        )
        decay = builder.create_exp_node(interval_sum)
        return builder.create_mul_node(decay, interval_end)

    def _build_sima_l2norm(
        self, builder: SimaBuilder, input_node: NodeOrHandle, scale: float
    ) -> NodeOrHandle:
        """Normalize NHWC Q/K heads over the last dimension."""
        norm = builder.create_rms_norm_node(
            input_node,
            float(np.float32(1e-6 / self.cfg.lm_cfg.linear_attn_cfg.key_head_dim)),
            np.ones(self.cfg.lm_cfg.linear_attn_cfg.key_head_dim, dtype=np.float32),
        )
        if scale == 1.0:
            return norm
        return builder.create_mul_node(norm, builder.create_constant_node(np.array(scale, dtype=np.float32)))

    def _sima_folded_matrix_mul(
        self, builder: SimaBuilder, left_blocks: list[NodeOrHandle], right_blocks: list[NodeOrHandle]
    ) -> list[NodeOrHandle]:
        """Batch independent NHWC block multiplications by folding blocks into head axis."""
        assert len(left_blocks) == len(right_blocks)
        folded_left = left_blocks[0] if len(left_blocks) == 1 else builder.create_concat_node(left_blocks, 1)
        folded_right = (
            right_blocks[0] if len(right_blocks) == 1 else builder.create_concat_node(right_blocks, 1)
        )
        folded_out = builder.create_einsum_node(
            folded_left,
            folded_right,
            equation="nhwc,nhcq->nhwq",
            layout="NHWC",
        )
        if len(left_blocks) == 1:
            return [folded_out]

        blocks = []
        num_heads = self.cfg.lm_cfg.linear_attn_cfg.num_value_heads
        for block_idx in range(len(left_blocks)):
            blocks.append(
                builder.create_slice_node(
                    folded_out,
                    [block_idx * num_heads],
                    [(block_idx + 1) * num_heads],
                    [1],
                    [1],
                )
            )
        return blocks

    def _build_sima_direct_chunk_inverse(
        self,
        builder: SimaBuilder,
        initial_attn: NodeOrHandle,
        chunk_size: int,
        quantizable: bool,
    ) -> NodeOrHandle:
        """Build the exact NHWC lower-triangular inverse for one folded token block."""
        if chunk_size == 4:
            eye = self._sima_constant(
                builder,
                np.eye(chunk_size, dtype=np.float32).reshape(1, 1, chunk_size, chunk_size),
                quantizable,
            )
            a_squared = builder.create_einsum_node(
                initial_attn,
                initial_attn,
                equation="nhwc,nhcq->nhwq",
                layout="NHWC",
            )
            i_plus_a = builder.create_add_node(initial_attn, eye)
            i_plus_a_squared = builder.create_add_node(a_squared, eye)
            return builder.create_einsum_node(
                i_plus_a_squared,
                i_plus_a,
                equation="nhwc,nhcq->nhwq",
                layout="NHWC",
            )

        half = chunk_size // 2
        top_rows = builder.create_slice_node(initial_attn, [0], [half], [1], [2])
        bottom_rows = builder.create_slice_node(initial_attn, [half], [chunk_size], [1], [2])
        a00 = builder.create_slice_node(top_rows, [0], [half], [1], [3])
        a10 = builder.create_slice_node(bottom_rows, [0], [half], [1], [3])
        a11 = builder.create_slice_node(bottom_rows, [half], [chunk_size], [1], [3])

        inv00 = self._build_sima_direct_chunk_inverse(builder, a00, half, quantizable)
        inv11 = self._build_sima_direct_chunk_inverse(builder, a11, half, quantizable)
        a10_inv00 = builder.create_einsum_node(
            a10,
            inv00,
            equation="nhwc,nhcq->nhwq",
            layout="NHWC",
        )
        inv10 = builder.create_einsum_node(
            inv11,
            a10_inv00,
            equation="nhwc,nhcq->nhwq",
            layout="NHWC",
        )
        inv01 = self._sima_constant(
            builder,
            np.zeros(
                (
                    1,
                    (self.num_tokens // 32) * self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
                    half,
                    half,
                ),
                dtype=np.float32,
            ),
            quantizable,
        )
        top = builder.create_concat_node([inv00, inv01], 3)
        bottom = builder.create_concat_node([inv10, inv11], 3)
        return builder.create_concat_node([top, bottom], 2)

    def _build_sima_block_chunk_inverse(
        self, builder: SimaBuilder, initial_attn: NodeOrHandle, quantizable: bool, block_size: int = 32
    ) -> NodeOrHandle:
        """Build the NHWC grouped-prefill lower-triangular inverse from 32-token blocks."""
        assert self.num_tokens % block_size == 0
        num_blocks = self.num_tokens // block_size

        attn_blocks: dict[tuple[int, int], NodeOrHandle] = {}
        for row in range(num_blocks):
            row_block = builder.create_slice_node(
                initial_attn,
                [row * block_size],
                [(row + 1) * block_size],
                [1],
                [2],
            )
            for col in range(row + 1):
                attn_blocks[(row, col)] = builder.create_slice_node(
                    row_block,
                    [col * block_size],
                    [(col + 1) * block_size],
                    [1],
                    [3],
                )

        inverse_blocks: dict[tuple[int, int], NodeOrHandle] = {}
        diag_blocks = [attn_blocks[(block_idx, block_idx)] for block_idx in range(num_blocks)]
        folded_diag = diag_blocks[0] if len(diag_blocks) == 1 else builder.create_concat_node(diag_blocks, 1)
        folded_diag_inv = self._build_sima_direct_chunk_inverse(
            builder, folded_diag, block_size, quantizable
        )
        num_heads = self.cfg.lm_cfg.linear_attn_cfg.num_value_heads
        for block_idx in range(num_blocks):
            inverse_blocks[(block_idx, block_idx)] = builder.create_slice_node(
                folded_diag_inv,
                [block_idx * num_heads],
                [(block_idx + 1) * num_heads],
                [1],
                [1],
            )

        for span in range(1, num_blocks):
            span_targets = [(row, row - span) for row in range(span, num_blocks)]
            term_specs = [
                (target_idx, row, mid, col)
                for target_idx, (row, col) in enumerate(span_targets)
                for mid in range(col, row)
            ]
            term_products = self._sima_folded_matrix_mul(
                builder,
                [attn_blocks[(row, mid)] for _, row, mid, _ in term_specs],
                [inverse_blocks[(mid, col)] for _, _, mid, col in term_specs],
            )
            grouped_terms: list[list[NodeOrHandle]] = [[] for _ in span_targets]
            for (target_idx, _, _, _), term in zip(term_specs, term_products):
                grouped_terms[target_idx].append(term)

            merged_blocks = []
            for terms in grouped_terms:
                merged = terms[0]
                for term in terms[1:]:
                    merged = builder.create_add_node(merged, term)
                merged_blocks.append(merged)

            span_inverse_blocks = self._sima_folded_matrix_mul(
                builder,
                [inverse_blocks[(row, row)] for row, _ in span_targets],
                merged_blocks,
            )
            for (row, col), inv_block in zip(span_targets, span_inverse_blocks):
                inverse_blocks[(row, col)] = inv_block

        zero_block = self._sima_constant(
            builder,
            np.zeros((1, num_heads, block_size, block_size), dtype=np.float32),
            quantizable,
        )
        row_nodes = []
        for row in range(num_blocks):
            row_nodes.append(
                builder.create_concat_node(
                    [
                        inverse_blocks[(row, col)] if col <= row else zero_block
                        for col in range(num_blocks)
                    ],
                    3,
                )
            )
        return builder.create_concat_node(row_nodes, 2)

    def _build_sima_decode_delta(
        self,
        builder: SimaBuilder,
        query: NodeOrHandle,
        key: NodeOrHandle,
        value: NodeOrHandle,
        beta: NodeOrHandle,
        decay: NodeOrHandle,
        state: NodeOrHandle,
    ) -> tuple[NodeOrHandle, NodeOrHandle]:
        """Build the NHWC single-token recurrent Gated DeltaNet update."""
        state = builder.create_mul_node(state, decay)
        kv_mem = builder.create_batch_matmul_node(key, state, transpose_a=False, transpose_b=False)
        delta = builder.create_subtract_node(value, kv_mem)
        delta = builder.create_mul_node(delta, beta)
        state_add = builder.create_batch_matmul_node(key, delta, transpose_a=True, transpose_b=False)
        state = builder.create_add_node(state, state_add)
        out = builder.create_batch_matmul_node(query, state, transpose_a=False, transpose_b=False)
        return out, state

    def _build_sima_group_delta(
        self,
        builder: SimaBuilder,
        query: NodeOrHandle,
        key: NodeOrHandle,
        query_unscaled: NodeOrHandle,
        key_unscaled: NodeOrHandle,
        value: NodeOrHandle,
        beta: NodeOrHandle,
        g: NodeOrHandle,
        state: NodeOrHandle,
        quantizable: bool,
    ) -> tuple[NodeOrHandle, NodeOrHandle]:
        """Build the grouped prefill computation in NHWC head-major layout."""
        g_cum = self._build_sima_static_triangular_sums(builder, g, upper=True, quantizable=quantizable)
        strict_lower = self._sima_constant(
            builder,
            np.broadcast_to(
                np.tril(np.ones((self.num_tokens, self.num_tokens), dtype=np.float32), k=-1).reshape(
                    1, 1, self.num_tokens, self.num_tokens
                ),
                (
                    1,
                    self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
                    self.num_tokens,
                    self.num_tokens,
                ),
            ).copy(),
            quantizable,
        )
        decay_mask = self._build_sima_global_interval_decay_mask(builder, g, quantizable)

        v_beta = builder.create_mul_node(value, beta)
        k_beta = builder.create_mul_node(key, beta)
        raw_kk = builder.create_einsum_node(
            key_unscaled,
            key_unscaled,
            equation="nhwc,nhqc->nhwq",
            layout="NHWC",
        )
        beta_scaled = builder.create_mul_node(
            beta,
            self._sima_constant(
                builder,
                -1.0 / self.cfg.lm_cfg.linear_attn_cfg.key_head_dim,
                quantizable,
            ),
        )
        kk = builder.create_mul_node(raw_kk, beta_scaled)
        init_attn = builder.create_mul_node(kk, decay_mask)
        init_attn = builder.create_mul_node(init_attn, strict_lower)
        attn = self._build_sima_block_chunk_inverse(builder, init_attn, quantizable)

        value_i = builder.create_einsum_node(
            attn,
            v_beta,
            equation="nhwc,nhcq->nhwq",
            layout="NHWC",
        )
        g_exp = builder.create_exp_node(g_cum)
        k_beta_exp = builder.create_mul_node(k_beta, g_exp)
        k_cumdecay = builder.create_einsum_node(
            attn,
            k_beta_exp,
            equation="nhwc,nhcq->nhwq",
            layout="NHWC",
        )
        v_prime = builder.create_einsum_node(
            k_cumdecay,
            state,
            equation="nhwc,nhcq->nhwq",
            layout="NHWC",
        )
        v_new = builder.create_subtract_node(value_i, v_prime)

        raw_qk = builder.create_einsum_node(
            query_unscaled,
            key_unscaled,
            equation="nhwc,nhqc->nhwq",
            layout="NHWC",
        )
        qk = builder.create_mul_node(
            raw_qk,
            self._sima_constant(
                builder,
                1.0
                / (
                    self.cfg.lm_cfg.linear_attn_cfg.key_head_dim
                    * math.sqrt(self.cfg.lm_cfg.linear_attn_cfg.key_head_dim)
                ),
                quantizable,
            ),
        )
        qk = builder.create_mul_node(qk, decay_mask)
        q_exp = builder.create_mul_node(query, g_exp)
        attn_inter = builder.create_einsum_node(
            q_exp,
            state,
            equation="nhwc,nhcq->nhwq",
            layout="NHWC",
        )
        attn_value = builder.create_einsum_node(
            qk,
            v_new,
            equation="nhwc,nhcq->nhwq",
            layout="NHWC",
        )
        core_attn_out = builder.create_add_node(attn_inter, attn_value)

        suffix_g = self._build_sima_static_triangular_sums(
            builder, g, upper=False, quantizable=quantizable
        )
        suffix_g_exp = builder.create_exp_node(suffix_g)
        final_g_exp = builder.create_slice_node(suffix_g_exp, [0], [1], [1], [2])
        final_decay_mask = builder.create_slice_node(
            suffix_g_exp, [1], [self.num_tokens], [1], [2]
        )
        final_decay_mask_tail = self._sima_constant(
            builder,
            np.ones((1, self.cfg.lm_cfg.linear_attn_cfg.num_value_heads, 1, 1), dtype=np.float32),
            quantizable,
        )
        final_decay_mask = builder.create_concat_node(
            [final_decay_mask, final_decay_mask_tail], 2
        )
        v_new_weighted = builder.create_mul_node(v_new, final_decay_mask)
        state_updates = builder.create_einsum_node(
            key,
            v_new_weighted,
            equation="nhcw,nhcq->nhwq",
            layout="NHWC",
        )
        state_base = builder.create_mul_node(state, final_g_exp)
        linear_delta_state_out = builder.create_add_node(state_base, state_updates)

        return core_attn_out, linear_delta_state_out

    def _build_sima_nodes(
        self, base_layer: str, quantizable: bool, merged_lora: bool = False
    ):
        linear_base = f"{base_layer}.linear_attn"
        repeat = (
            self.cfg.lm_cfg.linear_attn_cfg.num_value_heads
            // self.cfg.lm_cfg.linear_attn_cfg.num_key_heads
        )
        input_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        scale_shape = (1, 1, self.num_tokens, 1)
        conv_state_shape = (
            1,
            1,
            self.cfg.lm_cfg.linear_attn_cfg.conv_kernel_dim - 1,
            self.cfg.lm_cfg.linear_attn_cfg.conv_dim,
        )
        valid_mask_shape = (1, 1, self.num_tokens, 1)
        state_shape = (
            1,
            self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
            self.cfg.lm_cfg.linear_attn_cfg.key_head_dim,
            self.cfg.lm_cfg.linear_attn_cfg.value_head_dim,
        )

        builder = SimaBuilder(Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target)
        input_dtype = (
            ScalarType.int8
            if self.uses_quantized_input_embeddings and self.layer_idx == 0
            else activation_type(quantizable)
        )
        model_input = builder.create_placeholder_node(
            "input", TensorType(input_dtype, input_shape)
        )
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            model_input_scale = builder.create_placeholder_node(
                "input_scale", TensorType(activation_type(quantizable), scale_shape)
            )
        model_conv_state = builder.create_placeholder_node(
            "linear_conv_state", TensorType(activation_type(quantizable), conv_state_shape)
        )
        model_inputs = [model_input]
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            model_inputs.append(model_input_scale)
        model_inputs.append(model_conv_state)
        if self.num_tokens > 1:
            model_valid_mask = builder.create_placeholder_node(
                "linear_valid_mask", TensorType(activation_type(quantizable), valid_mask_shape)
            )
            model_inputs.append(model_valid_mask)
        else:
            model_valid_mask = None
        model_delta_state = builder.create_placeholder_node(
            "linear_delta_state", TensorType(activation_type(quantizable), state_shape)
        )
        model_inputs.append(model_delta_state)

        builder.begin_subnet(model_inputs)

        mla_input = builder.create_placeholder_node(
            "input", TensorType(input_dtype, input_shape)
        )
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            mla_input_scale = builder.create_placeholder_node(
                "input_scale", TensorType(activation_type(quantizable), scale_shape)
            )
        mla_conv_state = builder.create_placeholder_node(
            "linear_conv_state", TensorType(activation_type(quantizable), conv_state_shape)
        )
        if self.num_tokens > 1:
            mla_valid_mask = builder.create_placeholder_node(
                "linear_valid_mask", TensorType(activation_type(quantizable), valid_mask_shape)
            )
        else:
            mla_valid_mask = None
        mla_delta_state = builder.create_placeholder_node(
            "linear_delta_state", TensorType(activation_type(quantizable), state_shape)
        )

        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            residual = builder.create_dynamic_dequant_node(mla_input, mla_input_scale)
        else:
            residual = mla_input

        norm_input = self._build_sima_rms_norm(
            builder, f"{base_layer}.input_layernorm", residual
        )
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(linear_base, "in_proj_qkv")
        mixed_qkv = build_conv_from_dense_with_lora(
            builder,
            self.get_hf_param,
            self.check_hf_param,
            f"{linear_base}.in_proj_qkv",
            norm_input,
            lora_rank=lora_rank,
            merged_lora=merged_lora,
        )
        z = build_conv_from_dense_with_lora(
            builder, self.get_hf_param, self.check_hf_param, f"{linear_base}.in_proj_z", norm_input
        )
        b = build_conv_from_dense_with_lora(
            builder, self.get_hf_param, self.check_hf_param, f"{linear_base}.in_proj_b", norm_input
        )
        a = build_conv_from_dense_with_lora(
            builder, self.get_hf_param, self.check_hf_param, f"{linear_base}.in_proj_a", norm_input
        )

        conv_tail = builder.create_concat_node([mla_conv_state, mixed_qkv], 2)
        linear_conv_state_out = builder.create_slice_node(
            conv_tail,
            [1],
            [self.num_tokens + self.cfg.lm_cfg.linear_attn_cfg.conv_kernel_dim - 1],
            [1],
            [2],
        )
        conv_out = build_conv(
            builder,
            self.get_hf_param,
            self.check_hf_param,
            f"{linear_base}.conv1d",
            conv_tail,
            is_fc=False,
            is_depthwise=True,
        )
        conv_out = build_activation(builder, conv_out, "silu", quantizable)
        if mla_valid_mask is not None:
            conv_out = builder.create_mul_node(conv_out, mla_valid_mask)

        q_flat = builder.create_slice_node(
            conv_out, [0], [self.cfg.lm_cfg.linear_attn_cfg.key_dim], [1], [3]
        )
        k_flat = builder.create_slice_node(
            conv_out,
            [self.cfg.lm_cfg.linear_attn_cfg.key_dim],
            [2 * self.cfg.lm_cfg.linear_attn_cfg.key_dim],
            [1],
            [3],
        )
        v_flat = builder.create_slice_node(
            conv_out,
            [2 * self.cfg.lm_cfg.linear_attn_cfg.key_dim],
            [self.cfg.lm_cfg.linear_attn_cfg.conv_dim],
            [1],
            [3],
        )

        query = builder.create_slice_concat_node(
            q_flat,
            axis=1,
            split_axis=3,
            split_block=self.cfg.lm_cfg.linear_attn_cfg.num_key_heads,
            split_repeat=repeat,
        )
        key = builder.create_slice_concat_node(
            k_flat,
            axis=1,
            split_axis=3,
            split_block=self.cfg.lm_cfg.linear_attn_cfg.num_key_heads,
            split_repeat=repeat,
        )
        value = builder.create_slice_concat_node(
            v_flat,
            axis=1,
            split_axis=3,
            split_block=self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
            split_repeat=1,
        )

        query_unscaled = self._build_sima_l2norm(builder, query, 1.0)
        key_unscaled = self._build_sima_l2norm(builder, key, 1.0)
        query = builder.create_mul_node(
            query_unscaled,
            self._sima_constant(
                builder, 1.0 / self.cfg.lm_cfg.linear_attn_cfg.key_head_dim, quantizable
            ),
        )
        key = builder.create_mul_node(
            key_unscaled,
            self._sima_constant(
                builder,
                1.0 / math.sqrt(self.cfg.lm_cfg.linear_attn_cfg.key_head_dim),
                quantizable,
            ),
        )

        beta = builder.create_sigmoid_node(b)
        beta = builder.create_slice_concat_node(
            beta,
            axis=1,
            split_axis=3,
            split_block=self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
            split_repeat=1,
        )
        if mla_valid_mask is not None:
            beta = builder.create_mul_node(beta, mla_valid_mask)

        dt_bias = self._sima_constant(
            builder,
            self.get_hf_param(f"{linear_base}.dt_bias").astype(np.float32).reshape(1, 1, 1, -1),
            quantizable,
        )
        a_dt = builder.create_add_node(a, dt_bias)
        softplus = builder.create_softplus_node(a_dt)
        neg_a = self._sima_constant(
            builder,
            (-np.exp(self.get_hf_param(f"{linear_base}.A_log").astype(np.float32))).reshape(
                1, 1, 1, -1
            ),
            quantizable,
        )
        g = builder.create_mul_node(softplus, neg_a)
        g = builder.create_slice_concat_node(
            g,
            axis=1,
            split_axis=3,
            split_block=self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
            split_repeat=1,
        )
        if mla_valid_mask is not None:
            g = builder.create_mul_node(g, mla_valid_mask)

        if self.num_tokens == 1:
            decay = builder.create_exp_node(g)
            core_attn_out, linear_delta_state_out = self._build_sima_decode_delta(
                builder,
                query,
                key,
                value,
                beta,
                decay,
                mla_delta_state,
            )
        else:
            core_attn_out, linear_delta_state_out = self._build_sima_group_delta(
                builder,
                query,
                key,
                query_unscaled,
                key_unscaled,
                value,
                beta,
                g,
                mla_delta_state,
                quantizable,
            )

        z_heads = builder.create_slice_concat_node(
            z,
            axis=1,
            split_axis=3,
            split_block=self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
            split_repeat=1,
        )
        core_attn_out = builder.create_rms_norm_node(
            core_attn_out,
            float(np.float32(self.cfg.lm_cfg.rms_norm_eps)),
            self.get_hf_param(f"{linear_base}.norm.weight"),
        )
        z_heads = build_activation(builder, z_heads, "silu", quantizable)
        core_attn_out = builder.create_mul_node(core_attn_out, z_heads)
        core_attn_out = builder.create_slice_concat_node(
            core_attn_out,
            axis=3,
            split_axis=1,
            split_block=self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
            split_repeat=1,
        )

        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(linear_base, "out_proj")
        out_proj = build_conv_from_dense_with_lora(
            builder,
            self.get_hf_param,
            self.check_hf_param,
            f"{linear_base}.out_proj",
            core_attn_out,
            lora_rank=lora_rank,
            merged_lora=merged_lora,
        )
        add1 = builder.create_add_node(residual, out_proj)
        rms_norm2 = self._build_sima_rms_norm(builder, f"{base_layer}.post_attention_layernorm", add1)
        mlp = self._build_sima_mlp(
            builder,
            f"{base_layer}.mlp",
            [rms_norm2, add1],
            quantizable,
            merged_lora=merged_lora,
            with_residual_add=True,
        )
        _ = builder.create_tuple_node([mlp, linear_conv_state_out, linear_delta_state_out])

        mla_node = builder.finish_subnet("MLA_0")
        tuple_items = builder.create_tuple_get_item_nodes(mla_node)
        if activation_type(quantizable) == ScalarType.float32:
            builder.create_tuple_node(tuple_items)
        else:
            builder.create_tuple_node(
                [
                    builder.create_cast_node(item, ScalarType.float32, backend=Backend.EV)
                    for item in tuple_items
                ]
            )
        return builder.finish(self.model_name)

    def _repeat_mask_to_value_heads(self, base_name: str, mask: OnnxNode) -> OnnxNode:
        """Repeat a singleton-head static mask with Slice/Concat for AFE Einsum folding.
        """
        mask_heads = []
        for head_idx in range(self.cfg.lm_cfg.linear_attn_cfg.num_value_heads):
            mask_heads.append(
                self._onnx_builder.build_op(
                    f"{base_name}.head_{head_idx}",
                    [
                        mask,
                        np.array([0], dtype=np.int64),
                        np.array([1], dtype=np.int64),
                        np.array([2], dtype=np.int64),
                    ],
                    "Slice",
                )
            )
        return self._onnx_builder.build_op(f"{base_name}.repeat", mask_heads, "Concat", axis=2)

    def _build_static_triangular_sums(
        self, base_name: str, g: OnnxNode, upper: bool
    ) -> OnnxNode:
        """Build prefix/suffix sums with one static triangular mask and one Einsum.
        """
        mask = np.triu if upper else np.tril
        mask_name = "prefix_mask" if upper else "suffix_mask"
        mask_node = self._onnx_builder.create_initializer(
            f"{base_name}.{mask_name}",
            value=mask(np.ones((self.num_tokens, self.num_tokens), dtype=np.float32)).reshape(
                1, self.num_tokens, 1, self.num_tokens
            ),
        )
        mask_node = self._repeat_mask_to_value_heads(f"{base_name}.{mask_name}", mask_node)
        return self._onnx_builder.build_op(
            f"{base_name}.sum",
            [mask_node, g],
            "Einsum",
            equation="nchw,nqhc->nqhw",
        )

    def _build_global_interval_decay_mask(self, base_name: str, g: OnnxNode) -> OnnxNode:
        """Build pairwise decay from direct interval sums.

        The factorized masks compute sum(g[i + 1 : j + 1]) without subtracting
        large prefix sums, which is much more stable in BF16.
        """
        interval_start_mask = np.triu(
            np.ones((self.num_tokens, self.num_tokens), dtype=np.float32), k=1
        )
        interval_end_mask = np.triu(np.ones((self.num_tokens, self.num_tokens), dtype=np.float32))

        interval_start_node = self._onnx_builder.create_initializer(
            f"{base_name}.start_mask",
            value=interval_start_mask.reshape(1, self.num_tokens, 1, self.num_tokens),
        )
        interval_end_node = self._onnx_builder.create_initializer(
            f"{base_name}.end_mask",
            value=interval_end_mask.reshape(1, self.num_tokens, 1, self.num_tokens),
        )
        interval_end_repeated = self._repeat_mask_to_value_heads(
            f"{base_name}.end_mask", interval_end_node
        )

        masked_g = self._onnx_builder.build_op(
            f"{base_name}.masked_g",
            [interval_start_node, g],
            "Mul",
        )
        interval_sum = self._onnx_builder.build_op(
            f"{base_name}.sum",
            [interval_end_repeated, masked_g],
            "Einsum",
            equation="nchw,nqhc->nqhw",
        )
        decay = self._onnx_builder.build_op(f"{base_name}.exp", [interval_sum], "Exp")
        return self._onnx_builder.build_op(
            f"{base_name}.decay",
            [decay, interval_end_repeated],
            "Mul",
        )

    def _build_l2norm(
        self, base_name: str, input_node: OnnxNode, dim: int, scale: float = 1.0
    ) -> OnnxNode:
        """Normalize Q/K heads with an optional final scale.
        """
        square = self._onnx_builder.build_op(f"{base_name}.mul1", [input_node, input_node], "Mul")
        mean = self._onnx_builder.build_op(
            f"{base_name}.mean", [square], "ReduceMean", axes=[1], keepdims=1
        )
        add = self._onnx_builder.build_op(f"{base_name}.add", [mean, 1e-6 / dim], "Add")
        sqrt = self._onnx_builder.build_op(f"{base_name}.sqrt", [add], "Sqrt")
        div = self._onnx_builder.build_op(f"{base_name}.div", [input_node, sqrt], "Div")
        weight = self._onnx_builder.create_initializer(
            f"{base_name}.weight",
            value=np.full((dim,), scale, dtype=np.float32),
            reshape_str="c->nchw",
        )
        return self._onnx_builder.build_op(f"{base_name}.mul2", [div, weight], "Mul")

    def _folded_matrix_mul(
        self, base_name: str, left_blocks: list[OnnxNode], right_blocks: list[OnnxNode]
    ) -> list[OnnxNode]:
        """Batch independent block multiplications by folding blocks into head axis.

        The result is split back into one output block per input block pair.
        """
        assert len(left_blocks) == len(right_blocks)
        folded_left = (
            left_blocks[0]
            if len(left_blocks) == 1
            else self._onnx_builder.build_op(
                f"{base_name}.left_fold", left_blocks, "Concat", axis=2
            )
        )
        folded_right = (
            right_blocks[0]
            if len(right_blocks) == 1
            else self._onnx_builder.build_op(
                f"{base_name}.right_fold", right_blocks, "Concat", axis=2
            )
        )
        folded_out = self._onnx_builder.build_op(
            f"{base_name}.mul",
            [folded_right, folded_left],
            "Einsum",
            equation="nchw,nqhc->nqhw",
        )
        if len(left_blocks) == 1:
            return [folded_out]

        split = self._onnx_builder.build_split_and_concat(
            f"{base_name}.split",
            folded_out,
            len(left_blocks),
            split_axis=2,
            concat_axis=None,
        )
        return [[split, block_idx] for block_idx in range(len(left_blocks))]

    def _build_direct_chunk_inverse(
        self,
        base_name: str,
        initial_attn: OnnxNode,
        chunk_size: int,
    ) -> OnnxNode:
        """Build the exact upper-triangular inverse for one folded token block.

        This recursively combines exact 4-token inverses and avoids the BF16-sensitive
        repeated-squaring Taylor path.
        """
        if chunk_size == 4:
            eye = np.eye(chunk_size, dtype=np.float32).reshape(1, chunk_size, 1, chunk_size)
            a_squared = self._onnx_builder.build_op(
                f"{base_name}.a_squared",
                [initial_attn, initial_attn],
                "Einsum",
                equation="nchw,nqhc->nqhw",
            )
            i_plus_a = self._onnx_builder.build_op(
                f"{base_name}.i_plus_a", [initial_attn, eye], "Add"
            )
            i_plus_a_squared = self._onnx_builder.build_op(
                f"{base_name}.i_plus_a_squared", [a_squared, eye], "Add"
            )
            return self._onnx_builder.build_op(
                f"{base_name}.base_inv",
                [i_plus_a_squared, i_plus_a],
                "Einsum",
                equation="nchw,nqhc->nqhw",
            )

        half = chunk_size // 2

        top_rows = self._onnx_builder.build_op(
            f"{base_name}.top_rows",
            [
                initial_attn,
                np.array([0], dtype=np.int64),
                np.array([half], dtype=np.int64),
                np.array([1], dtype=np.int64),
            ],
            "Slice",
        )
        bottom_rows = self._onnx_builder.build_op(
            f"{base_name}.bottom_rows",
            [
                initial_attn,
                np.array([half], dtype=np.int64),
                np.array([chunk_size], dtype=np.int64),
                np.array([1], dtype=np.int64),
            ],
            "Slice",
        )

        a00 = self._onnx_builder.build_op(
            f"{base_name}.a00",
            [
                top_rows,
                np.array([0], dtype=np.int64),
                np.array([half], dtype=np.int64),
                np.array([3], dtype=np.int64),
            ],
            "Slice",
        )
        a01 = self._onnx_builder.build_op(
            f"{base_name}.a01",
            [
                top_rows,
                np.array([half], dtype=np.int64),
                np.array([chunk_size], dtype=np.int64),
                np.array([3], dtype=np.int64),
            ],
            "Slice",
        )
        a11 = self._onnx_builder.build_op(
            f"{base_name}.a11",
            [
                bottom_rows,
                np.array([half], dtype=np.int64),
                np.array([chunk_size], dtype=np.int64),
                np.array([3], dtype=np.int64),
            ],
            "Slice",
        )

        inv00 = self._build_direct_chunk_inverse(f"{base_name}.inv00", a00, half)
        inv11 = self._build_direct_chunk_inverse(f"{base_name}.inv11", a11, half)

        a01_inv11 = self._onnx_builder.build_op(
            f"{base_name}.a01_inv11",
            [inv11, a01],
            "Einsum",
            equation="nchw,nqhc->nqhw",
        )
        inv01 = self._onnx_builder.build_op(
            f"{base_name}.inv01",
            [a01_inv11, inv00],
            "Einsum",
            equation="nchw,nqhc->nqhw",
        )
        inv10 = self._onnx_builder.create_initializer(
            f"{base_name}.inv10_zero",
            value=np.zeros(
                (
                    1,
                    half,
                    (self.num_tokens // 32) * self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
                    half,
                ),
                dtype=np.float32,
            ),
        )

        top = self._onnx_builder.build_op(f"{base_name}.top", [inv00, inv01], "Concat", axis=3)
        bottom = self._onnx_builder.build_op(
            f"{base_name}.bottom", [inv10, inv11], "Concat", axis=3
        )
        return self._onnx_builder.build_op(f"{base_name}.all", [top, bottom], "Concat", axis=1)

    def _build_block_chunk_inverse(
        self, base_name: str, initial_attn: OnnxNode, block_size: int = 32
    ) -> OnnxNode:
        """Build the full grouped-prefill triangular inverse from 32-token blocks.

        Diagonal blocks and off-diagonal spans are folded into the head axis to
        reduce graph fragmentation while preserving triangular dependencies.
        """
        assert self.num_tokens % block_size == 0
        num_blocks = self.num_tokens // block_size

        # Slice the upper-triangular attention matrix into 32-token blocks.
        attn_blocks: dict[tuple[int, int], OnnxNode] = {}
        for row in range(num_blocks):
            row_block = self._onnx_builder.build_op(
                f"{base_name}.block_{row}.rows",
                [
                    initial_attn,
                    np.array([row * block_size], dtype=np.int64),
                    np.array([(row + 1) * block_size], dtype=np.int64),
                    np.array([1], dtype=np.int64),
                ],
                "Slice",
            )
            for col in range(row, num_blocks):
                attn_blocks[(row, col)] = self._onnx_builder.build_op(
                    f"{base_name}.block_{row}_{col}.cols",
                    [
                        row_block,
                        np.array([col * block_size], dtype=np.int64),
                        np.array([(col + 1) * block_size], dtype=np.int64),
                        np.array([3], dtype=np.int64),
                    ],
                    "Slice",
                )

        # Fold all diagonal blocks into the head axis and invert them together.
        inverse_blocks: dict[tuple[int, int], OnnxNode] = {}
        diag_blocks = [attn_blocks[(block_idx, block_idx)] for block_idx in range(num_blocks)]
        folded_diag = self._onnx_builder.build_op(
            f"{base_name}.diag.fold", diag_blocks, "Concat", axis=2
        )
        folded_diag_inv = self._build_direct_chunk_inverse(
            f"{base_name}.diag.inverse", folded_diag, block_size
        )
        for block_idx in range(num_blocks):
            head_start = block_idx * self.cfg.lm_cfg.linear_attn_cfg.num_value_heads
            head_end = (block_idx + 1) * self.cfg.lm_cfg.linear_attn_cfg.num_value_heads
            inverse_blocks[(block_idx, block_idx)] = self._onnx_builder.build_op(
                f"{base_name}.diag.unfold.{block_idx}",
                [
                    folded_diag_inv,
                    np.array([head_start], dtype=np.int64),
                    np.array([head_end], dtype=np.int64),
                    np.array([2], dtype=np.int64),
                ],
                "Slice",
            )

        # Build wider off-diagonal spans from already-computed narrower spans.
        for span in range(1, num_blocks):
            span_targets = [(row, row + span) for row in range(num_blocks - span)]

            # First multiply every A(row, mid) @ inverse(mid, col) term in one batch.
            term_specs = [
                (target_idx, row, mid, col)
                for target_idx, (row, col) in enumerate(span_targets)
                for mid in range(row + 1, col + 1)
            ]
            term_products = self._folded_matrix_mul(
                f"{base_name}.span_{span}.terms",
                [attn_blocks[(row, mid)] for _, row, mid, _ in term_specs],
                [inverse_blocks[(mid, col)] for _, _, mid, col in term_specs],
            )

            grouped_terms: list[list[OnnxNode]] = [[] for _ in span_targets]
            for (target_idx, _, _, _), term in zip(term_specs, term_products):
                grouped_terms[target_idx].append(term)

            # Sum all paths that contribute to the same output block.
            merged_blocks = []
            for target_idx, terms in enumerate(grouped_terms):
                merged = terms[0]
                for term_idx, term in enumerate(terms[1:], start=1):
                    merged = self._onnx_builder.build_op(
                        f"{base_name}.span_{span}.target_{target_idx}.sum{term_idx}",
                        [merged, term],
                        "Add",
                    )
                merged_blocks.append(merged)

            # Apply the row diagonal inverse to finish this span.
            span_inverse_blocks = self._folded_matrix_mul(
                f"{base_name}.span_{span}.left_inv",
                [inverse_blocks[(row, row)] for row, _ in span_targets],
                merged_blocks,
            )
            for (row, col), inv_block in zip(span_targets, span_inverse_blocks):
                inverse_blocks[(row, col)] = inv_block

        # Reassemble the full upper-triangular inverse matrix.
        zero_block = self._onnx_builder.create_initializer(
            f"{base_name}.zero_block",
            value=np.zeros(
                (
                    1,
                    block_size,
                    self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
                    block_size,
                ),
                dtype=np.float32,
            ),
        )
        row_nodes = []
        for row in range(num_blocks):
            row_blocks = []
            for col in range(num_blocks):
                block = inverse_blocks[(row, col)] if col >= row else zero_block
                row_blocks.append(block)
            row_nodes.append(
                self._onnx_builder.build_op(
                    f"{base_name}.concat_row_{row}",
                    row_blocks,
                    "Concat",
                    axis=3,
                )
            )

        return self._onnx_builder.build_op(
            f"{base_name}.concat_rows",
            row_nodes,
            "Concat",
            axis=1,
        )

    def _build_decode_delta(
        self,
        base_name: str,
        query: OnnxNode,
        key: OnnxNode,
        value: OnnxNode,
        beta: OnnxNode,
        decay: OnnxNode,
        state: OnnxNode,
    ) -> tuple[OnnxNode, OnnxNode]:
        """Build the single-token recurrent Gated DeltaNet update.

        This is the decode path: update the delta state and produce one attention output.
        """
        state = self._onnx_builder.build_op(f"{base_name}.state_decay", [state, decay], "Mul")
        value_token_major = self._onnx_builder.build_op(
            f"{base_name}.value_token_major", [value], "Transpose", perm=[0, 3, 2, 1]
        )
        kv_mem = self._onnx_builder.build_op(
            f"{base_name}.kv_mem",
            [state, key],
            "Einsum",
            equation="nchw,nchq->nqhw",
        )
        delta = self._onnx_builder.build_op(
            f"{base_name}.delta_sub", [value_token_major, kv_mem], "Sub"
        )
        delta = self._onnx_builder.build_op(f"{base_name}.delta_beta", [delta, beta], "Mul")
        state_add = self._onnx_builder.build_op(f"{base_name}.state_add_mul", [key, delta], "Mul")
        state = self._onnx_builder.build_op(f"{base_name}.state_add", [state, state_add], "Add")
        out = self._onnx_builder.build_op(
            f"{base_name}.out",
            [query, state],
            "Einsum",
            equation="nchw,nchq->nqhw",
        )
        return out, state

    def _build_group_delta(
        self,
        base_name: str,
        query: OnnxNode,
        key: OnnxNode,
        query_unscaled: OnnxNode,
        key_unscaled: OnnxNode,
        value: OnnxNode,
        beta: OnnxNode,
        g: OnnxNode,
        state: OnnxNode,
    ) -> tuple[OnnxNode, OnnxNode]:
        """Build the grouped prefill Gated DeltaNet computation.

        This computes all token outputs and the final recurrent state for the group.
        """
        g_cum = self._build_static_triangular_sums(f"{base_name}.g_cum", g, upper=True)

        strict_lower = np.triu(
            np.ones((self.num_tokens, self.num_tokens), dtype=np.float32), k=1
        ).reshape(1, self.num_tokens, 1, self.num_tokens)
        decay_mask = self._build_global_interval_decay_mask(
            f"{base_name}.interval_decay", g
        )

        v_beta = self._onnx_builder.build_op(f"{base_name}.v_beta", [value, beta], "Mul")
        k_beta = self._onnx_builder.build_op(f"{base_name}.k_beta", [key, beta], "Mul")
        raw_kk = self._onnx_builder.build_op(
            f"{base_name}.raw_kk",
            [key_unscaled, key_unscaled],
            "Einsum",
            equation="nchw,nchq->nqhw",
        )
        beta_scaled = self._onnx_builder.build_op(
            f"{base_name}.beta_scaled",
            [beta, -1.0 / self.cfg.lm_cfg.linear_attn_cfg.key_head_dim],
            "Mul",
        )
        kk = self._onnx_builder.build_op(
            f"{base_name}.kk", [raw_kk, beta_scaled], "Mul"
        )
        init_attn = self._onnx_builder.build_op(
            f"{base_name}.init_attn_decay", [kk, decay_mask], "Mul"
        )
        init_attn = self._onnx_builder.build_op(
            f"{base_name}.init_attn_mask", [init_attn, strict_lower], "Mul"
        )
        attn = self._build_block_chunk_inverse(f"{base_name}.tri_solve", init_attn)

        value_i = self._onnx_builder.build_op(
            f"{base_name}.value", [attn, v_beta], "Einsum", equation="nchw,nqhc->nqhw"
        )
        g_exp = self._onnx_builder.build_op(f"{base_name}.g_exp", [g_cum], "Exp")
        k_beta_exp = self._onnx_builder.build_op(
            f"{base_name}.k_beta_exp", [k_beta, g_exp], "Mul"
        )
        k_cumdecay = self._onnx_builder.build_op(
            f"{base_name}.k_cumdecay",
            [attn, k_beta_exp],
            "Einsum",
            equation="nchw,nqhc->nqhw",
        )
        v_prime = self._onnx_builder.build_op(
            f"{base_name}.v_prime",
            [k_cumdecay, state],
            "Einsum",
            equation="nchw,nchq->nqhw",
        )
        v_new = self._onnx_builder.build_op(f"{base_name}.v_new", [value_i, v_prime], "Sub")

        raw_qk = self._onnx_builder.build_op(
            f"{base_name}.raw_qk",
            [query_unscaled, key_unscaled],
            "Einsum",
            equation="nchw,nchq->nqhw",
        )
        qk = self._onnx_builder.build_op(
            f"{base_name}.qk",
            [
                raw_qk,
                1.0
                / (
                    self.cfg.lm_cfg.linear_attn_cfg.key_head_dim
                    * math.sqrt(self.cfg.lm_cfg.linear_attn_cfg.key_head_dim)
                ),
            ],
            "Mul",
        )
        qk = self._onnx_builder.build_op(
            f"{base_name}.qk_decay", [qk, decay_mask], "Mul"
        )
        q_exp = self._onnx_builder.build_op(f"{base_name}.q_exp", [query, g_exp], "Mul")
        attn_inter = self._onnx_builder.build_op(
            f"{base_name}.attn_inter",
            [q_exp, state],
            "Einsum",
            equation="nchw,nchq->nqhw",
        )
        attn_value = self._onnx_builder.build_op(
            f"{base_name}.attn_value", [qk, v_new], "Einsum", equation="nchw,nqhc->nqhw"
        )
        core_attn_out = self._onnx_builder.build_op(
            f"{base_name}.out", [attn_inter, attn_value], "Add"
        )

        suffix_g = self._build_static_triangular_sums(
            f"{base_name}.state_base.suffix_g", g, upper=False
        )
        suffix_g_exp = self._onnx_builder.build_op(
            f"{base_name}.state_base.suffix_g_exp", [suffix_g], "Exp"
        )

        final_g_exp = self._onnx_builder.build_op(
            f"{base_name}.state_base.final_g_exp",
            [
                suffix_g_exp,
                np.array([0], dtype=np.int64),
                np.array([1], dtype=np.int64),
                np.array([3], dtype=np.int64),
            ],
            "Slice",
        )

        final_decay_mask = self._onnx_builder.build_op(
            f"{base_name}.state_update.final_decay_mask.exp_sliced",
            [
                suffix_g_exp,
                np.array([1], dtype=np.int64),
                np.array([self.num_tokens], dtype=np.int64),
                np.array([3], dtype=np.int64),
            ],
            "Slice",
        )
        final_decay_mask_tail = self._onnx_builder.create_initializer(
            f"{base_name}.state_update.final_decay_mask.tail",
            value=np.ones(
                (1, 1, self.cfg.lm_cfg.linear_attn_cfg.num_value_heads, 1),
                dtype=np.float32,
            ),
        )
        final_decay_mask = self._onnx_builder.build_op(
            f"{base_name}.state_update.final_decay_mask.concat",
            [final_decay_mask, final_decay_mask_tail],
            "Concat",
            axis=3,
        )
        v_new_weighted = self._onnx_builder.build_op(
            f"{base_name}.state_update.v_new_weighted",
            [v_new, final_decay_mask],
            "Mul",
        )
        v_new_weighted = self._onnx_builder.build_op(
            f"{base_name}.state_update.v_new_weighted.token_major",
            [v_new_weighted],
            "Transpose",
            perm=[0, 3, 2, 1],
        )
        state_updates = self._onnx_builder.build_op(
            f"{base_name}.state_update.all",
            [v_new_weighted, key],
            "Einsum",
            equation="nchw,nqhc->nqhw",
        )
        state_base = self._onnx_builder.build_op(
            f"{base_name}.state_base.all", [state, final_g_exp], "Mul"
        )

        linear_delta_state_out = self._onnx_builder.build_op(
            f"{base_name}.state.all", [state_base, state_updates], "Add"
        )
        return core_attn_out, linear_delta_state_out

    def _build_onnx_nodes(self, base_layer: str, input_nodes: list[OnnxNode]) -> list[OnnxNode]:
        linear_base = f"{base_layer}.linear_attn"
        repeat = (
            self.cfg.lm_cfg.linear_attn_cfg.num_value_heads
            // self.cfg.lm_cfg.linear_attn_cfg.num_key_heads
        )
        norm_input = self._build_rms_norm(f"{base_layer}.input_layernorm", input_nodes[0])

        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(linear_base, "in_proj_qkv")
        mixed_qkv = self._onnx_builder.build_conv_from_dense_with_lora(
            f"{linear_base}.in_proj_qkv", norm_input, lora_rank=lora_rank
        )
        z = self._onnx_builder.build_conv_from_dense_with_lora(
            f"{linear_base}.in_proj_z", norm_input
        )
        b = self._onnx_builder.build_conv_from_dense_with_lora(
            f"{linear_base}.in_proj_b", norm_input
        )
        a = self._onnx_builder.build_conv_from_dense_with_lora(
            f"{linear_base}.in_proj_a", norm_input
        )

        conv_tail = self._onnx_builder.build_op(
            f"{linear_base}.conv_tail.concat", [input_nodes[1], mixed_qkv], "Concat", axis=3
        )
        linear_conv_state_out = self._onnx_builder.build_op(
            f"{linear_base}.conv_tail.window",
            [
                conv_tail,
                np.array([1], dtype=np.int64),
                np.array(
                    [self.num_tokens + self.cfg.lm_cfg.linear_attn_cfg.conv_kernel_dim - 1],
                    dtype=np.int64,
                ),
                np.array([3], dtype=np.int64),
            ],
            "Slice",
        )

        w_raw = self.get_hf_param(f"{linear_base}.conv1d.weight")
        if isinstance(w_raw, tuple):
            w_raw = w_raw[1]
        w_conv2d = w_raw.reshape(
            self.cfg.lm_cfg.linear_attn_cfg.conv_dim,
            1,
            1,
            self.cfg.lm_cfg.linear_attn_cfg.conv_kernel_dim,
        )
        w_node = self._onnx_builder.create_initializer(
            f"{linear_base}.conv1d.weight", value=w_conv2d
        )
        conv_inputs = [conv_tail, w_node]
        if self.check_hf_param(f"{linear_base}.conv1d.bias"):
            b_raw = self.get_hf_param(f"{linear_base}.conv1d.bias")
            if isinstance(b_raw, tuple):
                b_raw = b_raw[1]
            b_node = self._onnx_builder.create_initializer(
                f"{linear_base}.conv1d.bias", value=b_raw
            )
            conv_inputs.append(b_node)
        conv_out = self._onnx_builder.build_op(
            f"{linear_base}.depthwise_conv2d",
            conv_inputs,
            "Conv",
            dilations=[1, 1],
            group=self.cfg.lm_cfg.linear_attn_cfg.conv_dim,
            kernel_shape=[1, self.cfg.lm_cfg.linear_attn_cfg.conv_kernel_dim],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        )
        conv_out = self._onnx_builder.build_activation(
            f"{linear_base}.conv_act", conv_out, "silu"
        )
        valid_mask = input_nodes[2] if self.num_tokens > 1 else None
        if valid_mask is not None:
            conv_out = self._onnx_builder.build_op(
                f"{linear_base}.mask.conv_out", [conv_out, valid_mask], "Mul"
            )

        q_flat = self._onnx_builder.build_op(
            f"{linear_base}.q",
            [
                conv_out,
                np.array([0], dtype=np.int64),
                np.array([self.cfg.lm_cfg.linear_attn_cfg.key_dim], dtype=np.int64),
                np.array([1], dtype=np.int64),
            ],
            "Slice",
        )
        k_flat = self._onnx_builder.build_op(
            f"{linear_base}.k",
            [
                conv_out,
                np.array([self.cfg.lm_cfg.linear_attn_cfg.key_dim], dtype=np.int64),
                np.array([2 * self.cfg.lm_cfg.linear_attn_cfg.key_dim], dtype=np.int64),
                np.array([1], dtype=np.int64),
            ],
            "Slice",
        )
        v_flat = self._onnx_builder.build_op(
            f"{linear_base}.v",
            [
                conv_out,
                np.array([2 * self.cfg.lm_cfg.linear_attn_cfg.key_dim], dtype=np.int64),
                np.array([self.cfg.lm_cfg.linear_attn_cfg.conv_dim], dtype=np.int64),
                np.array([1], dtype=np.int64),
            ],
            "Slice",
        )

        query = self._onnx_builder.build_split_expand_concat(
            f"{linear_base}.q_heads.reshape",
            q_flat,
            self.cfg.lm_cfg.linear_attn_cfg.num_key_heads,
            repeat,
            split_axis=1,
            concat_axis=2,
            concat_shape=(
                1,
                self.cfg.lm_cfg.linear_attn_cfg.key_head_dim,
                self.cfg.lm_cfg.linear_attn_cfg.num_key_heads * repeat,
                self.num_tokens,
            ),
        )
        key = self._onnx_builder.build_split_expand_concat(
            f"{linear_base}.k_heads.reshape",
            k_flat,
            self.cfg.lm_cfg.linear_attn_cfg.num_key_heads,
            repeat,
            split_axis=1,
            concat_axis=2,
            concat_shape=(
                1,
                self.cfg.lm_cfg.linear_attn_cfg.key_head_dim,
                self.cfg.lm_cfg.linear_attn_cfg.num_key_heads * repeat,
                self.num_tokens,
            ),
        )
        value = self._onnx_builder.build_split_expand_concat(
            f"{linear_base}.v_heads.reshape",
            v_flat,
            self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
            1,
            split_axis=1,
            concat_axis=2,
            concat_shape=(
                1,
                self.cfg.lm_cfg.linear_attn_cfg.value_head_dim,
                self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
                self.num_tokens,
            ),
        )

        query_unscaled = self._build_l2norm(
            f"{linear_base}.q_l2norm_unscaled",
            query,
            self.cfg.lm_cfg.linear_attn_cfg.key_head_dim,
        )
        key_unscaled = self._build_l2norm(
            f"{linear_base}.k_l2norm_unscaled",
            key,
            self.cfg.lm_cfg.linear_attn_cfg.key_head_dim,
        )
        query = self._onnx_builder.build_op(
            f"{linear_base}.q_scaled_global",
            [query_unscaled, 1.0 / self.cfg.lm_cfg.linear_attn_cfg.key_head_dim],
            "Mul",
        )
        key = self._onnx_builder.build_op(
            f"{linear_base}.k_scaled_global",
            [key_unscaled, 1.0 / math.sqrt(self.cfg.lm_cfg.linear_attn_cfg.key_head_dim)],
            "Mul",
        )

        beta = self._onnx_builder.build_op(f"{linear_base}.beta", [b], "Sigmoid")
        beta = self._onnx_builder.build_op(
            f"{linear_base}.beta.reshape",
            [
                beta,
                np.array(
                    (1, 1, self.cfg.lm_cfg.linear_attn_cfg.num_value_heads, self.num_tokens),
                    dtype=np.int64,
                ),
            ],
            "Reshape",
        )
        if valid_mask is not None:
            beta = self._onnx_builder.build_op(
                f"{linear_base}.mask.beta", [beta, valid_mask], "Mul"
            )
        a_dt = self._onnx_builder.build_op(
            f"{linear_base}.a_dt",
            [
                a,
                self._onnx_builder.create_initializer(
                    f"{linear_base}.dt_bias", value=self.get_hf_param(f"{linear_base}.dt_bias"),
                    reshape_str="c->nchw",
                ),
            ],
            "Add",
        )
        softplus = self._onnx_builder.build_op(f"{linear_base}.softplus", [a_dt], "Softplus")
        neg_a_exp = -np.exp(self.get_hf_param(f"{linear_base}.A_log").astype(np.float32))
        neg_a_exp_node = self._onnx_builder.create_initializer(
            f"{linear_base}.neg_A", value=neg_a_exp, reshape_str="c->nchw"
        )
        g_pre_mask = self._onnx_builder.build_op(
            f"{linear_base}.g_mul", [softplus, neg_a_exp_node], "Mul"
        )
        g_pre_mask = self._onnx_builder.build_op(
            f"{linear_base}.g.reshape",
            [
                g_pre_mask,
                np.array(
                    (1, 1, self.cfg.lm_cfg.linear_attn_cfg.num_value_heads, self.num_tokens),
                    dtype=np.int64,
                ),
            ],
            "Reshape",
        )
        g = g_pre_mask
        if valid_mask is not None:
            g = self._onnx_builder.build_op(f"{linear_base}.mask.g", [g, valid_mask], "Mul")

        state_flat = input_nodes[3] if self.num_tokens > 1 else input_nodes[2]
        state_flat = self._onnx_builder.build_op(
            f"{linear_base}.state.to_khv", [state_flat], "Transpose", perm=[0, 3, 2, 1]
        )

        if self.num_tokens == 1:
            decay = self._onnx_builder.build_op(f"{linear_base}.decay", [g], "Exp")
            core_attn_out, linear_delta_state_out = self._build_decode_delta(
                f"{linear_base}.decode", query, key, value, beta, decay, state_flat
            )
        else:
            core_attn_out, linear_delta_state_out = self._build_group_delta(
                f"{linear_base}.group",
                query,
                key,
                query_unscaled,
                key_unscaled,
                value,
                beta,
                g,
                state_flat,
            )

        z_heads = self._onnx_builder.build_split_expand_concat(
            f"{linear_base}.z_heads.reshape",
            z,
            self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
            1,
            split_axis=1,
            concat_axis=2,
            concat_shape=(
                1,
                self.cfg.lm_cfg.linear_attn_cfg.value_head_dim,
                self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
                self.num_tokens,
            ),
        )
        core_attn_out = self._onnx_builder.build_rms_norm(
            f"{linear_base}.norm", core_attn_out, self.cfg.lm_cfg.rms_norm_eps, 0.0
        )
        z_heads = self._onnx_builder.build_activation(
            f"{linear_base}.norm.gate_silu", z_heads, "silu"
        )
        core_attn_out = self._onnx_builder.build_op(
            f"{linear_base}.norm.mul_gate", [core_attn_out, z_heads], "Mul"
        )
        core_attn_out = self._onnx_builder.build_split_and_concat(
            f"{linear_base}.merge_heads.reshape",
            core_attn_out,
            self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
            split_axis=2,
            concat_axis=1,
        )

        out_proj = self._onnx_builder.build_conv_from_dense_with_lora(
            f"{linear_base}.out_proj", core_attn_out
        )
        add1 = self._onnx_builder.build_op(f"{base_layer}.add1", [input_nodes[0], out_proj], "Add")
        rms_norm2 = self._build_rms_norm(f"{base_layer}.post_attention_layernorm", add1)
        mlp = self._build_onnx_mlp(f"{base_layer}.mlp", [rms_norm2, add1], with_residual_add=True)
        linear_delta_state_out = self._onnx_builder.build_op(
            f"{linear_base}.state.to_vhk",
            [linear_delta_state_out],
            "Transpose",
            perm=[0, 3, 2, 1],
        )

        output_nodes = [mlp, linear_conv_state_out, linear_delta_state_out]
        return output_nodes

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """Use default HWC16 tessellation for hidden, scale, mask, and state inputs."""
        return {}

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """Use default HWC16 tessellation for hidden and recurrent-state outputs."""
        return {}
