import math
import numpy as np
from dataclasses import dataclass

from sima_lmm.model.base import LayerConfiguration, LoraGenMode
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.model.model_graph import ModelGraph, Node


@dataclass
class LanguageLinearModel(LanguagePartBaseModel):
    """Fused Qwen3.5 Gated DeltaNet layer.

    Inputs are NHWC:
        - input: (1, 1, num_tokens, hidden)
        - input_scale layer 0 with quantized embeddings: (1, 1, num_tokens, 1)
        - linear_conv_state: (1, 1, linear_conv_kernel_dim - 1, linear_conv_dim)
        - linear_valid_mask group only: (1, 1, num_tokens, 1)
        - linear_delta_state: (1, num_value_heads, key_head_dim, value_head_dim)

    Outputs are NHWC:
        - hidden: (1, 1, num_tokens, hidden)
        - linear_conv_state_out: (1, 1, num_tokens + kernel - 2, linear_conv_dim)
        - linear_delta_state_out: (1, num_value_heads, key_head_dim, value_head_dim)
    """

    num_tokens: int
    layer_idx: int

    def __post_init__(self):
        assert self.num_tokens >= 1
        assert self.num_tokens in (1, 4, 8, 16) or self.num_tokens % 32 == 0, (
            "Qwen3.5 linear_attention requires 1, 4, 8, 16, or a multiple of 32 tokens."
        )
        assert 0 <= self.layer_idx < self.cfg.lm_cfg.num_hidden_layers
        assert self.cfg.lm_cfg.linear_attn_cfg is not None
        assert self.cfg.lm_cfg.linear_attn_cfg.conv_kernel_dim > 1
        assert self.layer_idx < self.cfg.lm_cfg.num_hidden_layers - 1, (
            "Qwen3.5 linear_attention is not expected on the final layer."
        )

    @property
    def split_mlp(self) -> bool:
        return self.cfg.pipeline_cfg.split_mlp

    @property
    def _delta_block_size(self) -> int:
        return min(self.num_tokens, 32)

    def generate_graph(
        self,
        layer_cfg: LayerConfiguration,
        quantizable: bool,
    ):
        base_layer = f"{self.hf_model.language_model_param_base_name}.layers.{self.layer_idx}"
        merged_lora = layer_cfg.get("lora", LoraGenMode.LORA_DISABLED) == LoraGenMode.LORA_MERGED
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

        input_specs = {"input": input_shape}
        input_dtypes = {}
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            input_dtypes["input"] = np.int8
            input_specs["input_scale"] = scale_shape
        input_specs["linear_conv_state"] = conv_state_shape
        if self.num_tokens > 1:
            input_specs["linear_valid_mask"] = valid_mask_shape
        input_specs["linear_delta_state"] = state_shape
        graph = ModelGraph(self, input_specs, quantizable, input_dtypes=input_dtypes)
        mla_input = graph.inputs["input"]
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            mla_input_scale = graph.inputs["input_scale"]
        mla_conv_state = graph.inputs["linear_conv_state"]
        if self.num_tokens > 1:
            mla_valid_mask = graph.inputs["linear_valid_mask"]
        else:
            mla_valid_mask = None
        mla_delta_state = graph.inputs["linear_delta_state"]

        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            residual = graph.dequant(mla_input, mla_input_scale)
        else:
            residual = mla_input

        norm_input = graph.rms_norm(f"{base_layer}.input_layernorm", residual)
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(linear_base, "in_proj_qkv")
        mixed_qkv = graph.linear(
            f"{linear_base}.in_proj_qkv", norm_input, lora_rank=lora_rank, merged_lora=merged_lora
        )
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(linear_base, "in_proj_z")
        z = graph.linear(
            f"{linear_base}.in_proj_z", norm_input, lora_rank=lora_rank, merged_lora=merged_lora
        )
        a, b = self._build_ab_projections(graph, linear_base, norm_input, merged_lora)

        conv_tail = graph.concat([mla_conv_state, mixed_qkv], 2)
        linear_conv_state_out = graph.slice(
            conv_tail,
            start=1,
            stop=self.num_tokens + self.cfg.lm_cfg.linear_attn_cfg.conv_kernel_dim - 1,
            axis=2,
        )
        conv_out = graph.conv(f"{linear_base}.conv1d", conv_tail, is_depthwise=True)
        conv_out = graph.activation(conv_out, "silu")
        if mla_valid_mask is not None:
            conv_out = graph.mul(conv_out, mla_valid_mask)

        q_flat = graph.slice(
            conv_out, start=0, stop=self.cfg.lm_cfg.linear_attn_cfg.key_dim, axis=3
        )
        k_flat = graph.slice(
            conv_out,
            start=self.cfg.lm_cfg.linear_attn_cfg.key_dim,
            stop=2 * self.cfg.lm_cfg.linear_attn_cfg.key_dim,
            axis=3,
        )
        v_flat = graph.slice(
            conv_out,
            start=2 * self.cfg.lm_cfg.linear_attn_cfg.key_dim,
            stop=self.cfg.lm_cfg.linear_attn_cfg.conv_dim,
            axis=3,
        )

        query = graph.split_heads(
            q_flat, self.cfg.lm_cfg.linear_attn_cfg.num_key_heads, repeat=repeat
        )
        key = graph.split_heads(
            k_flat, self.cfg.lm_cfg.linear_attn_cfg.num_key_heads, repeat=repeat
        )
        value = graph.split_heads(v_flat, self.cfg.lm_cfg.linear_attn_cfg.num_value_heads)

        query_unscaled = self._build_l2norm(graph, query, 1.0)
        key_unscaled = self._build_l2norm(graph, key, 1.0)
        query = graph.mul(
            query_unscaled,
            graph.constant(1.0 / self.cfg.lm_cfg.linear_attn_cfg.key_head_dim),
        )
        key = graph.mul(
            key_unscaled,
            graph.constant(1.0 / math.sqrt(self.cfg.lm_cfg.linear_attn_cfg.key_head_dim)),
        )

        beta = graph.sigmoid(b)
        beta = graph.split_heads(beta, self.cfg.lm_cfg.linear_attn_cfg.num_value_heads)
        if mla_valid_mask is not None:
            beta = graph.mul(beta, mla_valid_mask)

        dt_bias = graph.constant(
            graph.parameter(f"{linear_base}.dt_bias").astype(np.float32).reshape(1, 1, 1, -1),
        )
        a_dt = graph.add(a, dt_bias)
        softplus = graph.softplus(a_dt)
        neg_a = graph.constant(
            (-np.exp(graph.parameter(f"{linear_base}.A_log").astype(np.float32))).reshape(1, 1, 1, -1),
        )
        g = graph.mul(softplus, neg_a)
        g = graph.split_heads(g, self.cfg.lm_cfg.linear_attn_cfg.num_value_heads)
        if mla_valid_mask is not None:
            g = graph.mul(g, mla_valid_mask)

        if self.num_tokens == 1:
            decay = graph.exp(g)
            core_attn_out, linear_delta_state_out = self._build_decode_delta(
                graph,
                query,
                key,
                value,
                beta,
                decay,
                mla_delta_state,
            )
        else:
            core_attn_out, linear_delta_state_out = self._build_group_delta(
                graph,
                query,
                key,
                query_unscaled,
                key_unscaled,
                value,
                beta,
                g,
                mla_delta_state,
            )

        z_heads = graph.split_heads(z, self.cfg.lm_cfg.linear_attn_cfg.num_value_heads)
        core_attn_out = graph.rms_norm(
            f"{linear_base}.norm",
            core_attn_out,
            epsilon=self.cfg.lm_cfg.rms_norm_eps,
        )
        z_heads = graph.activation(z_heads, "silu")
        core_attn_out = graph.mul(core_attn_out, z_heads)
        core_attn_out = graph.merge_heads(core_attn_out)

        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(linear_base, "out_proj")
        out_proj = graph.linear(
            f"{linear_base}.out_proj", core_attn_out, lora_rank=lora_rank, merged_lora=merged_lora
        )
        add1 = graph.add(residual, out_proj)
        rms_norm2 = graph.rms_norm(f"{base_layer}.post_attention_layernorm", add1)
        mlp = self._build_mlp(
            graph,
            f"{base_layer}.mlp",
            [rms_norm2, add1],
            merged_lora=merged_lora,
            with_residual_add=True,
        )
        graph.save([mlp, linear_conv_state_out, linear_delta_state_out])

    def _get_ab_projection_params(
        self, linear_base: str
    ) -> dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]] | None:
        """Join output channels without changing weight precision or scale groups.

        Mixed weight formats retain separate projections because one convolution
        cannot represent both formats without requantizing one of them.
        """
        params = [self.get_hf_param(f"{linear_base}.in_proj_{key}.weight") for key in ("a", "b")]
        quantized = [isinstance(param, tuple) for param in params]
        if quantized[0] != quantized[1]:
            return None
        weights = [param[1] if isinstance(param, tuple) else param for param in params]
        expected = (self.cfg.lm_cfg.linear_attn_cfg.num_value_heads, self.cfg.lm_cfg.hidden_size)
        if any(weight.shape != expected for weight in weights):
            raise ValueError(f"{linear_base}: A/B projection weights must have shape {expected}")
        if weights[0].dtype != weights[1].dtype:
            return None
        joined_weights = np.concatenate(weights, axis=0)
        if quantized[0]:
            # HF/GGUF scales are output-channel first, including grouped INT4.
            scales = [param[0].reshape(expected[0], -1) for param in params]
            if scales[0].shape != scales[1].shape or scales[0].dtype != scales[1].dtype:
                return None
            joined_weights = (np.concatenate(scales, axis=0), joined_weights)
        fused_base = f"{linear_base}.in_proj_ab"
        result = {f"{fused_base}.weight": joined_weights}
        bias_names = [f"{linear_base}.in_proj_{key}.bias" for key in ("a", "b")]
        if any(self.check_hf_param(name) for name in bias_names):
            biases = [
                self.get_hf_param(name) if self.check_hf_param(name)
                else np.zeros(expected[0], dtype=np.float32)
                for name in bias_names
            ]
            result[f"{fused_base}.bias"] = np.concatenate(biases)
        return result

    def _build_ab_projections(
        self,
        graph: ModelGraph,
        linear_base: str,
        norm_input: Node,
        merged_lora: bool,
    ) -> tuple[Node, Node]:
        lora_ranks = {"a": None, "b": None}
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_ranks = {
                key: self.cfg.lm_cfg.get_lora_rank(linear_base, f"in_proj_{key}") for key in lora_ranks
            }
        if any(rank is not None for rank in lora_ranks.values()):
            return tuple(
                graph.linear(
                    f"{linear_base}.in_proj_{key}",
                    norm_input,
                    lora_rank=lora_ranks[key],
                    merged_lora=merged_lora,
                )
                for key in ("a", "b")
            )

        params = self._get_ab_projection_params(linear_base)
        if params is None:
            return tuple(graph.linear(f"{linear_base}.in_proj_{key}", norm_input) for key in ("a", "b"))
        # Fused A/B weights are synthesized locally, outside the model weight source.
        ab = graph._build_conv(
            f"{linear_base}.in_proj_ab",
            norm_input,
            get_param_func=params.__getitem__,
            check_param_func=params.__contains__,
        )
        heads = self.cfg.lm_cfg.linear_attn_cfg.num_value_heads
        return (
            graph.slice(ab, start=0, stop=heads, axis=3),
            graph.slice(ab, start=heads, stop=2 * heads, axis=3),
        )

    def _build_static_triangular_sums(
        self, graph: ModelGraph, g: Node, upper: bool
    ) -> Node:
        """Build NHWC prefix/suffix sums with a static triangular matrix multiplication."""
        # The mask axes are (output_token, sum_token).
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
        return graph.matmul(graph.constant(mask), g)

    def _build_global_interval_decay_mask(
        self, graph: ModelGraph, g: Node
    ) -> Node:
        """Build NHWC pairwise decay in lower-triangular query/key orientation."""
        interval_end_mask = np.tril(np.ones((self.num_tokens, self.num_tokens), dtype=np.float32))
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

        interval_start = graph.constant(interval_start_mask)
        interval_end = graph.constant(interval_end_mask)

        # Keep g on its native token axis: start_mask[t, i] multiplies g[t].
        masked_g = graph.mul(interval_start, g)
        interval_sum = graph.matmul(interval_end, masked_g)
        decay = graph.exp(interval_sum)
        return graph.mul(decay, interval_end)

    def _build_l2norm(
        self, graph: ModelGraph, input_node: Node, scale: float
    ) -> Node:
        """Normalize NHWC Q/K heads over the last dimension."""
        norm = graph.rms_norm(
            None,
            input_node,
            epsilon=1e-6 / self.cfg.lm_cfg.linear_attn_cfg.key_head_dim,
        )
        if scale == 1.0:
            return norm
        return graph.mul(
            norm, graph.constant(np.array(scale, dtype=np.float32), dtype=np.float32)
        )

    def _folded_matrix_mul(
        self, graph: ModelGraph, left_blocks: list[Node], right_blocks: list[Node]
    ) -> list[Node]:
        """Batch independent NHWC block multiplications by folding blocks into head axis."""
        assert len(left_blocks) == len(right_blocks)
        folded_left = (
            left_blocks[0] if len(left_blocks) == 1 else graph.concat(left_blocks, 1)
        )
        folded_right = (
            right_blocks[0] if len(right_blocks) == 1 else graph.concat(right_blocks, 1)
        )
        folded_out = graph.matmul(folded_left, folded_right)
        if len(left_blocks) == 1:
            return [folded_out]

        blocks = []
        num_heads = self.cfg.lm_cfg.linear_attn_cfg.num_value_heads
        for block_idx in range(len(left_blocks)):
            blocks.append(
                graph.slice(
                    folded_out, start=block_idx * num_heads, stop=(block_idx + 1) * num_heads, axis=1
                )
            )
        return blocks

    def _build_direct_chunk_inverse(
        self,
        graph: ModelGraph,
        initial_attn: Node,
        chunk_size: int,
    ) -> Node:
        """Build the exact NHWC lower-triangular inverse for one folded token block."""
        if chunk_size == 4:
            eye = graph.constant(
                np.eye(chunk_size, dtype=np.float32).reshape(1, 1, chunk_size, chunk_size),
            )
            a_squared = graph.matmul(initial_attn, initial_attn)
            i_plus_a = graph.add(initial_attn, eye)
            i_plus_a_squared = graph.add(a_squared, eye)
            return graph.matmul(i_plus_a_squared, i_plus_a)

        half = chunk_size // 2
        top_rows = graph.slice(initial_attn, start=0, stop=half, axis=2)
        bottom_rows = graph.slice(initial_attn, start=half, stop=chunk_size, axis=2)
        a00 = graph.slice(top_rows, start=0, stop=half, axis=3)
        a10 = graph.slice(bottom_rows, start=0, stop=half, axis=3)
        a11 = graph.slice(bottom_rows, start=half, stop=chunk_size, axis=3)

        inv00 = self._build_direct_chunk_inverse(graph, a00, half)
        inv11 = self._build_direct_chunk_inverse(graph, a11, half)
        a10_inv00 = graph.matmul(a10, inv00)
        inv10 = graph.matmul(inv11, a10_inv00)
        inv01 = graph.constant(
            np.zeros(
                (
                    1,
                    (self.num_tokens // self._delta_block_size)
                    * self.cfg.lm_cfg.linear_attn_cfg.num_value_heads,
                    half,
                    half,
                ),
                dtype=np.float32,
            ),
        )
        top = graph.concat([inv00, inv01], 3)
        bottom = graph.concat([inv10, inv11], 3)
        return graph.concat([top, bottom], 2)

    def _build_block_chunk_inverse(
        self, graph: ModelGraph, initial_attn: Node, block_size: int = 32
    ) -> Node:
        """Build the NHWC grouped lower-triangular inverse from fixed-size blocks."""
        assert self.num_tokens % block_size == 0
        num_blocks = self.num_tokens // block_size

        attn_blocks: dict[tuple[int, int], Node] = {}
        for row in range(num_blocks):
            row_block = graph.slice(
                initial_attn, start=row * block_size, stop=(row + 1) * block_size, axis=2
            )
            for col in range(row + 1):
                attn_blocks[(row, col)] = graph.slice(
                    row_block, start=col * block_size, stop=(col + 1) * block_size, axis=3
                )

        inverse_blocks: dict[tuple[int, int], Node] = {}
        diag_blocks = [attn_blocks[(block_idx, block_idx)] for block_idx in range(num_blocks)]
        folded_diag = diag_blocks[0] if len(diag_blocks) == 1 else graph.concat(diag_blocks, 1)
        folded_diag_inv = self._build_direct_chunk_inverse(
            graph, folded_diag, block_size
        )
        if num_blocks == 1:
            return folded_diag_inv
        num_heads = self.cfg.lm_cfg.linear_attn_cfg.num_value_heads
        for block_idx in range(num_blocks):
            inverse_blocks[(block_idx, block_idx)] = graph.slice(
                folded_diag_inv, start=block_idx * num_heads, stop=(block_idx + 1) * num_heads, axis=1
            )

        for span in range(1, num_blocks):
            span_targets = [(row, row - span) for row in range(span, num_blocks)]
            term_specs = [
                (target_idx, row, mid, col)
                for target_idx, (row, col) in enumerate(span_targets)
                for mid in range(col, row)
            ]
            term_products = self._folded_matrix_mul(
                graph,
                [attn_blocks[(row, mid)] for _, row, mid, _ in term_specs],
                [inverse_blocks[(mid, col)] for _, _, mid, col in term_specs],
            )
            grouped_terms: list[list[Node]] = [[] for _ in span_targets]
            for (target_idx, _, _, _), term in zip(term_specs, term_products):
                grouped_terms[target_idx].append(term)

            merged_blocks = []
            for terms in grouped_terms:
                merged = terms[0]
                for term in terms[1:]:
                    merged = graph.add(merged, term)
                merged_blocks.append(merged)

            span_inverse_blocks = self._folded_matrix_mul(
                graph,
                [inverse_blocks[(row, row)] for row, _ in span_targets],
                merged_blocks,
            )
            for (row, col), inv_block in zip(span_targets, span_inverse_blocks):
                inverse_blocks[(row, col)] = inv_block

        zero_block = graph.constant(
            np.zeros((1, num_heads, block_size, block_size), dtype=np.float32),
        )
        row_nodes = []
        for row in range(num_blocks):
            row_nodes.append(
                graph.concat(
                    [
                        inverse_blocks[(row, col)] if col <= row else zero_block
                        for col in range(num_blocks)
                    ],
                    3,
                )
            )
        return graph.concat(row_nodes, 2)

    def _build_decode_delta(
        self,
        graph: ModelGraph,
        query: Node,
        key: Node,
        value: Node,
        beta: Node,
        decay: Node,
        state: Node,
    ) -> tuple[Node, Node]:
        """Build the NHWC single-token recurrent Gated DeltaNet update."""
        state = graph.mul(state, decay)
        kv_mem = graph.matmul(key, state)
        delta = graph.sub(value, kv_mem)
        delta = graph.mul(delta, beta)
        state_add = graph.matmul(key, delta, transpose_a=True)
        state = graph.add(state, state_add)
        out = graph.matmul(query, state)
        return out, state

    def _build_group_delta(
        self,
        graph: ModelGraph,
        query: Node,
        key: Node,
        query_unscaled: Node,
        key_unscaled: Node,
        value: Node,
        beta: Node,
        g: Node,
        state: Node,
    ) -> tuple[Node, Node]:
        """Build the grouped prefill computation in NHWC head-major layout."""
        g_cum = self._build_static_triangular_sums(graph, g, upper=True)
        strict_lower = graph.constant(
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
        )
        decay_mask = self._build_global_interval_decay_mask(graph, g)

        v_beta = graph.mul(value, beta)
        k_beta = graph.mul(key, beta)
        raw_kk = graph.matmul(key_unscaled, key_unscaled, transpose_b=True)
        beta_scaled = graph.mul(
            beta,
            graph.constant(-1.0 / self.cfg.lm_cfg.linear_attn_cfg.key_head_dim),
        )
        kk = graph.mul(raw_kk, beta_scaled)
        init_attn = graph.mul(kk, decay_mask)
        init_attn = graph.mul(init_attn, strict_lower)
        attn = self._build_block_chunk_inverse(
            graph,
            init_attn,
            block_size=self._delta_block_size,
        )

        value_i = graph.matmul(attn, v_beta)
        g_exp = graph.exp(g_cum)
        k_beta_exp = graph.mul(k_beta, g_exp)
        k_cumdecay = graph.matmul(attn, k_beta_exp)
        v_prime = graph.matmul(k_cumdecay, state)
        v_new = graph.sub(value_i, v_prime)

        raw_qk = graph.matmul(query_unscaled, key_unscaled, transpose_b=True)
        qk = graph.mul(
            raw_qk,
            graph.constant(
                1.0
                / (
                    self.cfg.lm_cfg.linear_attn_cfg.key_head_dim
                    * math.sqrt(self.cfg.lm_cfg.linear_attn_cfg.key_head_dim)
                ),
            ),
        )
        qk = graph.mul(qk, decay_mask)
        q_exp = graph.mul(query, g_exp)
        attn_inter = graph.matmul(q_exp, state)
        attn_value = graph.matmul(qk, v_new)
        core_attn_out = graph.add(attn_inter, attn_value)

        suffix_g = self._build_static_triangular_sums(
            graph, g, upper=False
        )
        suffix_g_exp = graph.exp(suffix_g)
        final_g_exp = graph.slice(suffix_g_exp, start=0, stop=1, axis=2)
        final_decay_mask = graph.slice(suffix_g_exp, start=1, stop=self.num_tokens, axis=2)
        final_decay_mask_tail = graph.constant(
            np.ones((1, self.cfg.lm_cfg.linear_attn_cfg.num_value_heads, 1, 1), dtype=np.float32),
        )
        final_decay_mask = graph.concat([final_decay_mask, final_decay_mask_tail], 2)
        v_new_weighted = graph.mul(v_new, final_decay_mask)
        state_updates = graph.matmul(key, v_new_weighted, transpose_a=True)
        state_base = graph.mul(state, final_g_exp)
        linear_delta_state_out = graph.add(state_base, state_updates)

        return core_attn_out, linear_delta_state_out
