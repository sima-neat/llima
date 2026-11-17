import numpy as np
from dataclasses import dataclass

from afe.apis.defines import TensorDRAMLayout

from sima_lmm.model.base import TensorTessellateParameters, LoraGenMode, LayerConfiguration
from sima_lmm.model.model_graph import ModelGraph, Node
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.config.vlm_config import LlmArchType, VlmArchType
from sima_lmm.utils import rope_channel_layout


@dataclass
class LanguagePreModel(LanguagePartBaseModel):
    """Base implementation for the pre cache model of the language model.

    Attributes:
        num_tokens: Number of tokens. Set to a value greater than 1 to consume multiple input tokens
            in one model.
        layer_idx: Transformer layer index.
    """
    num_tokens: int
    layer_idx: int

    def __post_init__(self):
        assert self.num_tokens >= 1
        assert 0 <= self.layer_idx < self.cfg.lm_cfg.num_hidden_layers

    @property
    def _layer_base_name(self) -> str:
        if self.is_dflash_draft:
            return f"layers.{self.layer_idx}"
        base = self.hf_model.language_model_param_base_name
        return base if self.is_eagle3_draft else f"{base}.layers.{self.layer_idx}"

    @property
    def uses_quantized_input_embeddings(self) -> bool:
        return super().uses_quantized_input_embeddings and not self.is_gemma4_mtp_draft

    @property
    def layer_type(self) -> str:
        return self.cfg.lm_cfg.layer_types[self.layer_idx]

    @property
    def _is_proportional_rope_layer(self) -> bool:
        return (
            self.layer_type == "full_attention"
            and self.cfg.lm_cfg.rope_cfg.rope_scaling.rope_type == "proportional"
        )

    @property
    def _head_dim(self) -> int:
        return self.cfg.lm_cfg.attn_cfg.get_head_dim(self.layer_type)

    @property
    def _q_size(self) -> int:
        return self.cfg.lm_cfg.attn_cfg.get_q_size(self.layer_type)

    @property
    def _kv_size(self) -> int:
        return self.cfg.lm_cfg.attn_cfg.get_kv_size(self.layer_type)

    @property
    def _rope_channel_layout(self) -> tuple[int, int, int]:
        """
        Channel layout of this layer's rotation: (head_dim, half_rope_dim, imag_start).

        The rotation pairs channel i with channel imag_start + i for i < half_rope_dim and
        leaves every other channel unchanged. Partial RoPE rotates a prefix of the head and
        passes the tail through; proportional RoPE (Gemma4 full-attention layers) keeps the
        imaginary half at head_dim // 2, so an unrotated segment sits between the halves.

        The freq_real and freq_imag inputs are head_dim-wide cos and sin tables laid out to
        match: the producers duplicate each frequency onto both channels of its pair, negate
        the sin of the real channel i, and fill unrotated channels with the neutral cos 1 and
        sin 0. The producers are
        sima_lmm.utils.calc_freq_real_imag on the host and rope_utils.cpp
        calc_freq_real_imag on the DevKit; both derive the layout from rope_channel_layout.
        """
        head_dim = self._head_dim
        half_rope_dim, imag_start = rope_channel_layout(
            self.cfg.lm_cfg.rope_cfg.get_rope_dimension_count(self.layer_type),
            head_dim,
            self._is_proportional_rope_layer
        )
        return head_dim, half_rope_dim, imag_start

    def _rope_rotate_half_perm(self) -> list[int]:
        """
        Channel permutation of the rotation's imaginary term, HuggingFace's rotate_half without
        its sign: output channel i is data[imag_start + i], output channel imag_start + i is
        data[i], and every unrotated channel maps to itself, where the neutral sin of 0 in
        the uploaded table discards it. The minus sign of the rotation lives in the sin table.

        The MLA executes the permutation as whole channel block moves when half_rope_dim and
        imag_start are multiples of the 16-channel row.
        """
        head_dim, half_rope_dim, imag_start = self._rope_channel_layout
        perm = list(range(head_dim))
        perm[:half_rope_dim] = range(imag_start, imag_start + half_rope_dim)
        perm[imag_start:imag_start + half_rope_dim] = range(half_rope_dim)
        return perm

    def generate_graph(
        self,
        layer_cfg: LayerConfiguration,
        quantizable: bool
    ):
        base_name = self._layer_base_name
        merged_lora = layer_cfg.get("lora", LoraGenMode.LORA_DISABLED) == LoraGenMode.LORA_MERGED
        input_shape = (1, 1, self.num_tokens, self._input_hidden_size)
        scale_shape = (1, 1, self.num_tokens, 1)
        freq_shape = (1, 1, self.num_tokens, self._head_dim)
        input_specs = {"input": input_shape}
        input_dtypes = {}
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            input_dtypes["input"] = np.int8
            input_specs["input_scale"] = scale_shape
        if self.is_eagle3_draft:
            input_specs["hidden_states"] = input_shape
        input_specs.update(freq_real=freq_shape, freq_imag=freq_shape)
        graph = ModelGraph(self, input_specs, quantizable, input_dtypes=input_dtypes)
        inputs = graph.inputs
        mla_input_input = inputs["input"]
        mla_input_freq_real = inputs["freq_real"]
        mla_input_freq_imag = inputs["freq_imag"]

        # Dequantize the selected embedding rows before the first layer consumes them.
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            rms_norm_in = graph.dequant(mla_input_input, inputs["input_scale"])
        else:
            rms_norm_in = mla_input_input

        if self.is_gemma4_mtp_draft and self.layer_idx == 0:
            rms_norm_in = graph.linear("pre_projection", rms_norm_in)

        norm_name = (
            f"{base_name}.operator_norm"
            if self.check_hf_param(f"{base_name}.operator_norm.weight")
            else f"{base_name}.input_layernorm"
        )
        rms_norm = graph.rms_norm(norm_name, rms_norm_in)
        # EAGLE3 draft model additionally normalizes the hidden_states and concatenates.
        if self.is_eagle3_draft:
            hidden_states_norm = graph.rms_norm(f"{base_name}.hidden_norm", inputs["hidden_states"])
            attn_input = graph.concat([rms_norm, hidden_states_norm], 3)
        else:
            attn_input = rms_norm
        rope_cos, rope_sin = mla_input_freq_real, mla_input_freq_imag
        mla_q_result = self._build_attn_query(
            graph,
            f"{base_name}.self_attn",
            attn_input,
            rope_cos,
            rope_sin,
            merged_lora,
        )
        gate_out = None
        if self.cfg.lm_cfg.attn_cfg.attn_output_gate:
            mla_q_out, gate_out = mla_q_result
        else:
            mla_q_out = mla_q_result
        output_nodes = [mla_q_out]
        if not self.cfg.lm_cfg.is_kv_shared_layer(self.layer_idx):
            mla_k_out = self._build_attn_key(
                graph,
                f"{base_name}.self_attn",
                attn_input,
                rope_cos,
                rope_sin,
                merged_lora,
            )
            mla_v_out = self._build_attn_value(
                graph, f"{base_name}.self_attn", attn_input, merged_lora
            )

            if self.cfg.pipeline_cfg.quantize_kv_cache:
                k_quant, k_scale = graph.quant(mla_k_out)
                v_quant, v_scale = graph.quant(mla_v_out)
                output_nodes.extend([k_quant, k_scale, v_quant, v_scale])
            else:
                output_nodes.extend([mla_k_out, mla_v_out])

        if gate_out is not None:
            output_nodes.append(gate_out)
        graph.save(output_nodes)

    def _build_rotary_emb(
        self, graph: ModelGraph, data: Node, rope_cos: Node, rope_sin: Node
    ) -> Node:
        """
        Create nodes that compute rotary embedding as

            data * cos + rotate_half(data) * sin

        This equals the complex multiply of each (real, imaginary) channel pair by
        (cos + i sin) exactly: the permutation only moves channels, the sin table carries the
        rotation's sign, and negation is exact, so both round the same products in bfloat16.
        """
        rotate_half = graph.channel_permute(data, self._rope_rotate_half_perm())
        return graph.add(graph.mul(data, rope_cos), graph.mul(rotate_half, rope_sin))

    def _build_attn_query(
        self,
        graph: ModelGraph,
        base_name: str,
        rms_norm: Node,
        rope_cos: Node,
        rope_sin: Node,
        merged_lora: bool = False,
    ) -> Node | tuple[Node, Node]:
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "q_proj")
        q_proj = graph.linear(
            f"{base_name}.q_proj",
            rms_norm,
            merged_lora=merged_lora,
            q_size=self._q_size,
            kv_size=self._kv_size,
            lora_rank=lora_rank,
        )

        q_norm_name = None
        for suffix in ("q_layernorm", "q_norm"):
            if self.check_hf_param(f"{base_name}.{suffix}.weight"):
                q_norm_name = f"{base_name}.{suffix}"
                break

        if q_norm_name and self.cfg.lm_cfg.arch == LlmArchType.OLMOE:
            q_proj = graph.rms_norm(q_norm_name, q_proj)

        gate_out = None
        if self.cfg.lm_cfg.attn_cfg.attn_output_gate:
            q_fused = graph.split_heads(q_proj, self.cfg.lm_cfg.attn_cfg.num_attention_heads)
            head_dim = self.cfg.lm_cfg.attn_cfg.head_dim
            gate_out = graph.slice(q_fused, start=head_dim, stop=2 * head_dim, axis=3)
            q_proj = graph.slice(q_fused, start=0, stop=head_dim, axis=3)
            gate_out = graph.merge_heads(gate_out)
            reshape1 = q_proj
        elif self.cfg.lm_cfg.attn_cfg.num_attention_heads > 1:
            reshape1 = graph.split_heads(q_proj, self.cfg.lm_cfg.attn_cfg.num_attention_heads)
        else:
            reshape1 = q_proj

        if q_norm_name and self.cfg.lm_cfg.arch != LlmArchType.OLMOE:
            reshape1 = graph.rms_norm(q_norm_name, reshape1)

        rotary_emb = self._build_rotary_emb(graph, reshape1, rope_cos, rope_sin)

        if self.cfg.model_type != VlmArchType.VLM_GEMMA4:
            rotary_emb = graph.mul(
                rotary_emb,
                graph.constant([self._head_dim**-0.5]),
            )
        if gate_out is not None:
            return rotary_emb, gate_out
        return rotary_emb

    def _build_attn_key(
        self,
        graph: ModelGraph,
        base_name: str,
        rms_norm: Node,
        rope_cos: Node,
        rope_sin: Node,
        merged_lora: bool = False,
    ) -> Node:
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "k_proj")
        k_proj = graph.linear(
            f"{base_name}.k_proj",
            rms_norm,
            merged_lora=merged_lora,
            q_size=self._q_size,
            kv_size=self._kv_size,
            lora_rank=lora_rank,
        )

        k_norm_name = None
        for suffix in ("k_layernorm", "k_norm"):
            if self.check_hf_param(f"{base_name}.{suffix}.weight"):
                k_norm_name = f"{base_name}.{suffix}"
                break

        if k_norm_name and self.cfg.lm_cfg.arch == LlmArchType.OLMOE:
            k_proj = graph.rms_norm(k_norm_name, k_proj)

        reshape1 = graph.split_heads(k_proj, self.cfg.lm_cfg.attn_cfg.num_key_value_heads)

        if k_norm_name and self.cfg.lm_cfg.arch != LlmArchType.OLMOE:
            reshape1 = graph.rms_norm(k_norm_name, reshape1)

        rotary_emb = self._build_rotary_emb(graph, reshape1, rope_cos, rope_sin)
        return rotary_emb

    def _build_attn_value(
        self, graph: ModelGraph, base_name: str, input_node: Node, merged_lora: bool = False
    ) -> Node:
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "v_proj")
        v_proj = graph.linear(
            f"{base_name}.v_proj",
            input_node,
            merged_lora=merged_lora,
            q_size=self._q_size,
            kv_size=self._kv_size,
            lora_rank=lora_rank,
        )
        if self.cfg.model_type != VlmArchType.VLM_GEMMA4:
            # Strided cache stores KV heads explicitly instead of flattened into kv_size.
            return graph.split_heads(v_proj, self.cfg.lm_cfg.attn_cfg.num_key_value_heads)

        # Gemma4 applies value RMS norm per KV head before writing V to cache.
        split = graph.split_heads(v_proj, self.cfg.lm_cfg.attn_cfg.num_key_value_heads)
        split = graph.rms_norm(None, split)
        return split

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters] :
        """
        Get the DRAM layouts to use for this model's outputs on the MLA.
        """
        # Define DRAM shape to enable strided KV cache access for kv cache outputs.
        dram_shape = (
            1,
            self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
            self.cfg.pipeline_cfg.max_num_tokens,
            self._head_dim
        )

        k_cache_params = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC16,
            dram_shape=dram_shape
        )

        v_cache_params = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC16,
            dram_shape=dram_shape
        )

        if self.cfg.pipeline_cfg.quantize_kv_cache:
            scale_dram_shape = (
                1,
                self.cfg.lm_cfg.attn_cfg.num_key_value_heads,
                self.cfg.pipeline_cfg.max_num_tokens,
                1
            )
            k_scale_params = TensorTessellateParameters(
                tile_shape=(0, 0, 0, 0),
                enable_mla=True,
                dram_layout=TensorDRAMLayout.HWC16,
                dram_shape=scale_dram_shape
            )
            v_scale_params = TensorTessellateParameters(
                tile_shape=(0, 0, 0, 0),
                enable_mla=True,
                dram_layout=TensorDRAMLayout.HWC16,
                dram_shape=scale_dram_shape
            )
            # Output order: [q, k_quant, k_scale, v_quant, v_scale]
            tessellate_params = {
                1: k_cache_params,
                2: k_scale_params,   # k_scale
                3: v_cache_params,
                4: v_scale_params,   # v_scale
            }
        else:
            # Output order: [q, k, v]
            tessellate_params = {1: k_cache_params, 2: v_cache_params}

        return tessellate_params
