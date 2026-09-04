import numpy as np
from dataclasses import dataclass

from afe.apis.defines import gen2_target
from afe.ir.defines import Status
from afe.ir.serializer import save_awesomenet
from afe.ir.build_node import TopKRetType
from afe.ir.tensor_type import TensorType

from sima_lmm.config.vlm_config import LlmArchType
from sima_lmm.model.base import LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import (
    SimaBuilder, activation_type, build_conv_from_dense_with_lora
)


@dataclass
class LanguageMoeRouterModel(LanguagePartBaseModel):
    """Router for a MoE layer: o_proj -> residual -> norm -> gating logits -> TopK
    (+ softmax over the k selected), all on the MLA. Outputs (router_values,
    router_indices, residual h, norm(h)); experts consume norm(h) directly.
    """
    num_tokens: int
    layer_idx: int

    def __post_init__(self):
        assert self.num_tokens >= 1
        assert 0 <= self.layer_idx < self.cfg.lm_cfg.num_hidden_layers
        assert self.cfg.lm_cfg.moe_cfg is not None

    @property
    def enable_filter_sharing(self) -> bool:
        return self.cfg.pipeline_cfg.enable_filter_sharing

    @property
    def layer_type(self) -> str:
        return self.cfg.lm_cfg.layer_types[self.layer_idx]

    @property
    def _layer_base_name(self) -> str:
        base = self.hf_model.language_model_param_base_name
        return f"{base}.layers.{self.layer_idx}"

    def gen_onnx_files(self):
        base_name = self._layer_base_name
        top_k = self.cfg.lm_cfg.moe_cfg.num_experts_per_tok
        self.create_onnx_builder()
        self._onnx_builder.create_input_node(
            "input", (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens)
        )
        self._onnx_builder.create_input_node(
            "self_attn",
            (1, self.cfg.lm_cfg.attn_cfg.get_q_size(self.layer_type), 1, self.num_tokens),
        )
        router_values, topk, residual, norm_hidden = self._build_onnx_nodes(
            base_name, self._onnx_builder.input_nodes
        )
        # Routing weights: softmax over the k selected experts.
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_names(router_values)[0],
            (1, top_k, 1, self.num_tokens),
        )
        # Selected expert indices (TopK indices, int64).
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_names(topk)[1],
            (1, top_k, 1, self.num_tokens), dtype=np.int64,
        )
        # Residual h = input + o_proj(self_attn), consumed by the combine.
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(residual),
            (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens),
        )
        # norm(h): experts consume this directly (computed once, shared).
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(norm_hidden),
            (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens),
        )
        self._onnx_builder.create_and_save_model()

        self._onnx_builder = None

    def _build_onnx_nodes(
        self, base_name: str, input_nodes: list[OnnxNode]
    ) -> tuple[OnnxNode, OnnxNode, OnnxNode, OnnxNode]:
        attn_out_name = (
            "out_proj"
            if self.check_hf_param(f"{base_name}.self_attn.out_proj.weight")
            else "o_proj"
        )
        o_proj = self._onnx_builder.build_conv_from_dense_with_lora(
            f"{base_name}.self_attn.{attn_out_name}", input_nodes[1], None
        )
        # h = input + o_proj(self_attn): residual the combine adds back.
        residual = self._onnx_builder.build_op(
            f"{base_name}.add1", [input_nodes[0], o_proj], "Add"
        )
        x = self._build_rms_norm(f"{base_name}.post_attention_layernorm", residual)

        # Gating linear -> per-expert logits. OLMoE uses mlp.gate; others mlp.router.
        router_weight_name = (
            f"{base_name}.mlp.gate"
            if self.check_hf_param(f"{base_name}.mlp.gate.weight")
            else f"{base_name}.mlp.router"
        )
        logits = self._onnx_builder.build_conv_from_dense_with_lora(
            router_weight_name, x, None
        )
        top_k = self.cfg.lm_cfg.moe_cfg.num_experts_per_tok

        def _build_topk(scores: OnnxNode) -> OnnxNode:
            return self._onnx_builder.build_op(
                f"{base_name}.mlp.router_topk",
                [scores, np.array([top_k], dtype=np.int64)],
                "TopK",
                axis=1, largest=1, sorted=1,
                output_names=[
                    f"{base_name}.mlp.router_topk.values",
                    f"{base_name}.mlp.router_topk.indices",
                ],
            )

        # Both orders pick the same top-k (indices from TopK); only the weights differ.
        if self.cfg.lm_cfg.arch == LlmArchType.OLMOE:
            # OLMoE: softmax over all experts, then top-k (no renorm). TopK values ARE
            # the weights; return the node directly (a wrapper spills to EV74).
            probs = self._onnx_builder.build_op(
                f"{base_name}.mlp.router_softmax", [logits], "Softmax", axis=1
            )
            topk = _build_topk(probs)
            router_values = topk
        else:
            # gpt_oss: top-k first, then softmax over the k selected.
            topk = _build_topk(logits)
            router_values = self._onnx_builder.build_op(
                f"{base_name}.mlp.router_weights", [[topk, 0]], "Softmax", axis=1
            )
        # router_values aligned with topk indices; x = norm(h) for experts.
        return router_values, topk, residual, x

    def gen_model_sdk_files_directly(
        self,
        layer_cfg: LayerConfiguration,
        log_level: int,
        quantizable: bool,
    ):
        g = self._build_sima_nodes(self._layer_base_name, quantizable)
        save_awesomenet(
            g, self.model_name + (".fp32" if quantizable else ""),
            str(self.sima_model_sdk_path)
        )

    def _build_sima_nodes(self, base_name: str, quantizable: bool):
        """Mirrors gen_onnx_files/_build_onnx_nodes on the SiMa Builder path.

        Tensors are (1, 1, num_tokens, C) here, so the experts lie on the last axis
        and TopK, which selects along that axis, needs no reshaping.
        """
        hidden = self.cfg.lm_cfg.hidden_size
        top_k = self.cfg.lm_cfg.moe_cfg.num_experts_per_tok
        dtype = activation_type(quantizable)
        hidden_shape = (1, 1, self.num_tokens, hidden)
        attn_shape = (
            1, 1, self.num_tokens, self.cfg.lm_cfg.attn_cfg.get_q_size(self.layer_type)
        )

        builder = SimaBuilder(
            Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target
        )
        model_input_input = builder.create_placeholder_node(
            "input", TensorType(dtype, hidden_shape)
        )
        model_input_self_attn = builder.create_placeholder_node(
            "self_attn", TensorType(dtype, attn_shape)
        )
        # MLA subgraph inputs are the same as the model inputs, with different names.
        builder.begin_subnet([model_input_input, model_input_self_attn])

        mla_input_input = builder.create_placeholder_node(
            "MLA_0/input", TensorType(dtype, hidden_shape)
        )
        mla_input_self_attn = builder.create_placeholder_node(
            "MLA_0/self_attn", TensorType(dtype, attn_shape)
        )

        # LFM2 uses self_attn.out_proj instead of o_proj.
        attn_out_name = (
            "out_proj"
            if self.check_hf_param(f"{base_name}.self_attn.out_proj.weight")
            else "o_proj"
        )
        o_proj = build_conv_from_dense_with_lora(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.self_attn.{attn_out_name}", mla_input_self_attn, None,
        )
        # h = input + o_proj(self_attn): residual the combine adds back.
        residual = builder.create_add_node(mla_input_input, o_proj)
        x = self._build_sima_rms_norm(
            builder, f"{base_name}.post_attention_layernorm", residual
        )

        # Gating linear -> per-expert logits. OLMoE uses mlp.gate; others mlp.router.
        router_weight_name = (
            f"{base_name}.mlp.gate"
            if self.check_hf_param(f"{base_name}.mlp.gate.weight")
            else f"{base_name}.mlp.router"
        )
        logits = build_conv_from_dense_with_lora(
            builder, self.get_hf_param, self.check_hf_param, router_weight_name, x, None,
        )

        # TopK cannot return values and indices from one node, so each branch builds
        # one node per return value over the same input.
        if self.cfg.lm_cfg.arch == LlmArchType.OLMOE:
            # OLMoE: softmax over all experts, then top-k (no renorm). TopK values
            # ARE the weights.
            probs = builder.create_softmax_node(logits, 3)
            router_values = builder.create_topk_node(probs, top_k, TopKRetType.VALUES)
            router_indices = builder.create_topk_node(probs, top_k, TopKRetType.INDICES)
        else:
            # gpt_oss: top-k first, then softmax over the k selected.
            selected = builder.create_topk_node(logits, top_k, TopKRetType.VALUES)
            router_indices = builder.create_topk_node(logits, top_k, TopKRetType.INDICES)
            router_values = builder.create_softmax_node(selected, 3)

        # Same order as the ONNX outputs: values, indices, residual h, norm(h).
        builder.create_tuple_node([router_values, router_indices, residual, x])

        mla_node = builder.finish_subnet("MLA_0")

        self._cast_bf16_outputs_to_fp32(builder, mla_node)

        return builder.finish(self.model_name)

    def get_mla_input_tessellate_params(self) -> dict[int, "TensorTessellateParameters"]:
        """Custom tessellate params for the model's inputs on the MLA."""
        return {}

    def get_mla_output_tessellate_params(self) -> dict[int, "TensorTessellateParameters"]:
        """Custom tessellate params for the model's outputs on the MLA."""
        return {}
