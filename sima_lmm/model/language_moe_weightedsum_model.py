from dataclasses import dataclass

from afe.apis.defines import gen2_target
from afe.ir.defines import Status
from afe.ir.serializer import save_awesomenet
from afe.ir.tensor_type import TensorType

from sima_lmm.model.base import LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePostBaseModel
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import SimaBuilder, activation_type


@dataclass
class LanguageMoeWeightedSumModel(LanguagePostBaseModel):
    """MoE combine: out = residual + sum_e(expert_e). Each expert already scaled its
    output by its routing weight, so this is a plain add-reduce. On the last layer it
    also carries the final transformer (norm + lm_head). Group variant sums all experts.
    """

    def __post_init__(self):
        assert self.num_tokens >= 1
        assert 0 <= self.layer_idx < self.cfg.lm_cfg.num_hidden_layers
        assert self.cfg.lm_cfg.moe_cfg is not None

    @property
    def _is_final_layer(self) -> bool:
        return self.layer_idx == self.cfg.lm_cfg.num_hidden_layers - 1

    def gen_onnx_files(self):
        hidden = self.cfg.lm_cfg.hidden_size
        moe = self.cfg.lm_cfg.moe_cfg
        nt = self.num_tokens
        # Prefill (multi-token) runs all experts; decode (single token) only the top-k.
        n_experts = moe.num_experts_per_tok if nt == 1 else moe.num_experts

        self.create_onnx_builder()
        b = self._onnx_builder
        for e in range(n_experts):
            b.create_input_node(f"expert_{e}", (1, hidden, 1, nt))
        b.create_input_node("residual", (1, hidden, 1, nt))

        combined = self._build_onnx_nodes(b.input_nodes, n_experts)
        if self._is_final_layer:
            base_name = self.hf_model.language_model_param_base_name
            output_nodes = self._build_onnx_post_transformer(base_name, combined)
            self._create_final_layer_output_nodes(output_nodes)
        else:
            b.create_output_node(b.get_node_output_name(combined), (1, hidden, 1, nt))
        b.create_and_save_model()

        self._onnx_builder = None

    def _build_onnx_nodes(self, input_nodes: list[OnnxNode], num_experts: int) -> OnnxNode:
        b = self._onnx_builder
        residual = input_nodes[-1]
        acc = input_nodes[0]
        for e in range(1, num_experts):
            acc = b.build_op(f"weightedsum.add_{e}", [acc, input_nodes[e]], "Add")
        # Add the residual h once, at the end.
        return b.build_op("weightedsum.add_residual", [acc, residual], "Add")

    def gen_model_sdk_files_directly(
        self,
        layer_cfg: LayerConfiguration,
        log_level: int,
        quantizable: bool,
    ):
        g = self._build_sima_nodes(quantizable)
        save_awesomenet(
            g, self.model_name + (".fp32" if quantizable else ""),
            str(self.sima_model_sdk_path)
        )

    def _build_sima_nodes(self, quantizable: bool):
        # The SiMa path lays tensors out as (1, 1, num_tokens, hidden), unlike the
        # (1, hidden, 1, num_tokens) used by the ONNX path above.
        shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        moe = self.cfg.lm_cfg.moe_cfg
        # Prefill (multi-token) runs all experts; decode (single token) only the top-k.
        n_experts = moe.num_experts_per_tok if self.num_tokens == 1 else moe.num_experts
        dtype = activation_type(quantizable)

        builder = SimaBuilder(
            Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target
        )
        subnet_inputs = [
            builder.create_placeholder_node(f"expert_{e}", TensorType(dtype, shape))
            for e in range(n_experts)
        ]
        subnet_inputs.append(
            builder.create_placeholder_node("residual", TensorType(dtype, shape))
        )
        # MLA subgraph inputs mirror the model inputs, with different node names.
        builder.begin_subnet(subnet_inputs)

        mla_experts = [
            builder.create_placeholder_node(f"MLA_0/expert_{e}", TensorType(dtype, shape))
            for e in range(n_experts)
        ]
        mla_residual = builder.create_placeholder_node(
            "MLA_0/residual", TensorType(dtype, shape)
        )

        acc = mla_experts[0]
        for e in range(1, n_experts):
            acc = builder.create_add_node(acc, mla_experts[e])
        # Add the residual h once, at the end.
        combined = builder.create_add_node(acc, mla_residual)

        if self._is_final_layer:
            outputs = self._build_post_transformer(builder, combined, quantizable)
            if len(outputs) > 1:
                _ = builder.create_tuple_node(outputs)

        mla_node = builder.finish_subnet("MLA_0")
        self._cast_bf16_outputs_to_fp32(builder, mla_node)
        return builder.finish(self.model_name)

    def get_mla_input_tessellate_params(self) -> dict[int, "TensorTessellateParameters"]:
        """Custom tessellate params for the model's inputs on the MLA."""
        return {}

    def get_mla_output_tessellate_params(self) -> dict[int, "TensorTessellateParameters"]:
        """Custom tessellate params for the model's output on the MLA."""
        return {}
