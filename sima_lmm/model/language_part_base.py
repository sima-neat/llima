from dataclasses import dataclass
import numpy as np

from afe.backends.backends import Backend
from afe.ir.build_node import NodeOrHandle

from sima_lmm.gguf.gguf_conversion import GgufModel
from sima_lmm.model.base import BaseModel
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.sima_builder import SimaBuilder, build_conv, build_logit_softcapping
from sima_lmm.config.vlm_config import LlmArchType, VlmArchType
from sima_lmm.utils import ceil_div


@dataclass
class LanguagePartBaseModel(BaseModel):
    def _build_rms_norm(self, base_name: str, input_node: OnnxNode) -> OnnxNode:
        weight_offset = 1.0 if self.cfg.lm_cfg.rms_norm_unit_offset else 0.0
        return self._onnx_builder.build_rms_norm(
            base_name, input_node, self.cfg.lm_cfg.rms_norm_eps, weight_offset
        )

    def _build_sima_rms_norm(
        self,
        builder: SimaBuilder,
        base_name: str,
        input_node: NodeOrHandle,
        weightless: bool = False,
        num_channels: int | None = None,
    ) -> NodeOrHandle:
        """
        Create an RMS norm with a multiplication applied to its outputs.
        """
        if weightless:
            assert num_channels is not None
            weight_tensor = np.ones(num_channels, dtype=np.float32)
        else:
            weight_offset = 1.0 if self.cfg.lm_cfg.rms_norm_unit_offset else 0.0
            weight_tensor = self.get_hf_param(f"{base_name}.weight") + weight_offset

        # Reduce the precision of epsilon so that it is exactly representable in float32
        epsilon = float(np.float32(self.cfg.lm_cfg.rms_norm_eps))
        return builder.create_rms_norm_node(input_node, epsilon, weight_tensor)

    def _build_onnx_mlp(
        self, base_name: str, input_nodes: list[OnnxNode], with_residual_add: bool = False
    ) -> OnnxNode:
        """Build ONNX nodes for the MLP block with optional splitting.

        Handles both LFM2-style weights (w1/w2/w3) and standard weights (gate_proj/up_proj/down_proj).
       """
        # Make sure that there is residual add input if needed.
        assert len(input_nodes) == (2 if with_residual_add else 1)

        # Determine weight naming convention based on what exists in the model.
        if self.check_hf_param(f"{base_name}.w2.weight"):
            gate_name, up_name, down_name = "w1", "w3", "w2"
        else:
            gate_name, up_name, down_name = "gate_proj", "up_proj", "down_proj"

        max_ch = self.cfg.lm_cfg.hidden_size
        if (
            self.split_mlp
            and (self.num_tokens != 1 or self.enable_filter_sharing)
            and (self.layer_idx < self.cfg.lm_cfg.num_hidden_layers - 1 or self.cfg.lm_cfg.speculative_decoding_cfg is not None)
        ):
            # Split MLP into multiple parts if intermediate_size is larger than max ch number
            # in order to prevent Large Tensor Helper activation in n2a compiler.
            # For single token models do splitting only if filter sharing is enabled.
            intermediate_size = self.cfg.lm_cfg.get_effective_intermediate_size(self.layer_idx)
            num_parts = ceil_div(intermediate_size, max_ch)
        else:
            intermediate_size = self.cfg.lm_cfg.get_effective_intermediate_size(self.layer_idx)
            num_parts = 1

        for part in range(num_parts):
            part_idx = f".{part}" if num_parts > 1 else ""
            offset = part * max_ch
            size = min(max_ch, (intermediate_size - offset))
            weight_slice_by_input_channels = (offset, size, 0, part) if num_parts > 1 else None
            weight_slice_by_output_ch = (offset, size, 1, part) if num_parts > 1 else None

            lora_rank = None
            if self.cfg.lm_cfg.lora_cfg is not None:
                lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, gate_name)
            gate_proj = self._onnx_builder.build_conv_from_dense_with_lora(
                f"{base_name}.{gate_name}", input_nodes[0], lora_rank,
                weight_slice=weight_slice_by_input_channels
            )

            act = self._onnx_builder.build_activation(
                f"{base_name}.act{part_idx}", gate_proj, self.cfg.lm_cfg.mlp_cfg.act
            )

            lora_rank = None
            if self.cfg.lm_cfg.lora_cfg is not None:
                lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, up_name)
            up_proj = self._onnx_builder.build_conv_from_dense_with_lora(
                f"{base_name}.{up_name}", input_nodes[0], lora_rank,
                weight_slice=weight_slice_by_input_channels
            )

            mul2 = self._onnx_builder.build_op(f"{base_name}.mul2{part_idx}", [act, up_proj], "Mul")

            lora_rank = None
            if self.cfg.lm_cfg.lora_cfg is not None:
                lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, down_name)

            down_proj = self._onnx_builder.build_conv_from_dense_with_lora(
                f"{base_name}.{down_name}", mul2, lora_rank, weight_slice=weight_slice_by_output_ch
            )

            if with_residual_add and part == 0:
                # Sums the MLP output with the residual stream.
                down_proj = self._onnx_builder.build_op(
                    f"{base_name}.add2{part_idx}", [input_nodes[1], down_proj], "Add"
                )
            elif part > 0:
                # Sums the MLP output with the output of previous MLP part output.
                down_proj = self._onnx_builder.build_op(
                    f"{base_name}.add_part{part_idx}", [prev_part_down_proj, down_proj], "Add"
                )
            prev_part_down_proj = down_proj

        return down_proj

    def _build_sima_mlp(
        self, builder, base_name: str, input_nodes: list[NodeOrHandle], quantizable: bool,
        merged_lora: bool = False, with_residual_add: bool =  False
    ) -> NodeOrHandle:
        """Build SiMa nodes for the MLP block with optional splitting.

        Handles both LFM2-style weights (w1/w2/w3) and standard weights (gate_proj/up_proj/down_proj).
        """
        from sima_lmm.model.sima_builder import build_conv_from_dense_with_lora, build_activation

        # Make sure that there is residual add input if needed.
        assert len(input_nodes) == (2 if with_residual_add else 1)

        # Determine weight naming convention based on what exists in the model.
        if self.check_hf_param(f"{base_name}.w2.weight"):
            gate_name, up_name, down_name = "w1", "w3", "w2"
        else:
            gate_name, up_name, down_name = "gate_proj", "up_proj", "down_proj"

        max_ch = self.cfg.lm_cfg.hidden_size
        if self.split_mlp and (self.num_tokens != 1 or self.enable_filter_sharing):
            # Split MLP into multiple parts if intermediate_size is larger than max ch number
            # in order to prevent Large Tensor Helper activation in n2a compiler.
            # For single token models do splitting only if filter sharing is enabled.
            intermediate_size = self.cfg.lm_cfg.get_effective_intermediate_size(self.layer_idx)
            num_parts = ceil_div(intermediate_size, max_ch)
        else:
            intermediate_size = self.cfg.lm_cfg.get_effective_intermediate_size(self.layer_idx)
            num_parts = 1

        for part in range(num_parts):
            offset = part * max_ch
            size = min(max_ch, (intermediate_size - offset ))
            weight_slice_by_input_channels = (offset, size, -1, part) if num_parts > 1 else None
            weight_slice_by_output_ch = (offset, size, -3, part) if num_parts > 1 else None

            lora_rank = None
            if self.cfg.lm_cfg.lora_cfg is not None:
                lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, gate_name)
            gate_proj = build_conv_from_dense_with_lora(
                builder, self.get_hf_param, self.check_hf_param, f"{base_name}.{gate_name}",
                input_nodes[0], lora_rank, merged_lora=merged_lora,
                weight_slice=weight_slice_by_input_channels
            )

            act = build_activation(builder, gate_proj, self.cfg.lm_cfg.mlp_cfg.act, quantizable)

            lora_rank = None
            if self.cfg.lm_cfg.lora_cfg is not None:
                lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, up_name)
            up_proj = build_conv_from_dense_with_lora(
                builder, self.get_hf_param, self.check_hf_param, f"{base_name}.{up_name}",
                input_nodes[0], lora_rank, merged_lora=merged_lora,
                weight_slice=weight_slice_by_input_channels
            )

            mul2 = builder.create_mul_node(act, up_proj)

            lora_rank = None
            if self.cfg.lm_cfg.lora_cfg is not None:
                lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, down_name)
            down_proj = build_conv_from_dense_with_lora(
                builder, self.get_hf_param, self.check_hf_param, f"{base_name}.{down_name}", mul2,
                lora_rank, merged_lora=merged_lora, weight_slice=weight_slice_by_output_ch
            )

            if with_residual_add and part == 0:
                # Sums the MLP output with the residual stream.
                down_proj = builder.create_add_node(input_nodes[1], down_proj)

            elif part > 0:
                # Sums the MLP output with the output of previous MLP part output.
                down_proj = builder.create_add_node(prev_part_down_proj, down_proj)
            prev_part_down_proj = down_proj

        return down_proj

    def _cast_bf16_outputs_to_fp32(self, builder: SimaBuilder, mla_node: NodeOrHandle):
        """Cast bfloat16 outputs to float32. Do not cast int outputs."""
        from afe.ir.defines import TensorValue, TupleValue, get_expected_tensor_value
        from afe.ir.tensor_type import ScalarType

        match mla_node.get_type().output:
            case TensorValue(value=t):
                if t.scalar == ScalarType.bfloat16:
                    _ = builder.create_cast_node(mla_node, ScalarType.float32, backend=Backend.EV)
            case TupleValue():
                tuple_items = []
                for node in builder.create_tuple_get_item_nodes(mla_node):
                    if (get_expected_tensor_value(node.get_type().output).scalar
                            == ScalarType.bfloat16):
                        tuple_items.append(
                            builder.create_cast_node(node, ScalarType.float32, backend=Backend.EV)
                        )
                    else:
                        tuple_items.append(node)
                builder.create_tuple_node(tuple_items)

    @property
    def is_draft(self) -> bool:
        cfg = self.cfg.lm_cfg.speculative_decoding_cfg
        return cfg is not None and cfg.is_draft

    @property
    def uses_quantized_input_embeddings(self) -> bool:
        return self.cfg.pipeline_cfg.quantize_embeddings


@dataclass
class LanguagePostBaseModel(LanguagePartBaseModel):
    """Abstract base class for post-cache language model implementations.

    This provides shared functionality for both transformer-based post-cache models
    (LanguagePostModel) and convolution-based post-cache models (LanguageConvPostModel).

    Attributes:
        num_tokens: Number of tokens. Set to a value greater than 1 to consume multiple input tokens
            in one model.
        layer_idx: Transformer layer index.
        final_softcapping: Final logit soft capping for gemma 2.
    """
    num_tokens: int
    layer_idx: int
    final_softcapping: float | None

    def _create_final_layer_output_nodes(self, output_nodes: list[OnnxNode]):
        """Create output nodes for the final transformer layer."""
        if not self.is_draft and self.cfg.lm_cfg.lm_head_num_splits == 1 and not self.cfg.pipeline_cfg.return_logits:
            output_name = self._onnx_builder.get_node_output_name(output_nodes[0])
            self._onnx_builder.create_output_node(output_name, (1, 1, 1, self.num_tokens), np.int64)
        else:
            # Find the last layer's size based on the weight tensor shape.
            output_vocab_size = self.get_hf_param(self._get_output_embed_name()).shape[0]
            assert 1 < output_vocab_size <= (
                self.cfg.lm_cfg.draft_vocab_size if self.cfg.lm_cfg.draft_vocab_size > 0
                else self.cfg.lm_cfg.token_cfg.vocab_size
            )

            for i in range(self.cfg.lm_cfg.lm_head_num_splits):
                split_begin = i * self.cfg.lm_cfg.lm_head_split_dim
                split_size = min(
                    output_vocab_size - split_begin,
                    self.cfg.lm_cfg.lm_head_split_dim
                )
                output_name = self._onnx_builder.get_node_output_name(output_nodes[i])
                self._onnx_builder.create_output_node(
                    output_name, (1, split_size, 1, self.num_tokens)
                )
            if self.is_draft:
                # EAGLE3 draft model also returns hidden_states as the last output
                hidden_states_name = self._onnx_builder.get_node_output_name(output_nodes[-1])
                self._onnx_builder.create_output_node(
                    hidden_states_name, (1, self.cfg.lm_cfg.hidden_size, 1, self.num_tokens)
                )

    def _build_onnx_post_transformer(self, base_name: str, input_node: OnnxNode) -> list[OnnxNode]:
        """
        Build ONNX nodes for the post-transformer projection (final norm + lm_head).
        """
        # LFM2 uses embedding_norm instead of norm for the final normalization.
        base_prefix = self.hf_model.language_model_param_base_name
        final_norm_name = (
            "embedding_norm" if self.check_hf_param(f"{base_prefix}.embedding_norm.weight") else "norm"
        )
        final_norm_full_name = f"{base_prefix}.{final_norm_name}"
        if self.is_draft:
            final_norm_full_name = final_norm_name
        rms_norm2 = self._build_rms_norm(final_norm_full_name, input_node)

        # Find the last layer's size based on the weight tensor shape.
        output_embed_name = self._get_output_embed_name()
        output_vocab_size = self.get_hf_param(output_embed_name).shape[0]
        assert 1 < output_vocab_size <= (
            self.cfg.lm_cfg.draft_vocab_size if self.cfg.lm_cfg.draft_vocab_size > 0
            else self.cfg.lm_cfg.token_cfg.vocab_size
        )

        lm_heads = list()
        kwargs = dict()
        kwargs["src_weight_name"] = output_embed_name
        for i in range(self.cfg.lm_cfg.lm_head_num_splits):
            split_begin = i * self.cfg.lm_cfg.lm_head_split_dim
            split_end = min(
                split_begin + self.cfg.lm_cfg.lm_head_split_dim,
                output_vocab_size
            )
            def param_process_func(x: np.ndarray) -> np.ndarray:
                return x[split_begin:split_end]
            kwargs["weight_process_func"] = param_process_func
            kwargs["bias_process_func"] = param_process_func
            lm_head = self._onnx_builder.build_conv(f"lm_head.{i}", rms_norm2, **kwargs)
            if self.final_softcapping is not None:
                assert self.cfg.lm_cfg.arch == LlmArchType.GEMMA and (
                    self.cfg.model_type in (VlmArchType.LLM_GEMMA2, VlmArchType.VLM_GEMMA4)
                )
                lm_head = self._onnx_builder.build_logit_softcapping(
                    f"{base_name}.final_softcap.{i}", lm_head, self.cfg.lm_cfg.final_logit_softcapping
                )
            lm_heads.append(lm_head)

        if self.is_draft:
            lm_heads.append(input_node)
            return lm_heads
        if self.cfg.lm_cfg.lm_head_num_splits == 1 and not self.cfg.pipeline_cfg.return_logits:
            argmax = self._onnx_builder.build_op(
                "argmax", lm_heads, "ArgMax", axis=1, keepdims=1
            )
            return [argmax]
        else:
            return lm_heads

    def _build_post_transformer(self, builder, input_node, quantizable) -> NodeOrHandle:
        """Build SiMa nodes for the post-transformer projection (final norm + lm_head)."""
        from afe.ir.tensor_type import ScalarType

        # LFM2 uses embedding_norm instead of norm for the final normalization.
        base_prefix = self.hf_model.language_model_param_base_name
        final_norm_name = "embedding_norm" if self.check_hf_param(f"{base_prefix}.embedding_norm.weight") else "norm"
        final_norm_full_name = f"{base_prefix}.{final_norm_name}"
        if self.is_draft:
            final_norm_full_name = final_norm_name
        rms_norm = self._build_sima_rms_norm(builder,
            final_norm_full_name, input_node
        )

        # Find the last layer's size based on the weight tensor shape.
        output_embed_name = self._get_output_embed_name()
        output_embed_param = self.get_hf_param(output_embed_name)
        output_embed_weight = (
            output_embed_param[1] if isinstance(output_embed_param, tuple)
            else output_embed_param
        )
        output_vocab_size = output_embed_weight.shape[0]
        assert 1 < output_vocab_size <= (
            self.cfg.lm_cfg.draft_vocab_size if self.cfg.lm_cfg.draft_vocab_size > 0
            else self.cfg.lm_cfg.token_cfg.vocab_size
        )

        lm_heads = []
        kwargs = {}
        kwargs["src_weight_name"] = output_embed_name
        for i in range(self.cfg.lm_cfg.lm_head_num_splits):
            split_begin = i * self.cfg.lm_cfg.lm_head_split_dim
            split_end = min(
                split_begin + self.cfg.lm_cfg.lm_head_split_dim,
                output_vocab_size
            )
            def param_process_func(x: np.ndarray) -> np.ndarray:
                return x[split_begin:split_end]
            kwargs["weight_process_func"] = param_process_func
            kwargs["bias_process_func"] = param_process_func
            lm_head = build_conv(builder, self.get_hf_param, self.check_hf_param,
                                 f"lm_head.{i}", rms_norm, **kwargs)
            if self.final_softcapping is not None:
                assert self.cfg.lm_cfg.arch == LlmArchType.GEMMA and (
                    self.cfg.lm_cfg.model_type == "gemma2"
                    or self.cfg.model_type == VlmArchType.VLM_GEMMA4
                )
                lm_head = build_logit_softcapping(
                    builder, lm_head, self.cfg.lm_cfg.final_logit_softcapping, quantizable
                )
            lm_heads.append(lm_head)

        if self.is_draft:
            lm_heads.append(input_node)
            return lm_heads
        if self.cfg.lm_cfg.lm_head_num_splits == 1 and not self.cfg.pipeline_cfg.return_logits:
            argmax = builder.create_argmax_node(lm_heads[0], ScalarType.int32)
            return [argmax]
        else:
            return lm_heads

    def _get_output_embed_name(self):
        """
        Get the name of the model's output embedding weight tensor.
        """
        if self.cfg.vm_cfg:
            # For VLMs, check for prefixed names first, which is the common case.
            ordered_candidates = [
                "lm_head.weight",
                "language_model.lm_head.weight",
                "language_model.model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight",
                "model.embed_tokens.weight",
            ]
        else:
            # For LLM-only models, standard names are expected.
            ordered_candidates = [
                "lm_head.weight",
                "model.embed_tokens.weight",
                "model.lm_head.weight",
            ]

        for name in ordered_candidates:
            if self.check_hf_param(name):
                return name

        raise RuntimeError(
            f"Cannot determine the tensor name for the output embedding, tried {ordered_candidates}"
        )
