from dataclasses import dataclass
import numpy as np

from sima_lmm.model.base import BaseModel
from sima_lmm.model.model_graph import Node
from sima_lmm.config.vlm_config import LlmArchType, VlmArchType


@dataclass
class LanguagePartBaseModel(BaseModel):

    def _build_mlp(
        self, graph, base_name: str, input_nodes: list[Node], merged_lora: bool = False, with_residual_add: bool =  False
    ) -> Node:
        """Build the MLP using the model's projection, expert and LoRA configuration.

        Handles both LFM2-style weights (w1/w2/w3) and standard weights (gate_proj/up_proj/down_proj).
        """
        assert len(input_nodes) == (2 if with_residual_add else 1)
        projections = ("w1", "w3", "w2") if self.check_hf_param(f"{base_name}.w2.weight") else (
            "gate_proj", "up_proj", "down_proj"
        )
        expert_idx = getattr(self, "expert_idx", -1)
        ranks = {}
        if self.cfg.lm_cfg.lora_cfg is not None:
            bundled = {
                "gate_proj": "experts.gate_up_proj", "up_proj": "experts.gate_up_proj",
                "down_proj": "experts.down_proj",
            }
            for name in projections:
                rank = self.cfg.lm_cfg.get_lora_rank(base_name, name)
                if rank is None and expert_idx >= 0:
                    rank = self.cfg.lm_cfg.get_lora_rank(base_name, bundled[name])
                ranks[name] = rank
        return graph.mlp(
            base_name, input_nodes[0], self.cfg.lm_cfg.mlp_cfg.act,
            projections=projections, residual=input_nodes[1] if with_residual_add else None,
            lora_ranks=ranks, merged_lora=merged_lora, expert_idx=expert_idx,
            de_interleave=self.cfg.lm_cfg.arch == LlmArchType.GPT_OSS,
            swiglu_limit=self.cfg.lm_cfg.mlp_cfg.swiglu_limit,
        )

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
    expert_idx: int = -1

    def _build_post_transformer(self, graph, input_node) -> list[Node]:
        """Build SiMa nodes for the post-transformer projection (final norm + lm_head)."""
        # LFM2 uses embedding_norm instead of norm for the final normalization.
        base_prefix = self.hf_model.language_model_param_base_name
        final_norm_name = (
            "embedding_norm" if self.check_hf_param(f"{base_prefix}.embedding_norm.weight") else "norm"
        )
        final_norm_full_name = f"{base_prefix}.{final_norm_name}"
        if self.is_draft:
            final_norm_full_name = final_norm_name
        rms_norm = graph.rms_norm(final_norm_full_name, input_node)

        # Find the last layer's size based on the weight tensor shape.
        output_embed_name = self._get_output_embed_name()
        output_embed_param = graph.parameter(output_embed_name)
        output_embed_weight = (
            output_embed_param[1] if isinstance(output_embed_param, tuple) else output_embed_param
        )
        output_vocab_size = output_embed_weight.shape[0]
        assert (
            1
            < output_vocab_size
            <= (
                self.cfg.lm_cfg.draft_vocab_size
                if self.cfg.lm_cfg.draft_vocab_size > 0
                else self.cfg.lm_cfg.token_cfg.vocab_size
            )
        )

        lm_heads = []
        kwargs = {}
        kwargs["src_weight_name"] = output_embed_name
        for i in range(self.cfg.lm_cfg.lm_head_num_splits):
            split_begin = i * self.cfg.lm_cfg.lm_head_split_dim
            split_end = min(split_begin + self.cfg.lm_cfg.lm_head_split_dim, output_vocab_size)

            def param_process_func(x: np.ndarray) -> np.ndarray:
                return x[split_begin:split_end]

            kwargs["weight_process_func"] = param_process_func
            kwargs["scale_process_func"] = param_process_func
            kwargs["bias_process_func"] = param_process_func
            lm_head = graph.linear(f"lm_head.{i}", rms_norm, **kwargs)
            if self.final_softcapping is not None:
                assert self.cfg.lm_cfg.arch == LlmArchType.GEMMA and (
                    self.cfg.lm_cfg.model_type == "gemma2"
                    or self.cfg.model_type == VlmArchType.VLM_GEMMA4
                )
                lm_head = graph.softcap(lm_head, self.cfg.lm_cfg.final_logit_softcapping)
            lm_heads.append(lm_head)

        if self.is_draft:
            lm_heads.append(input_node)
            return lm_heads
        if self.cfg.lm_cfg.lm_head_num_splits == 1 and not self.cfg.pipeline_cfg.return_logits:
            argmax = graph.argmax(lm_heads[0])
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
