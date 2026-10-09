from dataclasses import dataclass

import numpy as np
from sima_lmm.model.base import LoraGenMode, LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.model.model_graph import ModelGraph


@dataclass
class LanguageConvModel(LanguagePartBaseModel):
    """Fused conv layer (pre + cache + post) for LFM2.

    Inputs:
        - input: residual input (1, hidden, 1, num_tokens)
        - conv_cache: rolling window buffer (1, hidden, 1, L-1)

    Outputs:
        - hidden (1, hidden, 1, num_tokens) for non-last layer
        - if last layer, same as LanguagePostModel (argmax or split logits)
        - conv_cache_out:
            * grouped prefill (num_tokens > 1): full concat state
              (1, hidden, 1, num_tokens + L - 1), where concat = [conv_cache, bx]
            * decode (num_tokens == 1): rolling window state
              (1, hidden, 1, L - 1), i.e. concat[..., 1:]
    """

    num_tokens: int
    layer_idx: int
    final_softcapping: float | None

    def __post_init__(self):
        assert self.num_tokens >= 1
        assert 0 <= self.layer_idx < self.cfg.lm_cfg.num_hidden_layers
        assert (
            not self.cfg.lm_cfg.conv_bias
        ), "LanguageConvModel requires conv_bias=False due to missing padding mask logic."

    @property
    def enable_filter_sharing(self) -> bool:
        return self.cfg.pipeline_cfg.enable_filter_sharing

    def generate_graph(
        self,
        layer_cfg: LayerConfiguration,
        quantizable: bool,
    ):
        base_layer = f"{self.hf_model.language_model_param_base_name}.layers.{self.layer_idx}"
        base_name = f"{base_layer}.conv"
        merged_lora = layer_cfg.get("lora", LoraGenMode.LORA_DISABLED) == LoraGenMode.LORA_MERGED
        hidden_size = self.cfg.lm_cfg.hidden_size

        input_shape = (1, 1, self.num_tokens, hidden_size)
        scale_shape = (1, 1, self.num_tokens, 1)
        cache_shape = (1, 1, self.cfg.lm_cfg.conv_L_cache - 1, hidden_size)

        input_specs = {"input": input_shape}
        input_dtypes = {}
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            input_dtypes["input"] = np.int8
            input_specs["input_scale"] = scale_shape
        input_specs["conv_cache"] = cache_shape
        graph = ModelGraph(self, input_specs, quantizable, input_dtypes=input_dtypes)
        mla_input_input = graph.inputs["input"]
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            mla_input_scale = graph.inputs["input_scale"]
        mla_input_conv_cache = graph.inputs["conv_cache"]
        if self.uses_quantized_input_embeddings and self.layer_idx == 0:
            residual = graph.dequant(mla_input_input, mla_input_scale)
        else:
            residual = mla_input_input

        norm_input = graph.rms_norm(f"{base_layer}.operator_norm", residual)
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "in_proj")
        in_proj = graph.linear(
            f"{base_name}.in_proj", norm_input, lora_rank=lora_rank, merged_lora=merged_lora
        )

        b = graph.slice(in_proj, start=0, stop=hidden_size, axis=3)
        c = graph.slice(in_proj, start=hidden_size, stop=2 * hidden_size, axis=3)
        x = graph.slice(in_proj, start=2 * hidden_size, stop=3 * hidden_size, axis=3)
        bx = graph.mul(b, x)

        tail = graph.concat([mla_input_conv_cache, bx], 2)
        conv_cache_out = graph.slice(
            tail, start=1, stop=self.num_tokens + self.cfg.lm_cfg.conv_L_cache - 1, axis=2
        )

        conv_out = graph.conv(f"{base_name}.conv", tail, is_depthwise=True)

        gated = graph.mul(conv_out, c)
        lora_rank = None
        if self.cfg.lm_cfg.lora_cfg is not None:
            lora_rank = self.cfg.lm_cfg.get_lora_rank(base_name, "out_proj")
        out_proj = graph.linear(
            f"{base_name}.out_proj", gated, lora_rank=lora_rank, merged_lora=merged_lora
        )

        add1 = graph.add(residual, out_proj)

        if self.layer_idx == self.cfg.lm_cfg.num_hidden_layers - 1:
            outputs = [add1, conv_cache_out]
        else:
            rms_norm2 = graph.rms_norm(f"{base_layer}.ffn_norm", add1)
            mlp = self._build_mlp(
                graph, f"{base_layer}.feed_forward", [rms_norm2], merged_lora
            )
            add2 = graph.add(add1, mlp)
            outputs = [add2, conv_cache_out]

        graph.save(outputs)
