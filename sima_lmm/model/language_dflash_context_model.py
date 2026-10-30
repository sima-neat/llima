from dataclasses import dataclass

from sima_lmm.model.base import LayerConfiguration, LoraGenMode
from sima_lmm.model.language_pre_model import LanguagePreModel
from sima_lmm.model.model_graph import ModelGraph


@dataclass
class LanguageDFlashContextModel(LanguagePreModel):
    """Project fused target hidden states into one DFlash layer's K/V cache."""

    def generate_graph(self, layer_cfg: LayerConfiguration, quantizable: bool):
        merged_lora = layer_cfg.get("lora", LoraGenMode.LORA_DISABLED) == LoraGenMode.LORA_MERGED
        input_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        freq_shape = (
            1, 1, self.num_tokens,
            self.cfg.lm_cfg.rope_cfg.get_rope_dimension_count(self.layer_type) // 2,
        )
        graph = ModelGraph(self, {
            "input": input_shape, "freq_real": freq_shape, "freq_imag": freq_shape,
        }, quantizable)
        attention_name = f"layers.{self.layer_idx}.self_attn"
        key = self._build_attn_key(
            graph, attention_name, graph.inputs["input"],
            graph.inputs["freq_real"], graph.inputs["freq_imag"], merged_lora,
        )
        value = self._build_attn_value(graph, attention_name, graph.inputs["input"], merged_lora)
        if self.cfg.pipeline_cfg.quantize_kv_cache:
            key, key_scale = graph.quant(key)
            value, value_scale = graph.quant(value)
            outputs = [key, key_scale, value, value_scale]
        else:
            outputs = [key, value]
        graph.save(outputs)

    def get_mla_output_tessellate_params(self):
        return {
            index - 1: params
            for index, params in super().get_mla_output_tessellate_params().items()
        }
