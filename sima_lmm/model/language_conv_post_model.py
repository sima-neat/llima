from dataclasses import dataclass

from sima_lmm.model.base import LayerConfiguration, LoraGenMode
from sima_lmm.model.language_part_base import LanguagePostBaseModel
from sima_lmm.model.model_graph import ModelGraph


@dataclass
class LanguageConvPostModel(LanguagePostBaseModel):
    """Post-convolution model for the final layer of LFM2.
    This reusable model is compiled for num_tokens=1 and contains the FFN and lm_head.
    """

    def __post_init__(self):
        assert self.num_tokens == 1, "LanguageConvPostModel only supports num_tokens=1"

    @property
    def enable_filter_sharing(self) -> bool:
        return self.cfg.pipeline_cfg.enable_filter_sharing

    def generate_graph(
        self, layer_cfg: LayerConfiguration, quantizable: bool
    ):
        base_name = f"{self.hf_model.language_model_param_base_name}.layers.{self.layer_idx}"
        merged_lora = layer_cfg.get("lora", LoraGenMode.LORA_DISABLED) == LoraGenMode.LORA_MERGED
        input_shape = (1, 1, 1, self.cfg.lm_cfg.hidden_size)
        graph = ModelGraph(self, {"input": input_shape}, quantizable)
        mla_input_input = graph.inputs["input"]

        rms_norm2 = graph.rms_norm(f"{base_name}.ffn_norm", mla_input_input)

        mlp_base = (
            f"{base_name}.feed_forward"
            if self.check_hf_param(f"{base_name}.feed_forward.w2.weight")
            else f"{base_name}.mlp"
        )
        mlp = self._build_mlp(graph, mlp_base, [rms_norm2], merged_lora=merged_lora)
        add2 = graph.add(mla_input_input, mlp)

        outputs = self._build_post_transformer(graph, add2)

        graph.save(outputs)
