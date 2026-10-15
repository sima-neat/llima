from dataclasses import dataclass

from afe.ir.defines import get_expected_tensor_value

from sima_lmm.model.base import LayerConfiguration
from sima_lmm.model.language_part_base import LanguagePartBaseModel
from sima_lmm.model.model_graph import ModelGraph

@dataclass
class LanguageDraftFCModel(LanguagePartBaseModel):
    """FC Fusion layer for the EAGLE3 draft model.

    Projects concatenated hidden states of shape (1, hidden_size * 3, 1, num_tokens)
    to (1, hidden_size, 1, num_tokens) using a single linear layer

    EAGLE3 conditions the draft model on hidden states from three specific layers of
    the target model (low, mid, and high), which are concatenated along the channel
    dimension to form a tensor of shape (1, hidden_size * 3, 1, num_tokens).
    """
    num_tokens: int

    def __post_init__(self):
        assert self.num_tokens >= 1

    def generate_graph(
        self,
        layer_cfg: LayerConfiguration,
        quantizable: bool,
    ):
        output_shape = (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size)
        if self.is_dflash:
            input_count = len(self.cfg.lm_cfg.speculative_decoding_cfg.target_layer_ids)
            input_specs = {f"input_{index}": output_shape for index in range(input_count)}
        else:
            input_specs = {"input": (1, 1, self.num_tokens, self.cfg.lm_cfg.hidden_size * 3)}
        graph = ModelGraph(self, input_specs, quantizable)
        fc_input = graph.concat(list(graph.inputs.values()), 3) if self.is_dflash else graph.inputs["input"]
        output = graph.linear("fc", fc_input)
        if self.is_dflash:
            output = graph.rms_norm("hidden_norm", output)
        assert get_expected_tensor_value(output.get_type().output).shape == output_shape

        graph.save([output])
