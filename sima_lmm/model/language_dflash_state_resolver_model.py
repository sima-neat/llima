from dataclasses import dataclass

from afe.apis.defines import TensorDRAMLayout

from sima_lmm.model.base import BaseModel, LayerConfiguration, TensorTessellateParameters
from sima_lmm.model.model_graph import ModelGraph


@dataclass
class LanguageDFlashStateResolverModel(BaseModel):
    """Resolve any accepted Gated DeltaNet prefix from grouped intermediates."""

    block_size: int

    def __post_init__(self):
        assert self.cfg.lm_cfg.linear_attn_cfg is not None
        assert self.block_size in (4, 8, 16)

    @property
    def enable_filter_sharing(self) -> bool:
        return self.cfg.pipeline_cfg.enable_filter_sharing

    def generate_graph(self, layer_cfg: LayerConfiguration, quantizable: bool):
        linear_cfg = self.cfg.lm_cfg.linear_attn_cfg
        heads = linear_cfg.num_value_heads
        key_dim = linear_cfg.key_head_dim
        value_dim = linear_cfg.value_head_dim
        graph = ModelGraph(self, {
            "linear_delta_state_s1": (1, heads, key_dim, value_dim),
            "key": (1, heads, self.block_size, key_dim),
            "v_new": (1, heads, self.block_size, value_dim),
            "decay_row": (1, heads, 1, self.block_size),
        }, quantizable)
        state, key, value, decay_row = graph.inputs.values()
        base_decay = graph.slice(decay_row, start=0, stop=1, axis=3)
        state_base = graph.mul(state, base_decay)
        update_tail = graph.slice(decay_row, start=1, stop=self.block_size, axis=3)
        zero = graph.sub(base_decay, base_decay)
        update_weights = graph.concat([zero, update_tail], 3)
        update_weights = graph.transpose(update_weights, [0, 1, 3, 2])
        weighted_value = graph.mul(value, update_weights)
        state_update = graph.matmul(key, weighted_value, transpose_a=True)
        graph.save([graph.add(state_base, state_update)])

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        linear_cfg = self.cfg.lm_cfg.linear_attn_cfg
        return {
            3: TensorTessellateParameters(
                tile_shape=(0, 0, 0, 0),
                enable_mla=True,
                dram_layout=TensorDRAMLayout.HWC16,
                dram_shape=(
                    1,
                    linear_cfg.num_value_heads,
                    self.block_size,
                    self.block_size,
                ),
            )
        }

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        return {}
