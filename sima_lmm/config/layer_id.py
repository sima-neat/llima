import dataclasses

@dataclasses.dataclass(frozen=True)
class LayerID:
    """
    Identifies a part of a model that may be treated as a
    single model for the purpose of compiling and running
    on the MLA.
    The set of valid layer ID values depends on the model
    architecture.

    Args:
        part: Type of layer.  Valid values are 'group_pre',
            'group_cache', 'group_sliding_cache', 'group_post', 'single_pre',
            'single_cache', 'single_sliding_cache', 'single_post', 'group_conv',
            'single_conv', 'conv_post_final', 'vision', 'group_per_layer',
            'single_per_layer', and for Mixture-of-Experts models 'group_router',
            'single_router', 'group_expert', 'single_expert',
            'group_weightedsum', 'single_weightedsum'.
        part_idx: Index of the layer.
        expert_idx: Index of the expert within the layer, for the MoE
            'group_expert'/'single_expert' parts.  -1 for all non-expert parts.
    """
    part: str
    part_idx: int
    expert_idx: int = -1
