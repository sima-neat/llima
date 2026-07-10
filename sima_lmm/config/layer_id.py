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
            'single_conv', 'group_linear', 'single_linear', 'conv_post_final', 'vision',
            'group_per_layer', 'single_per_layer'.
        part_idx: Index of the layer.
    """
    part: str
    part_idx: int
