#########################################################
# Copyright (C) 2025 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################
"""
Functions for reading information from a SiMa IR model.
"""
import dataclasses

from afe.apis.defines import TensorDRAMLayout, TensorTessellateParameters
from afe.apis.model import Model as SDKModel
from afe.ir.node import node_is_tuple


def get_tessellate_parameters(
    model: SDKModel,
    input_params: dict[int, TensorTessellateParameters],
    output_params: dict[int, TensorTessellateParameters]
) -> TensorTessellateParameters:
    """
    Determine tessellation parameters for an SDK model that has a single MLA partition.
    These can be used to compile the MLA partition in isolation.

    Args:
        model: SDK model to examine
        input_params: DRAM layouts to use for the MLA partition's inputs.  The list
            must have one item for each of the partition's input tensors.
        output_params: DRAM layouts to use for the MLA partition's inputs.  The list
            must have one item for each of the partition's input tensors.
    Returns:
        Tessellation parameters that can be used to compile the MLA segment
    """

    def _get_tessellate_params(tess_params, idx, num_inputs):
        default_tessellate_params: TensorTessellateParameters = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC16,
            persistent_mem_name=None,
            dram_shape=None
        )
        return tess_params.get(idx, tess_params.get(idx - num_inputs, default_tessellate_params))

    assert isinstance(input_params, dict)
    assert isinstance(output_params, dict)

    # Get the MLA node
    assert "MLA_0" in model._net.nodes
    mla_node = model._net.nodes["MLA_0"]
    tessellate_parameters = dict()

    # Set input tessellate parameters
    for input_idx, input_name in enumerate(mla_node.input_names):
        tessellate_parameters[input_name] = dataclasses.replace(
            _get_tessellate_params(input_params, input_idx, len(mla_node.input_names)),
            persistent_mem_name=f"input_{input_idx}/{input_name}"
        )

    # Set output tessellate parameters
    output_node = mla_node.ir.nodes[mla_node.ir.output_node_name]
    out_names = output_node.input_node_names if node_is_tuple(output_node) else [output_node.name]
    for output_idx, output_name in enumerate(out_names):
        tessellate_parameters[f"{output_name}_output"] = dataclasses.replace(
            _get_tessellate_params(output_params, output_idx, len(out_names)),
            persistent_mem_name=f"output_{output_idx}/{output_name}"
        )

    return tessellate_parameters
