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

import math
import numpy as np
from typing import Callable, Sequence

from sima_utils.common import Platform

from afe.backends.backends import Backend
import afe.ir.attributes as attributes
from afe.ir.attributes import ConvAttrs
from afe.ir.defines import NodeName, Status, get_expected_tensor_value
from afe.ir.node import AwesomeNode
from afe.ir.net import AwesomeNet
from afe.ir.tensor_type import ScalarType, TensorType
import afe.ir.build_node as build_node
from afe.ir.build_node import NodeHandle, NodeOrHandle

from sima_lmm.model.onnx_builder import find_alternate_weight
from sima_lmm.utils import (
    ceil_div_row, mla_max_num_rows, mla_row_size, round_up_to_row
)


bfloat16 = ScalarType.numpy_type(ScalarType.bfloat16)

class _AwesomeNetBuilderData:
    """
    Data that will be used to create one AwesomeNet.
    """
    # Nodes of the parent net that will be passed as inputs.  Ignored for the top-level net.
    arguments: list[NodeName]
    # The net's placeholder nodes.
    placeholders: list[NodeName]
    # All nodes in the net.  Includes placeholder nodes.
    nodes: list[AwesomeNode]

    def __init__(self):
        self.arguments = []
        self.placeholders = []
        self.nodes = []


class SimaBuilder:
    """
    Builder of a SiMa IR model.

    During construction, the builder is associated with a current net,
    which is where new nodes are inserted.  Call begin_subnet to begin
    creating a sub-network and set it as the current net.  Call
    finish_subnet to finish constructing the sub-network, insert it into
    the parent network, and set the parent as the current net.

    The "create" methods create a new node and insert it into the current net.

    When all nodes are created, call finish to finish building and return the
    model.  A builder will be unusable after finish is called.
    """
    net_stack: list[_AwesomeNetBuilderData]
    name_counter: int
    status: Status
    target: Platform

    def __init__(self, status: Status, target: Platform):
        self.net_stack = [_AwesomeNetBuilderData()]
        self.name_counter = 0
        self.status = status
        self.target = target

    def _new_name(self) -> int:
        n = self.name_counter
        self.name_counter = n + 1
        return n

    def _record(self, n: AwesomeNode) -> AwesomeNode:
        assert len(self.net_stack)
        assert isinstance(n, AwesomeNode)
        self.net_stack[-1].nodes.append(n)
        return n

    def _finish_net(self, name: str, is_subgraph: bool, backend: Backend) -> AwesomeNet:
        b = self.net_stack.pop()
        assert len(b.nodes)
        return AwesomeNet(name, {n.name: n for n in b.nodes}, b.placeholders, b.nodes[-1].name,
                          self.status, _is_subgraph=is_subgraph, _backend=backend,
                          _target=self.target)

    def finish(self, name: str) -> AwesomeNet:
        """
        Return an AwesomeNet containing the created nodes.

        Args:
            name: Name of the AwesomeNet
        """
        assert len(self.net_stack) == 1
        return self._finish_net(name, False, Backend.NONE)
    
    def begin_subnet(self, input_nodes: list[NodeOrHandle]):
        """
        Start creating a sub-network.  Nodes that are bracketed between this begin_subnet
        call and the matching finish_subnet call will be in the sub-network.  The last of
        these nodes will become the output node.  All placeholder nodes in the
        subnetwork will become input nodes, in the same order they were created.

        Args:
            input_nodes: Nodes that are passed as inputs of the sub-network.
                The sub-network is expected to have placeholders of the corresponding
                number and type.
        """
        assert len(self.net_stack)
        b = _AwesomeNetBuilderData()
        b.arguments = [n.name for n in input_nodes]
        self.net_stack.append(b)

    def finish_subnet(self, name: str) -> AwesomeNode:
        """
        Finish creating a sub-network that was started by begin_subnet.

        Args:
            name: Name of the sub-network and the node that contains the sub-network.

        Returns:
            Node that contains the created sub-network.
        """
        assert len(self.net_stack) > 1
        subnet_arguments = self.net_stack[-1].arguments
        net = self._finish_net(name, True, Backend.MLA)
        mla_node = build_node.create_net_node(net, subnet_arguments, status=self.status)
        return self._record(mla_node)

    def create_placeholder_node(self, node_name: str, input_type: TensorType) -> AwesomeNode:
        assert len(self.net_stack)
        self.net_stack[-1].placeholders.append(NodeName(node_name))
        return self._record(build_node.create_placeholder_node(
            node_name, input_type, status=self.status
        ))

    def create_constant_node(self, value: np.ndarray) -> AwesomeNode:
        return self._record(build_node.create_constant_node(
            self._new_name(), np.atleast_1d(value), status=self.status
        ))

    def create_cast_node(self, data: NodeOrHandle, output_type: ScalarType) -> AwesomeNode:
        return self._record(build_node.create_cast_node(
            data, self._new_name(), output_type, status=self.status
        ))

    def create_softmax_node(self, data: NodeOrHandle, axis: int) -> AwesomeNode:
        return self._record(build_node.create_softmax_node(
            data, self._new_name(), axis, status=self.status
        ))

    def create_argmax_node(self, data: NodeOrHandle, result_scalar_type: ScalarType) -> AwesomeNode:
        return self._record(build_node.create_argmax_node(
            data, self._new_name(), result_scalar_type, status=self.status
        ))

    def create_broadcast_to_node(
            self, data: NodeOrHandle, output_shape: tuple[int, ...]
    ) -> AwesomeNode:
        return self._record(build_node.create_broadcast_to_node(
            data, self._new_name(), output_shape, status=self.status
        ))

    def create_batch_matmul_node(
            self, lhs: NodeOrHandle, rhs: NodeOrHandle, transpose_b: bool
    ) -> AwesomeNode:
        return self._record(build_node.create_batch_matmul_node(
            lhs, rhs, self._new_name(), transpose_b, status=self.status
        ))

    def create_add_node(self, lhs: NodeOrHandle, rhs: NodeOrHandle) -> AwesomeNode:
        return self._record(build_node.create_add_node(
            lhs, rhs, self._new_name(), status=self.status
        ))

    def create_subtract_node(self, lhs: NodeOrHandle, rhs: NodeOrHandle) -> AwesomeNode:
        return self._record(build_node.create_subtract_node(
            lhs, rhs, self._new_name(), status=self.status
        ))

    def create_mul_node(self, lhs: NodeOrHandle, rhs: NodeOrHandle) -> AwesomeNode:
        return self._record(build_node.create_mul_node(
            lhs, rhs, self._new_name(), status=self.status
        ))

    def create_sigmoid_node(self, data: NodeOrHandle) -> AwesomeNode:
        return self._record(build_node.create_sigmoid_node(
            data, self._new_name(), status=self.status
        ))

    def create_swish_node(self, data: NodeOrHandle) -> AwesomeNode:
        return self._record(build_node.create_swish_node(
            data, self._new_name(), status=self.status
        ))

    def create_gelu_node(self, data: NodeOrHandle) -> AwesomeNode:
        return self._record(build_node.create_gelu_node(data, self._new_name(), status=self.status))

    def create_quick_gelu_node(self, data: NodeOrHandle) -> AwesomeNode:
        return self._record(build_node.create_quick_gelu_node(
            data, self._new_name(), status=self.status
        ))

    def create_slice_concat_node(
            self, data: NodeOrHandle,
            axis: int, split_axis: int, split_block: int, split_repeat: int
    ) -> AwesomeNode:
        return self._record(build_node.create_slice_concat_node(
            data, self._new_name(), axis, split_axis, split_block, split_repeat, status=self.status
        ))

    def create_slice_node(
            self, data: NodeOrHandle,
            begin: list[int], end: list[int], stride: list[int], axis: list[int]
    ) -> AwesomeNode:
        return self._record(build_node.create_slice_node(
            data, self._new_name(), begin, end, stride, axis, status=self.status
        ))

    def create_concat_node(self, tensors: Sequence[NodeOrHandle], axis: int) -> AwesomeNode:
        return self._record(build_node.create_concat_node(
            tensors, self._new_name(), axis, status=self.status
        ))

    def create_conv_node(
            self, data: NodeOrHandle, weights: np.ndarray, bias: np.ndarray,
            conv_attrs: ConvAttrs, activation: None, scales: np.ndarray | None = None
    ) -> AwesomeNode:
        return self._record(build_node.create_conv_node(
            data, self._new_name(), weights, bias, conv_attrs, activation, scales, status=self.status
        ))

    def create_rms_norm_node(
            self, data: NodeOrHandle, epsilon: float, scale: np.ndarray
    ) -> AwesomeNode:
        return self._record(build_node.create_rms_norm_node(
            data, self._new_name(), epsilon, scale, status=self.status
        ))

    def create_layer_norm_node(
        self, data: NodeOrHandle, axis: int | tuple[int, int], epsilon: float
    ) -> AwesomeNode:
        return self._record(build_node.create_layer_norm_node(
            data, self._new_name(), axis, epsilon, status=self.status
        ))

    def create_tuple_node(self, data: list[NodeOrHandle]) -> AwesomeNode:
        return self._record(build_node.create_tuple_node(data, self._new_name(), status=self.status))

    def create_tuple_get_item_nodes(self, data: NodeOrHandle) -> list[AwesomeNode]:
        nodes = build_node.create_tuple_get_item_nodes_2(data, self._new_name(), status=self.status)
        for n in nodes:
            self._record(n)
        return nodes

    def create_einsum_node(
        self, lhs: NodeOrHandle, rhs: NodeOrHandle, equation: str, layout: str = "NHWC"
    ) -> NodeOrHandle:
        assert layout == "NHWC"
        return self._record(build_node.convert_einsum_to_batch_matmul(
            lhs, rhs, self._new_name(), equation, status=self.status
        ))

    def create_avgpool2d_node(
        self, data: NodeOrHandle, kernel_shape: tuple[int, int], strides: tuple[int, int]
    )-> NodeOrHandle:
        return self._record(build_node.create_avgpool2d_node(
            data, self._new_name(), kernel_shape, strides, status=self.status
        ))

    def create_dequantization_node(
        self, input_name: str, shape: tuple[int, ...], scale: float,
        zero_point: int = 0, input_dtype: np.dtype = np.int8, output_dtype: np.dtype = bfloat16
    ) -> AwesomeNode:
        assert np.isscalar(scale), "Per-channel quantization is not supported yet."
        dequant = build_node.DequantCast(
            shape=shape, scale=float(scale), zero_point = zero_point,
            input_dtype=input_dtype, output_dtype=output_dtype
        )
        return self._record(build_node.create_dequantization_node(
            input_name, self._new_name(), dequant, Backend.MLA)
        )


def build_conv(
        builder: SimaBuilder,
        get_param_func: Callable[[str], np.ndarray],
        check_param_func: Callable[[str], bool],
        base_name: str,
        ifm: NodeOrHandle,
        is_fc: bool = True,
        stride: tuple[int, ...] = (1, 1),
        relocatable: bool = False,
        weight_slice: tuple[int, int, int, int] | None = None,
        is_depthwise: bool = False,
        **kwargs
) -> AwesomeNode:
    """
    Build a convolution using parameters from a model.

    Args:
        builder: The SiMa graph builder.
        get_param_func: The function to get a named weight.
        check_param_func: The function to check for existence of a named weight.
        base_name: The name of the layer in the base model.
        ifm: The input node.
        is_fc: If True, treat as fully-connected layer (reshape "oi" -> "oihw").
        stride: Convolution stride.
        relocatable: If True, enable weight relocation.
        is_depthwise: If True, handle as depthwise convolution (groups=num_channels).
    """
    assert not (is_fc and is_depthwise), "is_fc and is_depthwise are mutually exclusive"

    ifm_type = get_expected_tensor_value(
        ifm.type if isinstance(ifm, NodeHandle) else ifm.get_type().output
    )

    potential_weight_name = f"{base_name}.weight"
    potential_bias_name = f"{base_name}.bias"
    reshape_str = kwargs.pop("reshape_str", "oi->oihw" if is_fc else None)
    src_weight_name = kwargs.pop("src_weight_name", potential_weight_name)
    src_bias_name = kwargs.pop("src_bias_name", potential_bias_name)
    weight_process_func = kwargs.pop("weight_process_func", lambda x: x)
    bias_process_func = kwargs.pop("bias_process_func", lambda x: x)
    q_size = kwargs.pop("q_size", None)
    kv_size = kwargs.pop("kv_size", None)

    # Some models have bundled weights with a different name for a layer.
    if not check_param_func(src_weight_name):
        src_weight_name, weight_process_func = find_alternate_weight(
            get_param_func, src_weight_name, q_size, kv_size
        )
        src_bias_name = src_weight_name.replace("weight", "bias")
        bias_process_func = weight_process_func

    params = get_param_func(src_weight_name)
    scales, weight_tensor = params if isinstance(params, tuple) else (None, params)

    # SiMaIR expects weights in the scales shape (num_c_blocks, out_channels)
    if scales is not None:
        scales = np.reshape(scales, newshape=[weight_tensor.shape[0], -1])
        scales = weight_process_func(scales)
        scales = np.transpose(scales, axes=[1, 0])

    if is_depthwise:
        # Depthwise convolution: use GHOW intermediate layout.
        if weight_tensor.ndim == 2:
            # GGUF: (G, W) -> (G, H=1, O=1, W)
            weight_tensor = layout_array(weight_tensor, "gw", "ghow")
        elif weight_tensor.ndim == 3:
            # SafeTensors: (G, H, W) -> (G, H, O=1, W)
            weight_tensor = layout_array(weight_tensor, "ghw", "ghow")
        weight_tensor = weight_process_func(weight_tensor)
        # Convert to standard HWIGO layout
        weight_tensor = layout_array(weight_tensor, "ghow", "hwigo")
    else:
        # Standard convolution: use OIHW layout when calling weight_process_func
        if reshape_str:
            src_layout, dst_layout = reshape_str.split("->")
            weight_tensor = layout_array(weight_tensor, src_layout, dst_layout)
        weight_tensor = weight_process_func(weight_tensor)
        # Convert to SiMa IR layout
        weight_tensor = layout_array(weight_tensor, "oihw", "hwigo")

    # Take part of weight tensor if slice is defined.
    if weight_slice is not None:
        num_in_channels = weight_tensor.shape[2]
        start, size, axis, idx = weight_slice
        # Normalize axis (e.g., -1 means last axis)
        axis = np.core.numeric.normalize_axis_index(axis, weight_tensor.ndim)
        weight_tensor = np.take(weight_tensor, np.arange(start, start + size), axis=axis)

        # Split scale if exists.
        if scales is not None:
            if axis == 2 and scales.shape[1] > 1:
                # Split scale by input channels axis (number of blocks).
                num_blocks = scales.shape[0]
                block_size = num_in_channels // num_blocks

                # Adjust start/size into block indices.
                start = start // block_size
                size = size // block_size
                scales = np.take(scales, np.arange(start, start + size), axis=0)
            else:
                # Split scale by output channels axis.
                scales = np.take(scales, np.arange(start, start + size), axis=1)

    if weight_tensor.dtype in [bfloat16, np.float16]:
        weight_tensor = weight_tensor.astype(np.float32)  # Model SDK requires float32

    if check_param_func(src_bias_name):
        bias_tensor = get_param_func(src_bias_name)
        bias_tensor = bias_process_func(bias_tensor)
        if bias_tensor.dtype == bfloat16:
            bias_tensor = bias_tensor.astype(np.float32)
    else:
        bias_tensor = None

    conv_attrs = attributes.ConvAttrs(
        stride=stride,
        dilation=(1,1),
        padding=((0,0), (0,0)),
        output_padding=((0,0), (0,0)),
        is_transposed=False,
        weight_shape=weight_tensor.shape,
        reloc_name=src_weight_name if relocatable else None,
        input_spatial_shape=ifm_type.shape[1:-1],
        batch_size=1,
        input_type=ifm_type.scalar
    )
    conv = builder.create_conv_node(ifm, weight_tensor, bias_tensor, conv_attrs, None, scales=scales)
    return conv


def derive_lora_name_from_base_model(base_name: str) -> str:
    """
    Derive LoRA adapter base name used in the safetensors file.
    Generally, the base name for LoRA adapter matches that for the base model.
    For VLM model, however, there is a slight twist between HF base model and LoRA adapter.

    Args:
        base_name: The base name for a layer in the base model.

    Returns:
        The base name for LoRA adapter weights.
    """
    lora_prefix = "base_model.model."
    if "language_model.model" in base_name:
        lora_base_name = base_name.replace("language_model.model", "model.language_model")
    else:
        lora_base_name = base_name
    return lora_prefix + lora_base_name


def _build_conv_lora(
        builder: SimaBuilder,
        base_name: str,
        ifm: NodeOrHandle,
        lora_shape: tuple[int, int],
        **kwargs
) -> AwesomeNode:
    """
    Build a convolution in SiMa format for a LoRA dense layer.
    """
    ifm_type = get_expected_tensor_value(
        ifm.type if isinstance(ifm, NodeHandle) else ifm.get_type().output
    )

    # Convert all-zero LoRA weight to SiMa IR layout.
    weight_tensor = np.zeros(lora_shape, dtype=np.float32)
    weight_tensor = layout_array(weight_tensor, "oi", "hwigo")

    scales = None
    bias_tensor = None

    # Construct name of LoRA weight in safetensors file.
    lora_base_name = derive_lora_name_from_base_model(base_name)

    conv_attrs = attributes.ConvAttrs(
        stride=(1,1),
        dilation=(1,1),
        padding=((0,0), (0,0)),
        output_padding=((0,0), (0,0)),
        is_transposed=False,
        weight_shape=weight_tensor.shape,
        reloc_name=f"{lora_base_name}.weight",
        input_spatial_shape=ifm_type.shape[1:-1],
        batch_size=1,
        input_type=ifm_type.scalar
    )
    conv = builder.create_conv_node(ifm, weight_tensor, bias_tensor, conv_attrs, None, scales=scales)
    return conv


def build_conv_from_dense_with_lora(
    builder: SimaBuilder,
    get_param_func: Callable[[str], np.ndarray],
    check_param_func: Callable[[str], bool],
    base_name: str,
    ifm: NodeOrHandle,
    lora_rank: int | None = None,
    merged_lora: bool = False,
    weight_slice: tuple[int, int, int, int] | None = None,
    **kwargs
) -> NodeOrHandle:
    """Builds a conv from dense op with LoRA branch.

    Args:
        builder: The SiMa builder.
        get_param_func: The function to get a named weight.
        check_param_func: The function to check for existence of a named weight.
        base_name: The name of the layer in the base model.
        ifm: The input node.
        lora_rank: The rank of LoRA adapter.
        merged_lora: Whether or not lora adapter is merged to the base model.
        weight_slice: Optional specification of a sub-region of the weight tensor. Includes
            size, offset, axis and index of weight slice. If None, use the entire weight.
    Returns:
        Created conv node or merged node of the conv and LoRA branch.
    """
    if weight_slice is not None and lora_rank is not None:
        raise NotImplementedError("Conv slicing is not supported for LoRA.")

    if lora_rank and merged_lora:
        proj = build_conv(
            builder, get_param_func, check_param_func, base_name, ifm, relocatable=True, **kwargs
        )
        return proj

    proj = build_conv(
        builder, get_param_func, check_param_func, base_name, ifm, weight_slice=weight_slice,
        **kwargs
    )

    if lora_rank:
        weight_shape = proj.ir._attrs.conv_attrs.weight_shape
        # Weight shape is "hwigo"
        output_channels, input_channels = weight_shape[-1], weight_shape[-3]
        a_shape = (lora_rank, input_channels)
        b_shape = (output_channels, lora_rank)

        lora_a = _build_conv_lora(
            builder, f"{base_name}.lora_A", ifm, a_shape
        )
        lora_b = _build_conv_lora(
            builder, f"{base_name}.lora_B", lora_a, b_shape
        )
        proj = builder.create_add_node(proj, lora_b)
    return proj


def build_logit_softcapping(
        builder: SimaBuilder, input_node: NodeOrHandle, scalar: float, quantizable: bool
) -> NodeOrHandle:
    """Build nodes for logit soft capping.

    Logit soft capping is used in GEMMA2 to prevent overconfident predictions.

        softcapping(x) = scalar * tanh(x/scalar)
                       = scalar * [2*sigmoid(2x/scalar)-1]
                       = (2*scalar) * sigmoid(x * (2/scalar)) - scalar
        operations: x - mul - sigmoid - mul - sub
    """
    dtype = activation_dtype(quantizable)
    mul1 = builder.create_mul_node(
        input_node, builder.create_constant_node(np.array(2.0/scalar, dtype=dtype))
    )
    sig = builder.create_sigmoid_node(mul1)
    mul2 = builder.create_mul_node(
        sig, builder.create_constant_node(np.array(2.0*scalar, dtype=dtype))
    )
    return builder.create_add_node(
        mul2, builder.create_constant_node(np.array([scalar], dtype=dtype))
    )


def build_activation(
    builder: SimaBuilder, input_node: NodeOrHandle, act_type: str, quantizable: bool
) -> NodeOrHandle:
    """Build nodes for activation.

    LLAMA uses "silu" which uses sigmoid.
    GEMMA uses "gelu_pytorch_tanh" which uses Gaussian ERF with tanh approximation.

    Because tanh(x) = 2*sigmoid(2x)-1, gelu can also be approximated by sigmoid.
        gelu_tanh(x) = 0.5 * x * [1 + tanh(root(2/PI)*x*(1 + 0.044715 * x * x))]
                     = x * sigmoid(x*(A + B * x * x))
        where A = 2 * root(2/PI), B = A * 0.044715

    Args:
        builder: Builder where the created nodes will be recorded
        input_node: Input of the activation layer
        act_type: Type of activation
        quantizable: Whether to create nodes for input to the quantizer

    Returns:
        Output node of the created activation layer
    """
    match act_type:
        case "silu":
            last = builder.create_swish_node(input_node)
        case "gelu":
            last = builder.create_gelu_node(input_node)
        case "gelu_tanh" | "gelu_pytorch_tanh":
            dtype = activation_dtype(quantizable)
            value_a = 2 * math.sqrt(2 / math.pi)
            value_b = 2 * math.sqrt(2 / math.pi) * 0.044715
            const_a = builder.create_constant_node(np.array(value_a, dtype=dtype))
            const_b = builder.create_constant_node(np.array(value_b, dtype=dtype))
            square = builder.create_mul_node(input_node, input_node)
            b_mul = builder.create_mul_node(square, const_b)
            a_add = builder.create_add_node(b_mul, const_a)
            x_mul = builder.create_mul_node(input_node, a_add)
            sig = builder.create_sigmoid_node(x_mul)
            last = builder.create_mul_node(input_node, sig)
        case "quick_gelu":
            last = builder.create_quick_gelu_node(input_node)
        case _:
            raise ValueError(f"Unsupported activation: {act_type}")
    return last


def layout_array(a: np.ndarray, in_layout: str, out_layout: str) -> np.ndarray:
    """
    Change an array's layout according to the given layout strings.
    The array is transposed and new dimensions are created to match the output layout.

    To add 3 extra dimensions of size 1:
    > layout_array(a, "c", "nhwc")

    To transpose from NCHW to NHWC layout:
    > layout_array(a, "nchw", "nhwc")
    """
    # Ensure no duplicate symbols in the layout strings
    assert len(set(in_layout)) == len(in_layout)
    assert len(set(out_layout)) == len(out_layout)

    # Count the number of extra dimensions to create so that
    # input and output have the same dimensionality
    dummy_dimensions = len(out_layout) - len(in_layout)
    assert dummy_dimensions >= 0

    # Find permutation on dimensions.  For the permutation, the
    # input tensor has extra dimensions added starting at index 0.
    dummy_dim = 0
    permutation = []
    for d in out_layout:
        try:
            i = in_layout.index(d) + dummy_dimensions
        except ValueError:
            i = dummy_dim
            dummy_dim += 1
        permutation.append(i)

    a = np.expand_dims(a, tuple(range(dummy_dimensions)))
    return np.transpose(a, permutation)


def activation_type(quantizable: bool) -> ScalarType:
    """
    Return the data type to use for most node inputs and outputs.
    We use float32 in models that will be processed by the quantizer.
    We use bfloat16 in models that will be executed.
    """
    return ScalarType.float32 if quantizable else ScalarType.bfloat16


def activation_dtype(quantizable: bool) -> np.dtype:
    """
    Return the data type to use for most node inputs and outputs.
    We use float32 in models that will be processed by the quantizer.
    We use bfloat16 in models that will be executed.
    """
    return np.float32 if quantizable else bfloat16


def load_tensor_from_source(
    source_name: str,
    get_param_func: Callable[[str], np.ndarray],
    check_param_func: Callable[[str], bool],
    reshape_str: str | None = None
) -> np.ndarray:
    assert check_param_func(source_name), f"No such tensor: {source_name}"
    t = get_param_func(source_name)
    if reshape_str:
        src_layout, dst_layout = reshape_str.split("->")
        assert len(t.shape) == len(src_layout)
        t = layout_array(t, src_layout, dst_layout)
    if t.dtype == bfloat16:
        t = t.astype(np.float32)  # Model SDK requires float32
    return t


def build_two_stage_layer_norm(
    builder: SimaBuilder,
    get_param_func: Callable[[str], np.ndarray],
    check_param_func: Callable[[str], bool],
    base_name: str,
    input_node: NodeOrHandle,
    axis: int,
    epsilon: float
) -> NodeOrHandle:
    """
    Build nodes for layer norm with scale and bias.
    The first stage is just layer norm, and the second stage is multiply by scale and
        add by bias, which is fused as a depthwise convolution.

    Args:
        builder: The SiMa graph builder.
        get_param_func: The function to get a named weight.
        check_param_func: The function to check for existence of a named weight.
        base_name: The name of the layer in the base model.
        input_node: The input node.
        axis: The axis to perform layer norm.
        epsilon: The small value to avoid division by zero.

    Returns:
        The output node of the created layer norm.
    """
    layer_norm = builder.create_layer_norm_node(
        input_node, axis, epsilon
    )

    # Depthwise convolution for element-wise scale and bias.
    ifm_type = get_expected_tensor_value(
        layer_norm.type if isinstance(layer_norm, NodeHandle) else layer_norm.get_type().output
    )
    n_channels = ifm_type.shape[-1]
    weight_shape=(1, 1, 1, n_channels, 1)
    tensor_name = f"{base_name}.weight"
    if check_param_func(tensor_name):
        weight_tensor = load_tensor_from_source(tensor_name, get_param_func, check_param_func)
        assert len(weight_tensor.shape) == 1 and weight_tensor.shape[0] == n_channels
    else:
        weight_tensor = np.ones((n_channels, ))
    weight_tensor = weight_tensor.reshape(weight_shape).astype(np.float32)

    tensor_name = f"{base_name}.bias"
    if check_param_func(tensor_name):
        bias_tensor = load_tensor_from_source(tensor_name, get_param_func, check_param_func)
        assert len(bias_tensor.shape) == 1 and bias_tensor.shape[0] == n_channels
    else:
        bias_tensor = None

    conv_attrs = attributes.ConvAttrs(
        stride=(1,1),
        dilation=(1,1),
        padding=((0,0), (0,0)),
        output_padding=((0,0), (0,0)),
        is_transposed=False,
        weight_shape=weight_shape,
        reloc_name=None,
        input_spatial_shape=ifm_type.shape[1:-1],
        batch_size=1,
        input_type=ifm_type.scalar
    )
    return builder.create_conv_node(layer_norm, weight_tensor, bias_tensor, conv_attrs, None)


def build_matmul_and_split_heads(
    builder: SimaBuilder,
    get_param_func: Callable[[str], np.ndarray],
    check_param_func: Callable[[str], bool],
    base_name: str,
    input_node: NodeOrHandle,
    num_heads: int,
    seq_len: int,
    post_matmul_scale: float = 1.0
) -> list[NodeOrHandle]:
    """
    Build multi-head projection.
    If the hardware constraint of MLA is satisfied, the matrix multiply is implemented as a single
    convolution, followed by spilt and concat. Otherwise, the matrix is split into sub matrices for
    the specified number of heads, each implemented as a convolution.

    Args:
        builder: The SiMa graph builder.
        get_param_func: The function to get a named weight.
        check_param_func: The function to check for existence of a named weight.
        base_name: The name of the layer in the base model.
        input_node: The input node.
        num_heads: The number of heads to split.
        seq_len: The length of token sequence.
        post_matmul_scale: The scaling factor after matrix multiply. Default is 1.0.

    Returns:
        The list of output nodes.
    """
    kv_len = seq_len
    elem_size = 2
    num_mla_rows_per_head = seq_len * ceil_div_row(kv_len) * elem_size
    concat_heads = num_mla_rows_per_head <= mla_max_num_rows
    if concat_heads:
        def param_process_func(x: np.ndarray) -> np.ndarray:
            assert x.shape[0] % num_heads == 0
            head_dim = x.shape[0] // num_heads
            if head_dim % mla_row_size:
                rounded_head_dim = round_up_to_row(head_dim)
                x = x.reshape(num_heads, head_dim, *x.shape[1:])
                pad_width = [(0, 0)] * x.ndim
                pad_width[1] = (0, rounded_head_dim - head_dim)
                x = np.pad(x, pad_width)
                x = x.reshape(-1, *x.shape[2:])
            return x * post_matmul_scale
        matmul = build_conv(
            builder, get_param_func, check_param_func,
            f"{base_name}.matmul", input_node, weight_process_func=param_process_func,
            bias_process_func=param_process_func, src_weight_name=f"{base_name}.weight",
            src_bias_name=f"{base_name}.bias"
        )
        # NHWC layout: split on axis C and concat on axis H
        reshape = builder.create_slice_concat_node(
            matmul, axis=1, split_axis=3, split_block=num_heads, split_repeat=1
        )
        return [reshape]
    else:
        heads = list()
        for i in range(num_heads):
            def param_process_func(x: np.ndarray) -> np.ndarray:
                assert x.shape[0] % num_heads == 0
                head_dim = x.shape[0] // num_heads
                begin = i * head_dim
                end = begin + head_dim
                return x[begin:end] * post_matmul_scale
            matmul = build_conv(
                builder, get_param_func, check_param_func,
                f"{base_name}.matmul.{i}", input_node, weight_process_func=param_process_func,
                bias_process_func=param_process_func, src_weight_name=f"{base_name}.weight",
                src_bias_name=f"{base_name}.bias"
            )
            heads.append(matmul)
        return heads


def build_merge_heads_and_matmul(
    builder: SimaBuilder,
    get_param_func: Callable[[str], np.ndarray],
    check_param_func: Callable[[str], bool],
    base_name: str,
    input_nodes: list[NodeOrHandle],
    num_heads: int
) -> NodeOrHandle:
    """
    Build output projection of attention block.
    The input nodes are concatenated, followed by a matrix multiply, implemented as a convolution.

    Args:
        builder: The SiMa graph builder.
        get_param_func: The function to get a named weight.
        check_param_func: The function to check for existence of a named weight.
        base_name: The name of the layer in the base model.
        input_nodes: The list of input nodes.
        num_heads: The number of heads to merge.

    Returns:
        The output node of attention output.
    """
    if len(input_nodes) == 1:
        matmul_input = input_nodes[0]
    else:
        assert len(input_nodes) == num_heads
        # Heads are not merged. Concatenate first.
        matmul_input = builder.create_concat_node(input_nodes, axis=1)

    # NHWC layout: split on axis H and concat on axis C
    reshape = builder.create_slice_concat_node(
        matmul_input, axis=3,
        split_axis=1,
        split_block=num_heads,
        split_repeat=1
    )

    def param_process_func(x: np.ndarray) -> np.ndarray:
        assert x.shape[1] % num_heads == 0
        assert x.shape[2] == x.shape[3] == 1
        head_dim = x.shape[1] // num_heads
        if len(input_nodes) == 1 and head_dim % mla_row_size:
            x = x.reshape(x.shape[0], num_heads, head_dim, 1)
            rounded_head_dim = round_up_to_row(head_dim)
            x = np.pad(x, ((0, 0), (0, 0), (0, rounded_head_dim - head_dim), (0, 0)))
            x = x.reshape(x.shape[0], -1, 1, 1)
        return x

    conv = build_conv(
        builder, get_param_func, check_param_func,
        base_name, reshape,
        weight_process_func=param_process_func
    )
    return conv
