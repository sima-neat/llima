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
from afe.ir.sima_builder import SimaBuilder

from sima_lmm.model.onnx_builder import find_alternate_weight
from sima_lmm.utils import (
    ceil_div_row, mla_max_num_rows, mla_row_size, round_up_to_row
)


bfloat16 = ScalarType.numpy_type(ScalarType.bfloat16)


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
    activation = kwargs.pop("activation", None)

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
        if bias_tensor.dtype in (bfloat16, np.float16):
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
    conv = builder.create_conv_node(ifm, weight_tensor, bias_tensor, conv_attrs, activation, scales=scales)
    return conv


def create_channel_slice(
        builder: SimaBuilder,
        input_node: NodeOrHandle,
        begin: int,
        end: int,
) -> NodeOrHandle:
    """
    Create a channel-axis slice that respects MLA channel slice alignment.

    The TVM/ONNX path rewrites channel slices with non-16-aligned boundaries into 1x1
    selector convolutions. The direct SiMaBuilder path needs to do the same for
    models such as Qwen2.5-VL, whose RoPE split is 80 -> 40 + 40.
    """
    input_type = get_expected_tensor_value(
        input_node.type if isinstance(input_node, NodeHandle) else input_node.get_type().output
    )
    channel_axis = len(input_type.shape) - 1
    assert 0 <= begin < end <= input_type.shape[channel_axis]

    if begin % 16 == 0 and end % 16 == 0:
        return builder.create_slice_node(input_node, [begin], [end], [1], [channel_axis])

    input_channels = input_type.shape[channel_axis]
    output_channels = end - begin
    weights = np.zeros((1, 1, input_channels, 1, output_channels), dtype=np.float32)
    for out_channel in range(output_channels):
        weights[0, 0, begin + out_channel, 0, out_channel] = 1.0

    conv_attrs = attributes.ConvAttrs(
        stride=(1, 1),
        dilation=(1, 1),
        padding=((0, 0), (0, 0)),
        output_padding=((0, 0), (0, 0)),
        is_transposed=False,
        weight_shape=weights.shape,
        reloc_name=None,
        input_spatial_shape=input_type.shape[1:-1],
        batch_size=input_type.shape[0],
        input_type=input_type.scalar,
    )
    return builder.create_conv_node(input_node, weights, None, conv_attrs)


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


def build_space_to_depth(
    builder: SimaBuilder, data: NodeOrHandle, blocksize: int
) -> AwesomeNode:
    """
    Implement SpaceToDepth as a strided convolution with a binary weight pattern.

    Input NHWC (1, H, W, C) -> output NHWC (1, H//blocksize, W//blocksize, C*blocksize*blocksize).
    """
    ifm_type = get_expected_tensor_value(
        data.type if isinstance(data, NodeHandle) else data.get_type().output
    )
    in_channels = ifm_type.shape[3]
    out_channels = in_channels * (blocksize ** 2)

    weights = np.zeros((out_channels, in_channels, blocksize, blocksize), dtype=np.float32)
    for i in range(in_channels):
        for j in range(blocksize):
            for k in range(blocksize):
                weights[i + (j * blocksize + k) * in_channels, i, j, k] = 1.0

    weights = np.transpose(np.expand_dims(weights, 0), (3, 4, 2, 0, 1))
    conv_attrs = ConvAttrs(
        stride=(blocksize, blocksize),
        dilation=(1, 1),
        padding=((0, 0), (0, 0)),
        output_padding=((0, 0), (0, 0)),
        is_transposed=False,
        weight_shape=weights.shape,
        reloc_name=None,
        input_spatial_shape=ifm_type.shape[1:3],
        batch_size=1,
        input_type=ifm_type.scalar,
    )
    return builder.create_conv_node(data, weights, None, conv_attrs)


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
            # AFE's GELU node does not support bfloat16, so expand GELU via Erf.
            dtype = activation_dtype(quantizable)
            scaled = builder.create_mul_node(
                input_node,
                builder.create_constant_node(np.array(1 / math.sqrt(2), dtype=dtype)),
            )
            erf = builder.create_erf_node(scaled)
            shifted = builder.create_add_node(
                erf, builder.create_constant_node(np.array(1.0, dtype=dtype))
            )
            mul = builder.create_mul_node(input_node, shifted)
            last = builder.create_mul_node(
                mul, builder.create_constant_node(np.array(0.5, dtype=dtype))
            )
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
    if t.dtype in (bfloat16, np.float16):
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
