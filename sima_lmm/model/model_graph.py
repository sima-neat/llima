"""Common native graph operations with model weights and precision bound once."""

import math
from typing import Callable, Sequence, TypedDict, Unpack

import numpy as np
from afe.apis.defines import gen2_target
from afe.backends.backends import Backend
from afe.ir.attributes import ClipAttrs, ConvAttrs, ReluAttrs
from afe.ir.build_node import NodeOrHandle, TopKRetType, as_handle
from afe.ir.defines import Status, get_expected_tensor_value
from afe.ir.execute import create_node_quant_executor
from afe.ir.net import AwesomeNet
from afe.ir.node import AwesomeNode
from afe.ir.serializer import save_awesomenet
from afe.ir.sima_builder import SimaBuilder
from afe.ir.tensor_type import ScalarType, TensorType

from sima_lmm.model.base import BaseModel
from sima_lmm.utils import ceil_div_row, mla_max_num_rows, mla_row_size, round_up_to_row


_bfloat16 = ScalarType.numpy_type(ScalarType.bfloat16)
Node = NodeOrHandle


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
    return np.float32 if quantizable else _bfloat16


class WeightOptions(TypedDict, total=False):
    """Overrides shared by projections and convolutions; source metadata is retained."""

    src_weight_name: str
    src_bias_name: str
    reshape_str: str | None
    weight_process_func: Callable[[np.ndarray], np.ndarray]
    scale_process_func: Callable[[np.ndarray], np.ndarray]
    bias_process_func: Callable[[np.ndarray], np.ndarray]
    q_size: int
    kv_size: int
    relocatable: bool
    activation: ReluAttrs | ClipAttrs | None
    expert_idx: int
    de_interleave: bool


def _validate_weight_options(options: dict) -> None:
    """Reject unsupported projection or convolution metadata overrides."""
    unknown = options.keys() - WeightOptions.__annotations__.keys()
    if unknown:
        raise ValueError(
            f"Unknown weight options: {sorted(unknown)}; see WeightOptions for supported overrides"
        )


def tensor_type(node: NodeOrHandle) -> TensorType:
    """Infer the tensor type of an AFE node or handle."""
    return get_expected_tensor_value(as_handle(node).type)


class ModelGraph(SimaBuilder):
    """AFE SimaBuilder with model weights, precision and a single MLA subnet.

    Common model operations and native AFE create_* methods use the same graph.
    Floating constants use the activation precision; tessellation hooks still apply.
    """

    def add(
        self,
        lhs: NodeOrHandle,
        rhs: NodeOrHandle,
        activation: ReluAttrs | ClipAttrs | None = None,
        *,
        backend: Backend = Backend.NONE,
    ) -> NodeOrHandle:
        """Add tensors elementwise with broadcasting and optional fused activation."""
        return self.create_add_node(lhs, rhs, activation, backend=backend)

    def sub(self, lhs: NodeOrHandle, rhs: NodeOrHandle) -> NodeOrHandle:
        """Subtract rhs from lhs elementwise with broadcasting."""
        return self.create_subtract_node(lhs, rhs)

    def mul(self, lhs: NodeOrHandle, rhs: NodeOrHandle) -> NodeOrHandle:
        """Multiply tensors elementwise with broadcasting."""
        return self.create_mul_node(lhs, rhs)

    def matmul(
        self, lhs: NodeOrHandle, rhs: NodeOrHandle,
        transpose_a: bool = False, transpose_b: bool = False,
    ) -> NodeOrHandle:
        """Multiply rank-four matrices, optionally transposing their last two axes.

        Batches must match. Divisible head counts use MLA's implicit head repetition.
        """
        a, b = tensor_type(lhs), tensor_type(rhs)
        if len(a.shape) != 4 or len(b.shape) != 4:
            raise ValueError(f"matmul needs rank-four inputs; got {a.shape} and {b.shape}")
        if a.scalar != b.scalar or a.scalar not in (ScalarType.float32, ScalarType.bfloat16):
            raise ValueError(f"matmul needs matching FP32/BF16 inputs; got {a.scalar} and {b.scalar}")
        if a.shape[0] != b.shape[0] or max(a.shape[1], b.shape[1]) % min(a.shape[1], b.shape[1]):
            raise ValueError(
                f"matmul needs equal batches and divisible head counts; got {a.shape} and {b.shape}"
            )
        if a.shape[2 if transpose_a else 3] != b.shape[3 if transpose_b else 2]:
            raise ValueError(f"matmul has mismatched contraction dimensions: {a.shape} and {b.shape}")
        return self.create_batch_matmul_node(lhs, rhs, transpose_a, transpose_b)

    def concat(self, tensors: Sequence[NodeOrHandle], axis: int) -> NodeOrHandle:
        """Join tensors along axis, preserving their supplied order."""
        return self.create_concat_node(tensors, axis)

    def transpose(self, data: NodeOrHandle, axes: list[int]) -> NodeOrHandle:
        """Reorder tensor dimensions according to axes."""
        return self.create_transpose_node(data, axes)

    def reshape(self, data: NodeOrHandle, new_shape: list[int]) -> NodeOrHandle:
        """Change tensor shape while preserving element order."""
        return self.create_reshape_node(data, new_shape)

    def softmax(self, data: NodeOrHandle, axis: int = -1) -> NodeOrHandle:
        """Convert values into probabilities along axis, defaulting to the last axis."""
        return self.create_softmax_node(data, axis)

    def topk(self, data: NodeOrHandle, k: int) -> tuple[NodeOrHandle, NodeOrHandle]:
        """Return the largest k values and their INT32 indices along the last axis."""
        return (
            self.create_topk_node(data, k, TopKRetType.VALUES),
            self.create_topk_node(data, k, TopKRetType.INDICES),
        )

    def sum_channels(self, data: NodeOrHandle) -> NodeOrHandle:
        """Sum the channels of a rank-four tensor through a weightless 1x1 projection."""
        channels = tensor_type(data).shape[-1]
        weights = np.ones((1, channels), dtype=np.float32)
        return self._build_conv(
            "channel_sum", data, get_param_func=lambda _: weights,
            check_param_func=lambda name: name.endswith(".weight"),
        )

    def sigmoid(self, data: NodeOrHandle) -> NodeOrHandle:
        """Apply 1 / (1 + exp(-x)) elementwise."""
        return self.create_sigmoid_node(data)

    def relu(self, data: NodeOrHandle, *, backend: Backend = Backend.MLA) -> NodeOrHandle:
        """Clamp negative values to zero elementwise."""
        return self.create_relu_node(data, backend=backend)

    def exp(self, data: NodeOrHandle) -> NodeOrHandle:
        """Compute exp(x) elementwise."""
        return self.create_exp_node(data)

    def reciprocal(self, data: NodeOrHandle) -> NodeOrHandle:
        """Compute 1 / x elementwise."""
        return self.create_reciprocal_node(data)

    def softplus(self, data: NodeOrHandle) -> NodeOrHandle:
        """Compute log(1 + exp(x)) elementwise."""
        return self.create_softplus_node(data)

    def cast(
        self, data: NodeOrHandle, output_type: ScalarType, *, backend: Backend = Backend.NONE
    ) -> NodeOrHandle:
        """Convert tensor values to output_type on the requested backend."""
        return self.create_cast_node(data, output_type, backend=backend)

    def clip(self, data: NodeOrHandle, a_min: float, a_max: float) -> NodeOrHandle:
        """Clamp tensor values to the interval [a_min, a_max]."""
        return self.create_clip_node(data, a_min, a_max)

    def avgpool2d(
        self, data: NodeOrHandle, kernel_shape: tuple[int, int], strides: tuple[int, int]
    ) -> NodeOrHandle:
        """Average spatial windows in NHWC layout using kernel_shape and strides."""
        return self.create_avgpool2d_node(data, kernel_shape, strides)

    def split_concat(
        self, data: NodeOrHandle, axis: int, split_axis: int, split_block: int, split_repeat: int
    ) -> NodeOrHandle:
        """Split along split_axis and join repeated chunks along axis."""
        return self.create_slice_concat_node(data, axis, split_axis, split_block, split_repeat)

    def __init__(
        self,
        model: BaseModel,
        input_specs: dict[str, tuple[int, ...] | TensorType],
        quantizable: bool,
        *,
        input_dtypes: dict[str, np.dtype | type | str] | None = None,
    ):
        """Create matching outer/subnet inputs; optional NumPy dtypes override precision."""
        self.model = model
        self.quantizable = quantizable
        self.dtype = np.dtype(activation_dtype(quantizable))
        self._finished_net: AwesomeNet | None = None
        input_dtypes = {} if input_dtypes is None else input_dtypes
        unknown = input_dtypes.keys() - input_specs.keys()
        if unknown:
            raise ValueError(f"Input dtypes supplied for unknown inputs: {sorted(unknown)}")
        types = {}
        for name, spec in input_specs.items():
            shape = spec.shape if isinstance(spec, TensorType) else spec
            if (
                not isinstance(name, str)
                or not name
                or not shape
                or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
            ):
                raise ValueError(
                    f"Invalid model input {name!r}: expected a name and positive static dimensions, got {shape}"
                )
            if isinstance(spec, TensorType):
                if name in input_dtypes:
                    raise ValueError(f"{name}: specify either TensorType or input_dtypes, not both")
                types[name] = spec
            else:
                dtype = input_dtypes.get(name, self.dtype)
                try:
                    scalar = ScalarType.from_numpy(np.dtype(dtype))
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{name}: unsupported input dtype {dtype!r}") from error
                types[name] = TensorType(scalar, shape)
        super().__init__(Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target)
        outer_inputs = [self.create_placeholder_node(name, spec) for name, spec in types.items()]
        self.begin_subnet(outer_inputs)
        self.inputs = {
            name: self.create_placeholder_node(f"MLA_0/{name}", spec)
            for name, spec in types.items()
        }

    def finish(
        self,
        outputs: Sequence[NodeOrHandle],
        *,
        transform_subnet: Callable[[AwesomeNet], None] | None = None,
    ) -> AwesomeNet:
        """Finish the subnet, preserving output order and integer types.

        BF16 outputs are cast to FP32 on EV. The optional transform runs before
        outer outputs are extracted.
        """
        if self._finished_net is not None:
            raise RuntimeError("Graph is already finished; use run() or save() on the finished graph")
        if not outputs:
            raise ValueError("A model graph needs at least one output")
        # Explicitly select outputs even when they are not the last nodes created.
        if len(outputs) > 1:
            self.create_tuple_node(list(outputs))
        mla = self.finish_subnet("MLA_0")
        if len(outputs) == 1:
            # Keep a tensor output; the compiler can flatten singleton tuples.
            mla.ir.output_node_name = as_handle(outputs[0]).name
        if transform_subnet is not None:
            transform_subnet(mla.ir)
        model_outputs = [mla] if len(outputs) == 1 else self.create_tuple_get_item_nodes(mla)
        for i, output in enumerate(model_outputs):
            if tensor_type(output).scalar == ScalarType.bfloat16:
                model_outputs[i] = self.cast(output, ScalarType.float32, backend=Backend.EV)
        if len(model_outputs) > 1:
            self.create_tuple_node(model_outputs)
        self._finished_net = super().finish(self.model.model_name)
        return self._finished_net

    def run(self, *, use_jax: bool = False, **inputs: np.ndarray) -> list[np.ndarray]:
        """Execute the finished graph with named NumPy inputs, without casting.

        NumPy execution uses AFE fast mode. JAX selects its reference implementation,
        where fast mode has no effect. JAX operations use its configured backend.
        Outputs follow the order supplied to finish().
        """
        if self._finished_net is None:
            raise RuntimeError("Call finish(outputs) or save(outputs) before run()")
        if not isinstance(use_jax, bool):
            raise TypeError("use_jax must be a bool")
        missing = self.inputs.keys() - inputs.keys()
        unexpected = inputs.keys() - self.inputs.keys()
        if missing or unexpected:
            raise ValueError(
                f"Graph input mismatch: missing {sorted(missing)}, unexpected {sorted(unexpected)}"
            )
        for name, value in inputs.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name}: expected a NumPy array, got {type(value).__name__}")
            expected = get_expected_tensor_value(self._finished_net.nodes[name].get_type().output)
            if value.shape != expected.shape:
                raise ValueError(f"{name}: expected shape {expected.shape}, got {value.shape}")
            dtype = np.dtype(expected.scalar.numpy_type())
            if value.dtype != dtype:
                raise TypeError(f"{name}: expected dtype {dtype}, got {value.dtype}")
        return self._finished_net.run(
            inputs, node_callable=create_node_quant_executor(fast_mode=True, use_jax=use_jax)
        )

    def save(
        self,
        outputs: Sequence[NodeOrHandle] | None = None,
        *,
        transform_subnet: Callable[[AwesomeNet], None] | None = None,
    ) -> None:
        """Save the graph, supplying outputs only when it has not been finished yet."""
        if self._finished_net is None:
            if outputs is None:
                raise ValueError("Supply outputs to save(), or call finish(outputs) first")
            self.finish(outputs, transform_subnet=transform_subnet)
        elif outputs is not None or transform_subnet is not None:
            raise ValueError("Graph is already finished; call save() without outputs or transform_subnet")
        save_awesomenet(
            self._finished_net,
            self.model.model_name + (".fp32" if self.quantizable else ""),
            str(self.model.sima_model_sdk_path),
        )

    def parameter(self, name: str) -> np.ndarray | tuple:
        """Load source data, retaining packed weights, scales and block metadata."""
        if not self.model.check_hf_param(name):
            raise ValueError(f"{self.model.model_name}: missing source tensor {name!r}")
        return self.model.get_hf_param(name)

    def constant(
        self, value: np.ndarray | Sequence | float | int, *, dtype: np.dtype | type | None = None
    ) -> NodeOrHandle:
        """Create a deterministically named constant; infer floating activation type."""
        data = np.asarray(value)
        if dtype is None:
            dtype = (
                self.dtype
                if data.dtype.kind == "f" or data.dtype == activation_dtype(False)
                else data.dtype
            )
        return self.create_constant_node(data.astype(dtype))

    def linear(
        self,
        name: str,
        data: NodeOrHandle,
        *,
        scale: float = 1.0,
        lora_rank: int | None = None,
        merged_lora: bool = False,
        **kwargs: Unpack[WeightOptions],
    ) -> NodeOrHandle:
        """Project channels with OI weights; preserve bias, scales, blocks and relocation.

        WeightOptions documents keyword overrides, including source
        names and matching weight/scale/bias transforms for fused or sliced weights.
        Optional scale is folded into floating weights, quantization scales and bias.
        """
        _validate_weight_options(kwargs)
        if len(tensor_type(data).shape) != 4:
            raise ValueError(f"{name}: linear projection needs rank-four NHWC input")
        if scale != 1.0:
            if lora_rank and not merged_lora:
                raise ValueError(f"{name}: projection scaling requires merged LoRA weights")
            for key in ("weight_process_func", "scale_process_func", "bias_process_func"):
                transform = kwargs.get(key, lambda x: x)

                def scaled(x, transform=transform):
                    value = transform(x)
                    floating = value.dtype.kind == "f" or value.dtype == activation_dtype(False)
                    return value * scale if floating else value

                kwargs[key] = scaled
        if lora_rank and merged_lora:
            kwargs["relocatable"] = True
        proj = self._build_conv(name, data, **kwargs)

        if lora_rank and not merged_lora:
            weight_shape = proj.ir._attrs.conv_attrs.weight_shape
            # Weight shape is "hwigo"
            output_channels, input_channels = weight_shape[-1], weight_shape[-3]
            a_shape = (lora_rank, input_channels)
            b_shape = (output_channels, lora_rank)

            bundled_expert = (
                kwargs.get("expert_idx", -1) >= 0
                and not self.model.check_hf_param(f"{name}.weight")
            )
            lora_name = f"{name}.expert.{kwargs['expert_idx']}" if bundled_expert else name
            lora_a = self._build_conv_lora(f"{lora_name}.lora_A", data, a_shape)
            lora_b = self._build_conv_lora(f"{lora_name}.lora_B", lora_a, b_shape)
            proj = self.create_add_node(proj, lora_b)
        return proj

    def slice(
        self,
        data: NodeOrHandle,
        begin: list[int] | None = None,
        end: list[int] | None = None,
        stride: list[int] | None = None,
        axis: int | list[int] | None = None,
        *,
        start: int | None = None,
        stop: int | None = None,
    ) -> NodeOrHandle:
        """Slice with axis lists or start/stop on one axis, with channel alignment handling.

        The single-axis form defaults to start=0, requires axis and stop,
        and uses stride one with 0 <= start < stop <= the axis length.
        """
        if axis is None:
            raise ValueError("slice requires an explicit axis")
        spec = tensor_type(data)
        if start is not None or stop is not None:
            if any(value is not None for value in (begin, end, stride)):
                raise ValueError("slice cannot mix start/stop with begin/end/stride")
            if not isinstance(axis, int) or not -len(spec.shape) <= axis < len(spec.shape):
                raise ValueError(
                    f"slice axis must be an integer in [-{len(spec.shape)}, "
                    f"{len(spec.shape) - 1}]; got {axis}"
                )
            start = 0 if start is None else start
            if not isinstance(start, int) or not isinstance(stop, int):
                raise ValueError("slice requires integer start/stop bounds; stop is required")
            if not 0 <= start < stop <= spec.shape[axis]:
                raise ValueError(
                    f"slice needs 0 <= start < stop <= {spec.shape[axis]} on axis {axis}; "
                    f"got start={start}, stop={stop}"
                )
            begin, end, stride, axis = [start], [stop], [1], [axis]
        elif any(value is None for value in (begin, end, stride)) or isinstance(axis, int):
            raise ValueError(
                "slice requires begin/end/stride and axis lists, or start/stop with one axis"
            )
        if (
            len(spec.shape) == 4
            and spec.scalar in (ScalarType.float32, ScalarType.bfloat16)
            and len(begin) == len(end) == len(stride) == len(axis) == 1
            and axis[0] in (3, -1)
            and stride[0] == 1
            and 0 <= begin[0] < end[0] <= spec.shape[-1]
        ):
            return self._create_channel_slice(data, begin[0], end[0])
        return self.create_slice_node(data, begin, end, stride, axis)

    def conv(
        self,
        name: str,
        data: NodeOrHandle,
        *,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0)),
        is_depthwise: bool = False,
        **kwargs: Unpack[WeightOptions],
    ) -> NodeOrHandle:
        """Convolve OIW/OIHW weights, or a source layout with an explicit transform."""
        _validate_weight_options(kwargs)
        src_weight_name = kwargs.get("src_weight_name", name + ".weight")
        source = self.parameter(src_weight_name)
        weight = source[1] if isinstance(source, tuple) else source
        if not is_depthwise:
            if weight.ndim not in (3, 4) and not (
                "weight_process_func" in kwargs or "reshape_str" in kwargs
            ):
                raise ValueError(f"{name}: convolution needs OIW/OIHW weights; got {weight.shape}")
            kwargs.setdefault("reshape_str", "oiw->oihw" if weight.ndim == 3 else None)

        def get_param(param_name):
            if param_name == src_weight_name:
                return source
            return self.model.get_hf_param(param_name)

        return self._build_conv(
            name,
            data,
            get_param_func=get_param,
            is_fc=False,
            stride=stride,
            padding=padding,
            is_depthwise=is_depthwise,
            **kwargs,
        )

    def layer_norm(
        self, name: str, data: NodeOrHandle, *, epsilon: float = 1e-5, axis: int = -1
    ) -> NodeOrHandle:
        """Apply LayerNorm along axis with source scale and optional bias."""
        epsilon = float(np.float32(epsilon))
        layer_norm = self.create_layer_norm_node(data, axis, epsilon)

        # Depthwise convolution for element-wise scale and bias.
        ifm_type = tensor_type(layer_norm)
        n_channels = ifm_type.shape[-1]
        weight_shape = (1, 1, 1, n_channels, 1)
        tensor_name = f"{name}.weight"
        if self.model.check_hf_param(tensor_name):
            weight_tensor = load_tensor_from_source(
                tensor_name, self.model.get_hf_param, self.model.check_hf_param
            )
            assert len(weight_tensor.shape) == 1 and weight_tensor.shape[0] == n_channels
        else:
            weight_tensor = np.ones((n_channels,))
        weight_tensor = weight_tensor.reshape(weight_shape).astype(np.float32)

        tensor_name = f"{name}.bias"
        if self.model.check_hf_param(tensor_name):
            bias_tensor = load_tensor_from_source(
                tensor_name, self.model.get_hf_param, self.model.check_hf_param
            )
            assert len(bias_tensor.shape) == 1 and bias_tensor.shape[0] == n_channels
        else:
            bias_tensor = None

        conv_attrs = ConvAttrs(
            stride=(1, 1),
            dilation=(1, 1),
            padding=((0, 0), (0, 0)),
            output_padding=((0, 0), (0, 0)),
            is_transposed=False,
            weight_shape=weight_shape,
            reloc_name=None,
            input_spatial_shape=ifm_type.shape[1:-1],
            batch_size=1,
            input_type=ifm_type.scalar,
        )
        return self.create_conv_node(layer_norm, weight_tensor, bias_tensor, conv_attrs, None)

    def rms_norm(
        self, name: str | None, data: NodeOrHandle, *,
        epsilon: float | None = None, weight_offset: float | None = None,
    ) -> NodeOrHandle:
        """Normalize channel RMS with source weights, or unit weights when name is None.

        Omitted epsilon selects the language configuration's epsilon and weight
        offset. Explicit epsilon uses zero weight offset unless overridden.
        """
        if epsilon is None:
            cfg = getattr(getattr(self.model, "cfg", None), "lm_cfg", None)
            if cfg is None:
                raise ValueError("rms_norm requires epsilon without a language configuration")
            epsilon = cfg.rms_norm_eps
            if weight_offset is None:
                weight_offset = 1.0 if cfg.rms_norm_unit_offset else 0.0
        if weight_offset is None:
            weight_offset = 0.0
        weight = (
            np.ones(tensor_type(data).shape[-1], np.float32)
            if name is None
            else self.parameter(name + ".weight") + weight_offset
        )
        return self.create_rms_norm_node(data, float(np.float32(epsilon)), weight)

    def activation(self, data: NodeOrHandle, name: str) -> NodeOrHandle:
        """Apply a supported named activation using the graph's precision."""
        match name:
            case "silu":
                last = self.create_swish_node(data)
            case "gelu":
                # AFE's GELU node does not support bfloat16, so expand GELU via Erf.
                dtype = self.dtype
                scaled = self.create_mul_node(
                    data,
                    self.create_constant_node(np.array(1 / math.sqrt(2), dtype=dtype)),
                )
                erf = self.create_erf_node(scaled)
                shifted = self.create_add_node(
                    erf, self.create_constant_node(np.array(1.0, dtype=dtype))
                )
                mul = self.create_mul_node(data, shifted)
                last = self.create_mul_node(
                    mul, self.create_constant_node(np.array(0.5, dtype=dtype))
                )
            case "gelu_tanh" | "gelu_pytorch_tanh":
                dtype = self.dtype
                value_a = 2 * math.sqrt(2 / math.pi)
                value_b = 2 * math.sqrt(2 / math.pi) * 0.044715
                const_a = self.create_constant_node(np.array(value_a, dtype=dtype))
                const_b = self.create_constant_node(np.array(value_b, dtype=dtype))
                square = self.create_mul_node(data, data)
                b_mul = self.create_mul_node(square, const_b)
                a_add = self.create_add_node(b_mul, const_a)
                x_mul = self.create_mul_node(data, a_add)
                sig = self.create_sigmoid_node(x_mul)
                last = self.create_mul_node(data, sig)
            case "quick_gelu":
                last = self.create_quick_gelu_node(data)
            case _:
                raise ValueError(f"Unsupported activation: {name}")
        return last

    def softcap(self, data: NodeOrHandle, scalar: float) -> NodeOrHandle:
        """Apply scalar * tanh(data / scalar) in activation precision."""
        dtype = self.dtype
        mul1 = self.create_mul_node(
            data, self.create_constant_node(np.array(2.0 / scalar, dtype=dtype))
        )
        sig = self.create_sigmoid_node(mul1)
        mul2 = self.create_mul_node(
            sig, self.create_constant_node(np.array(2.0 * scalar, dtype=dtype))
        )
        return self.create_add_node(
            mul2, self.create_constant_node(np.array([-scalar], dtype=dtype))
        )

    def _head_padding_options(
        self, output_name: str, num_heads: int, head_dim: int
    ) -> tuple[WeightOptions, WeightOptions]:
        """Return matching Q/K/V and output weight transforms for aligned head channels."""
        if head_dim % mla_row_size == 0:
            return {}, {}
        output_weight = self.parameter(f"{output_name}.weight")
        # Input-channel padding would move grouped quantization boundaries.
        if isinstance(output_weight, tuple) and (
            output_weight[0].size != output_weight[1].shape[0]
            or (len(output_weight) > 2 and output_weight[2] is not None)
        ):
            return {}, {}
        padded_head_dim = round_up_to_row(head_dim)

        def pad_heads(weight: np.ndarray, axis: int = 0) -> np.ndarray:
            """Insert zero channels after each head in weights, scales or bias."""
            shape = weight.shape
            values = weight.reshape(shape[:axis] + (num_heads, head_dim) + shape[axis + 1:])
            padding = [(0, 0)] * values.ndim
            padding[axis + 1] = (0, padded_head_dim - head_dim)
            padded_shape = shape[:axis] + (num_heads * padded_head_dim,) + shape[axis + 1:]
            return np.pad(values, padding).reshape(padded_shape)

        return {
            key: pad_heads
            for key in ("weight_process_func", "scale_process_func", "bias_process_func")
        }, {"weight_process_func": lambda weight: pad_heads(weight, axis=1)}

    def split_heads(self, data: NodeOrHandle, num_heads: int, *, repeat: int = 1) -> NodeOrHandle:
        """Reorder [N,1,T,C] into [N,H,T,C/H], optionally repeating each head."""
        shape = tensor_type(data).shape
        if len(shape) != 4 or shape[1] != 1 or num_heads <= 0 or shape[-1] % num_heads:
            raise ValueError(
                f"split_heads needs [N,1,T,C] with C divisible by {num_heads}; got {shape}"
            )
        if not isinstance(repeat, int) or repeat <= 0:
            raise ValueError(f"split_heads needs a positive integer repeat; got {repeat}")
        if num_heads == 1 and repeat == 1:
            return data
        head_dim = shape[-1] // num_heads
        if head_dim % mla_row_size and tensor_type(data).scalar in (
            ScalarType.float32,
            ScalarType.bfloat16,
        ):
            heads = [
                self.slice(data, start=i * head_dim, stop=(i + 1) * head_dim, axis=3)
                for i in range(num_heads)
            ]
            return self.concat([head for head in heads for _ in range(repeat)], axis=1)
        return self.split_concat(
            data, axis=1, split_axis=3, split_block=num_heads, split_repeat=repeat
        )

    def merge_heads(self, data: NodeOrHandle) -> NodeOrHandle:
        """Reorder [N,H,T,D] into [N,1,T,H*D] by joining head channels per token."""
        shape = tensor_type(data).shape
        if len(shape) != 4:
            raise ValueError(f"merge_heads needs [N,H,T,C]; got {shape}")
        return self.split_concat(data, axis=3, split_axis=1, split_block=shape[1], split_repeat=1)

    def attention(
        self,
        query: NodeOrHandle,
        key: NodeOrHandle,
        value: NodeOrHandle,
        *,
        mask: NodeOrHandle | None = None,
        score_scale: float | None = None,
    ) -> NodeOrHandle:
        """Compute softmax(QK^T + mask)V, optionally scaling scores before the mask."""
        shapes = [tensor_type(node).shape for node in (query, key, value)]
        if all(len(shape) == 4 for shape in shapes):
            heads = max(shape[1] for shape in shapes)
            if mask is not None:
                mask_shape = tensor_type(mask).shape
                score_shape = (shapes[0][0], heads, shapes[0][2], shapes[1][2])
                if (
                    len(mask_shape) > 4
                    or (len(mask_shape) not in (1, 4) and math.prod(mask_shape) != 1)
                    or any(
                        dim not in (1, score_dim)
                        for dim, score_dim in zip(reversed(mask_shape), reversed(score_shape))
                    )
                ):
                    raise ValueError(
                        f"attention mask {mask_shape} must be rank-four, a vector or a scalar "
                        f"and broadcast to score shape {score_shape}"
                    )
            rows = shapes[0][2] * ceil_div_row(shapes[1][2]) * 2
            if (
                heads > 1
                and rows > mla_max_num_rows
                and all(shape[1] in (1, heads) for shape in shapes)
            ):

                def head(node, i):
                    shape = tensor_type(node).shape
                    return (
                        node
                        if len(shape) != 4 or shape[1] == 1
                        else self.slice(node, start=i, stop=i + 1, axis=1)
                    )

                return self.concat(
                    [
                        self.attention(
                            head(query, i),
                            head(key, i),
                            head(value, i),
                            mask=head(mask, i) if mask is not None else None,
                            score_scale=score_scale,
                        )
                        for i in range(heads)
                    ],
                    axis=1,
                )
        scores = self.matmul(query, key, transpose_b=True)
        if score_scale is not None:
            scores = self.mul(scores, self.constant([score_scale]))
        if mask is not None:
            scores = self.add(scores, mask)
        return self.matmul(self.softmax(scores, axis=3), value)

    def rope(
        self,
        data: NodeOrHandle,
        real: NodeOrHandle,
        imag: NodeOrHandle,
        rope_dim: int | None = None,
        *,
        proportional: bool = False,
    ) -> NodeOrHandle:
        """Apply split-half RoPE, preserving channels outside the rotary dimensions."""
        shape = tensor_type(data).shape
        if len(shape) != 4:
            raise ValueError(f"RoPE needs rank-four input; got {shape}")
        channels = shape[-1]
        rope_dim = channels if rope_dim is None else rope_dim
        if rope_dim <= 0 or rope_dim > channels or rope_dim % 2 or (proportional and channels % 2):
            raise ValueError(
                f"RoPE dimension must be positive, even and <= {channels}; got {rope_dim}"
            )
        half = rope_dim // 2
        for frequency in (real, imag):
            spec = tensor_type(frequency)
            target = (*shape[:-1], half)
            if (
                spec.scalar != tensor_type(data).scalar
                or len(spec.shape) != 4
                or any(dim not in (1, expected) for dim, expected in zip(spec.shape, target))
            ):
                raise ValueError(
                    f"RoPE frequencies must broadcast to {target} with the input scalar type; got {spec}"
                )
        start = channels // 2 if proportional else half
        r = self._create_channel_slice(data, 0, half)
        i = self._create_channel_slice(data, start, start + half)
        rout = self.create_subtract_node(
            self.create_mul_node(r, real), self.create_mul_node(i, imag)
        )
        iout = self.create_add_node(self.create_mul_node(r, imag), self.create_mul_node(i, real))
        if proportional and rope_dim < channels:
            parts = [
                rout,
                self._create_channel_slice(data, half, start),
                iout,
                self._create_channel_slice(data, start + half, channels),
            ]
        else:
            parts = [rout, iout]
            if rope_dim < channels:
                # Preserve the original two-step concat for partial, non-proportional RoPE.
                parts = [
                    self.create_concat_node(parts, 3),
                    self._create_channel_slice(data, rope_dim, channels),
                ]
        return self.create_concat_node(parts, 3)

    def rope2d(
        self,
        data: NodeOrHandle,
        cos_x: NodeOrHandle,
        sin_x: NodeOrHandle,
        cos_y: NodeOrHandle,
        sin_y: NodeOrHandle,
    ) -> NodeOrHandle:
        """Rotate channel quarters [x-real, x-imag, y-real, y-imag] independently."""
        spec = tensor_type(data)
        if len(spec.shape) != 4 or spec.shape[-1] % 4:
            raise ValueError(
                f"rope2d needs rank-four input with channels divisible by four; got {spec.shape}"
            )
        quarter = spec.shape[-1] // 4
        target = (*spec.shape[:-1], quarter)
        for table in (cos_x, sin_x, cos_y, sin_y):
            freq = tensor_type(table)
            if (
                freq.scalar != spec.scalar
                or len(freq.shape) != 4
                or any(dim not in (1, expected) for dim, expected in zip(freq.shape, target))
            ):
                raise ValueError(
                    f"rope2d tables must broadcast to {target} with the input scalar; got {freq}"
                )
        xr, xi, yr, yi = [
            self.slice(data, start=i * quarter, stop=(i + 1) * quarter, axis=3) for i in range(4)
        ]
        real_x = self.sub(self.mul(xr, cos_x), self.mul(xi, sin_x))
        imag_x = self.add(self.mul(xr, sin_x), self.mul(xi, cos_x))
        real_y = self.sub(self.mul(yr, cos_y), self.mul(yi, sin_y))
        imag_y = self.add(self.mul(yr, sin_y), self.mul(yi, cos_y))
        return self.concat([real_x, imag_x, real_y, imag_y], axis=3)

    def space_to_depth(self, data: NodeOrHandle, blocksize: int) -> NodeOrHandle:
        """Merge spatial blocks into channels, grouping channels by position within each block."""
        shape = tensor_type(data).shape
        if (
            len(shape) != 4
            or shape[0] != 1
            or not isinstance(blocksize, int)
            or blocksize <= 0
            or shape[1] % blocksize
            or shape[2] % blocksize
        ):
            raise ValueError(
                f"space_to_depth needs [1,H,W,C] with H/W divisible by {blocksize}; got {shape}"
            )
        ifm_type = tensor_type(data)
        in_channels = ifm_type.shape[3]
        out_channels = in_channels * (blocksize**2)

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
        return self.create_conv_node(data, weights, None, conv_attrs)

    def argmax(self, data: NodeOrHandle, *, dtype: ScalarType = ScalarType.int32) -> NodeOrHandle:
        """Select the winning channel, returning INT32 indices by default."""
        return self.create_argmax_node(data, dtype)

    def quant(
        self, data: NodeOrHandle, *, per_token: bool = True
    ) -> tuple[NodeOrHandle, NodeOrHandle]:
        """Dynamically quantize activations to INT8; return values and their scale."""
        scale = self.create_dynamic_quant_scale_node(data, per_token_quant=per_token)
        return self.create_dynamic_quant_node(data, scale), scale

    def dequant(self, data: NodeOrHandle, scale: NodeOrHandle) -> NodeOrHandle:
        """Restore FP32/BF16 activations from INT8 values and their scale."""
        return self.create_dynamic_dequant_node(data, scale)

    def mlp(
        self,
        name: str,
        data: NodeOrHandle,
        activation: str,
        *,
        projections: tuple[str, ...] = ("fc1", "fc2"),
        residual: NodeOrHandle | None = None,
        lora_ranks: dict[str, int | None] | None = None,
        merged_lora: bool = False,
        expert_idx: int = -1,
        de_interleave: bool = False,
        swiglu_limit: float | None = None,
    ) -> NodeOrHandle:
        """Build dense or expert MLPs, optionally using GPT-OSS's clamped SwiGLU."""
        if len(projections) not in (2, 3):
            raise ValueError("MLP needs two projection names or three names in gate/up/down order")
        ranks = lora_ranks or {}

        def project(proj, node, bounds=None):
            rank = ranks.get(proj)
            branch_lora = bool(rank) and not merged_lora
            clip = None
            if bounds is not None and not branch_lora:
                input_type = tensor_type(node)
                channels = self.model.cfg.lm_cfg.get_effective_intermediate_size(self.model.layer_idx)
                clip = ClipAttrs(
                    a_min=bounds[0], a_max=bounds[1],
                    shape=(*input_type.shape[:-1], channels), scalar_type=input_type.scalar,
                )
            output = self.linear(
                f"{name}.{proj}", node, lora_rank=rank, merged_lora=merged_lora,
                expert_idx=expert_idx, de_interleave=de_interleave, activation=clip,
            )
            return self.clip(output, *bounds) if bounds is not None and branch_lora else output

        if swiglu_limit is not None:
            if len(projections) != 3:
                raise ValueError("Clamped SwiGLU needs gate/up/down projections")
            # The gate has no lower clamp; AFE requires a finite BF16 bound.
            gate = project(projections[0], data, (-float.fromhex("0x1.fep127"), swiglu_limit))
            up = project(projections[1], data, (-swiglu_limit, swiglu_limit))
            glu = self.mul(gate, self.sigmoid(self.mul(gate, self.constant([1.702]))))
            hidden = self.mul(self.add(up, self.constant([1.0])), glu)
        else:
            hidden = self.activation(project(projections[0], data), activation)
            if len(projections) == 3:
                hidden = self.mul(hidden, project(projections[1], data))
        output = project(projections[-1], hidden)
        return self.add(residual, output) if residual is not None else output

    def _build_conv(
        self,
        base_name: str,
        ifm: NodeOrHandle,
        *,
        get_param_func: Callable[[str], np.ndarray | tuple] | None = None,
        check_param_func: Callable[[str], bool] | None = None,
        is_fc: bool = True,
        stride: tuple[int, ...] = (1, 1),
        relocatable: bool = False,
        is_depthwise: bool = False,
        padding: tuple[tuple[int, int], ...] = ((0, 0), (0, 0)),
        **kwargs,
    ) -> AwesomeNode:
        """Lower a projection or convolution, allowing a local source for synthesized weights."""
        if get_param_func is None:
            get_param_func = self.model.get_hf_param
        if check_param_func is None:
            check_param_func = self.model.check_hf_param
        assert not (is_fc and is_depthwise), "is_fc and is_depthwise are mutually exclusive"

        ifm_type = tensor_type(ifm)

        potential_weight_name = f"{base_name}.weight"
        potential_bias_name = f"{base_name}.bias"
        reshape_str = kwargs.pop("reshape_str", "oi->oihw" if is_fc else None)
        src_weight_name = kwargs.pop("src_weight_name", potential_weight_name)
        src_bias_name = kwargs.pop("src_bias_name", potential_bias_name)
        has_weight_process_func = "weight_process_func" in kwargs
        has_scale_process_func = "scale_process_func" in kwargs
        weight_process_func = kwargs.pop("weight_process_func", lambda x: x)
        scale_process_func = kwargs.pop("scale_process_func", lambda x: x)
        bias_process_func = kwargs.pop("bias_process_func", lambda x: x)
        q_size = kwargs.pop("q_size", None)
        kv_size = kwargs.pop("kv_size", None)
        activation = kwargs.pop("activation", None)
        expert_idx = kwargs.pop("expert_idx", -1)
        de_interleave = kwargs.pop("de_interleave", False)
        bundled_expert = expert_idx >= 0 and not check_param_func(src_weight_name)
        expert_offset = None

        # Some models have bundled weights with a different name for a layer.
        if bundled_expert:
            projection = src_weight_name.removesuffix(".weight").rsplit(".", 1)[-1]
            prefix = src_weight_name.rsplit(".", 2)[0]
            if projection in ("gate_proj", "up_proj") and de_interleave:
                src_weight_name = f"{prefix}.experts.gate_up_proj"
                expert_offset = 0 if projection == "gate_proj" else 1
            elif projection == "down_proj":
                src_weight_name = f"{prefix}.experts.down_proj"
            else:
                raise NotImplementedError(f"{base_name}: unsupported bundled expert projection")
            if relocatable:
                raise NotImplementedError("Bundled MoE experts require LORA_BRANCH, not LORA_MERGED")
            src_bias_name = f"{src_weight_name}_bias"
        elif not check_param_func(src_weight_name):
            src_weight_name, partition = self._find_alternate_weight(
                src_weight_name, q_size, kv_size
            )
            src_bias_name = src_weight_name.replace("weight", "bias")
            # Select the projection before applying caller transforms, including scaling.
            weight_process_func = lambda x, process=weight_process_func: process(partition(x))
            scale_process_func = lambda x, process=scale_process_func: process(partition(x))
            bias_process_func = lambda x, process=bias_process_func: process(partition(x))

        params = get_param_func(src_weight_name)
        scales, weight_tensor, *metadata = params if isinstance(params, tuple) else (None, params)
        c_block_size = metadata[0] if metadata else None
        if bundled_expert:
            if scales is not None:
                raise ValueError(
                    f"{base_name}: bundled quantized experts need separate output-major weights and scales"
                )
            weight_tensor = weight_tensor[expert_idx].T
            if expert_offset is not None:
                weight_tensor = weight_tensor[expert_offset::2]

        # SiMaIR expects weights in the scales shape (num_c_blocks, out_channels)
        if scales is not None:
            scales = np.reshape(scales, newshape=[weight_tensor.shape[0], -1])
            if scales.shape[1] > 1 and has_weight_process_func and not has_scale_process_func:
                raise ValueError(
                    f"{base_name} has {scales.shape[1]} input-channel scale blocks and a custom "
                    "weight transform, but no scale_process_func was provided. Use per-output-channel "
                    "quantization or provide a matching scale transform."
                )
            scales = scale_process_func(scales)
            scales = np.transpose(scales, axes=[1, 0])

        if is_depthwise:
            # Depthwise convolution: use GHOW intermediate layout.
            if weight_tensor.ndim == 2:
                # GGUF: (G, W) -> (G, H=1, O=1, W)
                weight_tensor = _layout_array(weight_tensor, "gw", "ghow")
            elif weight_tensor.ndim == 3:
                # SafeTensors: (G, H, W) -> (G, H, O=1, W)
                weight_tensor = _layout_array(weight_tensor, "ghw", "ghow")
            weight_tensor = weight_process_func(weight_tensor)
            # Convert to standard HWIGO layout
            weight_tensor = _layout_array(weight_tensor, "ghow", "hwigo")
        else:
            # Standard convolution: use OIHW layout when calling weight_process_func
            if reshape_str:
                src_layout, dst_layout = reshape_str.split("->")
                weight_tensor = _layout_array(weight_tensor, src_layout, dst_layout)
            weight_tensor = weight_process_func(weight_tensor)
            # Convert to SiMa IR layout
            weight_tensor = _layout_array(weight_tensor, "oihw", "hwigo")

        if weight_tensor.dtype in [_bfloat16, np.float16]:
            weight_tensor = weight_tensor.astype(np.float32)  # Model SDK requires float32

        if check_param_func(src_bias_name):
            bias_tensor = get_param_func(src_bias_name)
            if bundled_expert:
                bias_tensor = bias_tensor[expert_idx].astype(np.float32)
                if expert_offset is not None:
                    bias_tensor = bias_tensor[expert_offset::2]
            bias_tensor = bias_process_func(bias_tensor)
            if bias_tensor.dtype in (_bfloat16, np.float16):
                bias_tensor = bias_tensor.astype(np.float32)
        else:
            bias_tensor = None

        conv_attrs = ConvAttrs(
            stride=stride,
            dilation=(1, 1),
            padding=padding,
            output_padding=((0, 0), (0, 0)),
            is_transposed=False,
            weight_shape=weight_tensor.shape,
            reloc_name=src_weight_name if relocatable else None,
            input_spatial_shape=ifm_type.shape[1:-1],
            batch_size=1,
            input_type=ifm_type.scalar,
        )
        conv = self.create_conv_node(
            ifm,
            weight_tensor,
            bias_tensor,
            conv_attrs,
            activation,
            scales=scales,
            c_block_size=c_block_size,
        )
        return conv

    def _create_channel_slice(self, input_node: NodeOrHandle, begin: int, end: int) -> NodeOrHandle:
        """Slice channels with an aligned native slice or a selector convolution."""
        input_type = tensor_type(input_node)
        channel_axis = len(input_type.shape) - 1
        assert 0 <= begin < end <= input_type.shape[channel_axis]

        if begin % 16 == 0 and end % 16 == 0:
            return self.create_slice_node(input_node, [begin], [end], [1], [channel_axis])

        input_channels = input_type.shape[channel_axis]
        output_channels = end - begin
        weights = np.zeros((1, 1, input_channels, 1, output_channels), dtype=np.float32)
        for out_channel in range(output_channels):
            weights[0, 0, begin + out_channel, 0, out_channel] = 1.0

        conv_attrs = ConvAttrs(
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
        return self.create_conv_node(input_node, weights, None, conv_attrs)

    def _build_conv_lora(
        self, base_name: str, ifm: NodeOrHandle, lora_shape: tuple[int, int]
    ) -> AwesomeNode:
        """Create a zero-initialized projection with relocatable LoRA weights."""
        ifm_type = tensor_type(ifm)

        # Convert all-zero LoRA weight to SiMa IR layout.
        weight_tensor = np.zeros(lora_shape, dtype=np.float32)
        weight_tensor = _layout_array(weight_tensor, "oi", "hwigo")

        scales = None
        bias_tensor = None

        # Construct name of LoRA weight in safetensors file.
        lora_base_name = _derive_lora_name_from_base_model(base_name)

        conv_attrs = ConvAttrs(
            stride=(1, 1),
            dilation=(1, 1),
            padding=((0, 0), (0, 0)),
            output_padding=((0, 0), (0, 0)),
            is_transposed=False,
            weight_shape=weight_tensor.shape,
            reloc_name=f"{lora_base_name}.weight",
            input_spatial_shape=ifm_type.shape[1:-1],
            batch_size=1,
            input_type=ifm_type.scalar,
        )
        conv = self.create_conv_node(
            ifm, weight_tensor, bias_tensor, conv_attrs, None, scales=scales
        )
        return conv

    @staticmethod
    def _get_array_partition(
        count: int, index: int, span: int
    ) -> Callable[[np.ndarray | tuple[np.ndarray, np.ndarray]], np.ndarray | tuple[np.ndarray, np.ndarray]]:
        """
        Get a function that divides tensor data into parts along
        axis 0 and returns one of the parts.

        This is a helper for fused projection weights.

        Args:
            count: Number of parts that the data is logically divided into.
            index: Index of the beginning of the part to return.
            span: Number of parts to include in the returned array.
        """
        def slice_array(a: np.ndarray) -> np.ndarray:
            size = a.shape[0]
            assert size % count == 0
            element_size = size // count
            i = element_size * index
            return a[i:i + element_size * span]

        def get(a: np.ndarray | tuple) -> np.ndarray | tuple:
            if isinstance(a, tuple):
                return slice_array(a[0]), slice_array(a[1]), *a[2:]
            return slice_array(a)

        return get

    @staticmethod
    def _find_alternate_weight(
        weight_name: str,
        q_size: int | None,
        kv_size: int | None
    ) -> tuple[str, Callable[[np.ndarray | tuple[np.ndarray, np.ndarray]], np.ndarray | tuple[np.ndarray, np.ndarray]]]:
        """
        Some transformer models, like Microsoft Phi-3.5, have bundled weights.
            - qkv_proj for q_proj, k_proj, and v_proj.
            - gate_up_proj for gate_proj and up_proj.
        If a weight name is not found, change it to the bundled weight name.

        Args:
            weight_name: The original name of a tensor weight, which may or may not exist.

        Returns:
            The alternate weight name and corresponding process function.
        """
        proj_name = weight_name.replace(".weight", "").split(".")[-1]
        if proj_name in ("q_proj", "k_proj", "v_proj"):
            assert q_size is not None
            assert kv_size is not None
            new_weight_name = weight_name.replace(proj_name, "qkv_proj")
        elif proj_name in ("gate_proj", "up_proj"):
            new_weight_name = weight_name.replace(proj_name, "gate_up_proj")
        else:
            raise NotImplementedError(f"{proj_name} not found in weight map.")
        match proj_name:
            case "q_proj":
                parts = q_size + 2 * kv_size
                index = 0
                span = q_size
            case "k_proj":
                parts = q_size + 2 * kv_size
                index = q_size
                span = kv_size
            case "v_proj":
                parts = q_size + 2 * kv_size
                index = q_size + kv_size
                span = kv_size
            case "gate_proj":
                parts = 2
                index = 0
                span = 1
            case "up_proj":
                parts = 2
                index = 1
                span = 1

        new_process_func = ModelGraph._get_array_partition(parts, index, span)

        return new_weight_name, new_process_func


def _derive_lora_name_from_base_model(base_name: str) -> str:
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


def _layout_array(a: np.ndarray, in_layout: str, out_layout: str) -> np.ndarray:
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


def load_tensor_from_source(
    source_name: str,
    get_param_func: Callable[[str], np.ndarray],
    check_param_func: Callable[[str], bool],
    reshape_str: str | None = None,
) -> np.ndarray:
    assert check_param_func(source_name), f"No such tensor: {source_name}"
    t = get_param_func(source_name)
    if reshape_str:
        src_layout, dst_layout = reshape_str.split("->")
        assert len(t.shape) == len(src_layout)
        t = _layout_array(t, src_layout, dst_layout)
    if t.dtype in (_bfloat16, np.float16):
        t = t.astype(np.float32)  # Model SDK requires float32
    return t
