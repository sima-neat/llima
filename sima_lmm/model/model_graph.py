"""Common native graph operations with model weights and precision bound once."""

from typing import TYPE_CHECKING, Callable, Sequence, TypedDict, Unpack

import numpy as np
from afe.ir.attributes import ClipAttrs, ReluAttrs
from afe.ir.build_node import NodeOrHandle, as_handle
from afe.ir.defines import Status, get_expected_tensor_value
from afe.ir.net import AwesomeNet
from afe.ir.serializer import save_awesomenet
from afe.ir.sima_builder import SimaBuilder
from afe.ir.tensor_type import ScalarType, TensorType
from afe.ir.utils import is_mla_supported_einsum_equation, transpose_flags_from_einsum_equation

from sima_lmm.model.sima_builder import (
    activation_dtype,
    build_activation,
    build_conv,
    build_conv_from_dense_with_lora,
    build_matmul_and_split_heads,
    build_merge_heads_and_matmul,
    build_two_stage_layer_norm,
    create_channel_slice,
    create_model_graph,
    finish_model_graph,
)

if TYPE_CHECKING:
    from sima_lmm.model.base import BaseModel


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


def _validate_weight_options(options: dict) -> None:
    unknown = options.keys() - WeightOptions.__annotations__.keys()
    if unknown:
        raise ValueError(
            f"Unknown weight options: {sorted(unknown)}; see WeightOptions for supported overrides"
        )


def tensor_type(node: NodeOrHandle) -> TensorType:
    """Infer the tensor type of an AFE node or handle."""
    return get_expected_tensor_value(as_handle(node).type)


def einsum(
    builder: SimaBuilder, equation: str, lhs: NodeOrHandle, rhs: NodeOrHandle
) -> NodeOrHandle:
    """Lower a supported rank-four contraction directly to MLA BatchMatmul.

    Only the four NHWC matmul equations (and renamed labels) are supported.
    Batch dimensions must match; heads may broadcast from one. Grouped-query
    attention's head repetition is not NumPy einsum semantics: use raw BMM.
    """
    equation = "".join(equation.split())
    try:
        labels = equation.replace("->", ",").split(",")
        supported = all(
            len(set(label)) == 4 and label.isascii() and label.isalpha() for label in labels
        ) and is_mla_supported_einsum_equation(equation, "NHWC")
    except (ValueError, AssertionError, IndexError):
        supported = False
    if not supported:
        raise ValueError(
            f"Unsupported einsum {equation!r}; use rank-four NHWC batch matmul, e.g. nhwc,nhqc->nhwq"
        )
    a, b = tensor_type(lhs), tensor_type(rhs)
    if len(a.shape) != 4 or len(b.shape) != 4:
        raise ValueError(f"einsum {equation!r} needs rank-four inputs; got {a.shape} and {b.shape}")
    if a.scalar != b.scalar or a.scalar not in (ScalarType.float32, ScalarType.bfloat16):
        raise ValueError(
            f"einsum {equation!r} needs matching FP32/BF16 inputs; got {a.scalar} and {b.scalar}"
        )
    ta, tb = transpose_flags_from_einsum_equation(equation)
    if a.shape[0] != b.shape[0] or (a.shape[1] != b.shape[1] and min(a.shape[1], b.shape[1]) != 1):
        raise ValueError(
            f"einsum {equation!r} needs equal batches and equal or singleton heads; got {a.shape} and {b.shape}"
        )
    if a.shape[2 if ta else 3] != b.shape[3 if tb else 2]:
        raise ValueError(
            f"einsum {equation!r} has mismatched contraction dimensions: {a.shape} and {b.shape}"
        )
    return builder.create_batch_matmul_node(lhs, rhs, transpose_a=ta, transpose_b=tb)


def rotary_embedding(
    builder: SimaBuilder,
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
        raise ValueError(f"RoPE dimension must be positive, even and <= {channels}; got {rope_dim}")
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
    r = create_channel_slice(builder, data, 0, half)
    i = create_channel_slice(builder, data, start, start + half)
    rout = builder.create_subtract_node(
        builder.create_mul_node(r, real), builder.create_mul_node(i, imag)
    )
    iout = builder.create_add_node(
        builder.create_mul_node(r, imag), builder.create_mul_node(i, real)
    )
    if proportional and rope_dim < channels:
        parts = [
            rout,
            create_channel_slice(builder, data, half, start),
            iout,
            create_channel_slice(builder, data, start + half, channels),
        ]
    else:
        parts = [rout, iout]
        if rope_dim < channels:
            # Preserve the original two-step concat for partial, non-proportional RoPE.
            parts = [
                builder.create_concat_node(parts, 3),
                create_channel_slice(builder, data, rope_dim, channels),
            ]
    return builder.create_concat_node(parts, 3)


def save_model_graph(model: "BaseModel", net: AwesomeNet, quantizable: bool) -> None:
    """Save a completed graph using the component's standard artifact name."""
    save_awesomenet(
        net,
        model.model_name + (".fp32" if quantizable else ""),
        str(model.sima_model_sdk_path),
    )


class ModelGraph:
    """A single MLA graph with ordered inputs, source weights and inferred types.

    Floating constants use the graph's activation precision. Node names come
    from AFE's deterministic counter. ``raw`` is the unmodified SimaBuilder
    escape hatch for exceptional topology; model tessellation hooks still apply.
    """

    def __init__(
        self,
        model: "BaseModel",
        input_specs: dict[str, tuple[int, ...] | TensorType],
        quantizable: bool,
    ):
        self.model = model
        self.quantizable = quantizable
        self.raw, self.inputs = create_model_graph(input_specs, quantizable)

    @classmethod
    def from_builder(cls, model: "BaseModel", builder: SimaBuilder) -> "ModelGraph":
        """Bind operations to an existing subnet, without beginning another one."""
        graph = cls.__new__(cls)
        graph.model, graph.raw = model, builder
        graph.quantizable = builder.status == Status.RELAY
        graph.inputs = {}
        return graph

    def finish(
        self,
        outputs: Sequence[NodeOrHandle],
        *,
        transform_subnet: Callable[[AwesomeNet], None] | None = None,
    ) -> AwesomeNet:
        return finish_model_graph(
            self.raw, outputs, self.model.model_name, transform_subnet=transform_subnet
        )

    def save(
        self,
        outputs: Sequence[NodeOrHandle],
        *,
        transform_subnet: Callable[[AwesomeNet], None] | None = None,
    ) -> None:
        """Finish and save under the model's configured path and precision suffix."""
        net = self.finish(outputs, transform_subnet=transform_subnet)
        save_model_graph(self.model, net, self.quantizable)

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
                activation_dtype(self.quantizable)
                if data.dtype.kind == "f" or data.dtype == activation_dtype(False)
                else data.dtype
            )
        return self.raw.create_constant_node(data.astype(dtype))

    def linear(
        self,
        name: str,
        data: NodeOrHandle,
        *,
        lora_rank: int | None = None,
        merged_lora: bool = False,
        **kwargs: Unpack[WeightOptions],
    ) -> NodeOrHandle:
        """Project channels with OI weights; preserve bias, scales, blocks and relocation.

        Keyword overrides are the existing build_conv options, including source
        names and matching weight/scale/bias transforms for fused or sliced weights.
        """
        _validate_weight_options(kwargs)
        if len(tensor_type(data).shape) != 4:
            raise ValueError(f"{name}: linear projection needs rank-four NHWC input")
        return build_conv_from_dense_with_lora(
            self.raw,
            self.model.get_hf_param,
            self.model.check_hf_param,
            name,
            data,
            lora_rank,
            merged_lora=merged_lora,
            **kwargs,
        )

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
        """Convolve OIW/OIHW source weights; infer the extra spatial axis for OIW."""
        _validate_weight_options(kwargs)
        source = self.parameter(kwargs.get("src_weight_name", name + ".weight"))
        weight = source[1] if isinstance(source, tuple) else source
        if not is_depthwise:
            if weight.ndim not in (3, 4):
                raise ValueError(f"{name}: convolution needs OIW/OIHW weights; got {weight.shape}")
            kwargs.setdefault("reshape_str", "oiw->oihw" if weight.ndim == 3 else None)
        return build_conv(
            self.raw,
            self.model.get_hf_param,
            self.model.check_hf_param,
            name,
            data,
            is_fc=False,
            stride=stride,
            padding=padding,
            is_depthwise=is_depthwise,
            **kwargs,
        )

    def layer_norm(
        self, name: str, data: NodeOrHandle, *, epsilon: float = 1e-5, axis: int = -1
    ) -> NodeOrHandle:
        return build_two_stage_layer_norm(
            self.raw,
            self.model.get_hf_param,
            self.model.check_hf_param,
            name,
            data,
            axis,
            float(np.float32(epsilon)),
        )

    def rms_norm(
        self, name: str | None, data: NodeOrHandle, *, epsilon: float, weight_offset: float = 0.0
    ) -> NodeOrHandle:
        weight = (
            np.ones(tensor_type(data).shape[-1], np.float32)
            if name is None
            else self.parameter(name + ".weight") + weight_offset
        )
        return self.raw.create_rms_norm_node(data, float(np.float32(epsilon)), weight)

    def activation(self, data: NodeOrHandle, name: str) -> NodeOrHandle:
        return build_activation(self.raw, data, name, self.quantizable)

    def split_heads(self, data: NodeOrHandle, num_heads: int) -> NodeOrHandle:
        shape = tensor_type(data).shape
        if len(shape) != 4 or shape[1] != 1 or num_heads <= 0 or shape[-1] % num_heads:
            raise ValueError(
                f"split_heads needs [N,1,T,C] with C divisible by {num_heads}; got {shape}"
            )
        return self.raw.create_slice_concat_node(
            data, axis=1, split_axis=3, split_block=num_heads, split_repeat=1
        )

    def merge_heads(self, data: NodeOrHandle) -> NodeOrHandle:
        shape = tensor_type(data).shape
        if len(shape) != 4:
            raise ValueError(f"merge_heads needs [N,H,T,C]; got {shape}")
        return self.raw.create_slice_concat_node(
            data, axis=3, split_axis=1, split_block=shape[1], split_repeat=1
        )

    def project_heads(
        self,
        name: str,
        data: NodeOrHandle,
        num_heads: int,
        *,
        scale: float = 1.0,
        kv_len: int | None = None,
        query_len: int | None = None,
    ) -> list[NodeOrHandle]:
        """Project/split heads with existing MLA padding and per-head fallback.

        For cross-attention K/V, pass the query length so all three projections
        make the same choice between packed heads and separate head branches.
        """
        shape = tensor_type(data).shape
        if len(shape) != 4 or num_heads <= 0:
            raise ValueError(
                f"{name}: head projection needs rank-four input and positive heads; got {shape}, {num_heads}"
            )
        query_len = shape[2] if query_len is None else query_len
        if query_len <= 0 or (kv_len is not None and kv_len <= 0):
            raise ValueError(f"{name}: query and KV lengths must be positive")
        return build_matmul_and_split_heads(
            self.raw,
            self.model.get_hf_param,
            self.model.check_hf_param,
            name,
            data,
            num_heads,
            query_len,
            scale,
            kv_len,
        )

    def project_merged_heads(
        self, name: str, heads: list[NodeOrHandle], num_heads: int
    ) -> NodeOrHandle:
        return build_merge_heads_and_matmul(
            self.raw, self.model.get_hf_param, self.model.check_hf_param, name, heads, num_heads
        )

    def einsum(self, equation: str, lhs: NodeOrHandle, rhs: NodeOrHandle) -> NodeOrHandle:
        return einsum(self.raw, equation, lhs, rhs)

    def attention(
        self,
        query: NodeOrHandle,
        key: NodeOrHandle,
        value: NodeOrHandle,
        *,
        mask: NodeOrHandle | None = None,
    ) -> NodeOrHandle:
        """Attend using already-scaled queries and an optional additive mask."""
        scores = self.einsum("nhwc,nhqc->nhwq", query, key)
        if mask is not None:
            scores = self.raw.create_add_node(scores, mask)
        return self.einsum("nhwc,nhcq->nhwq", self.raw.create_softmax_node(scores, axis=3), value)

    def rope(
        self,
        data: NodeOrHandle,
        real: NodeOrHandle,
        imag: NodeOrHandle,
        rope_dim: int | None = None,
        *,
        proportional: bool = False,
    ) -> NodeOrHandle:
        return rotary_embedding(self.raw, data, real, imag, rope_dim, proportional=proportional)

    def quantize(
        self, data: NodeOrHandle, *, per_token: bool = True
    ) -> tuple[NodeOrHandle, NodeOrHandle]:
        scale = self.raw.create_dynamic_quant_scale_node(data, per_token_quant=per_token)
        return self.raw.create_dynamic_quant_node(data, scale), scale

    def dequantize(self, data: NodeOrHandle, scale: NodeOrHandle) -> NodeOrHandle:
        return self.raw.create_dynamic_dequant_node(data, scale)

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
    ) -> NodeOrHandle:
        """Build two-projection or gated (gate, up, down) MLPs."""
        if len(projections) not in (2, 3):
            raise ValueError("MLP needs two projection names or three names in gate/up/down order")
        ranks = lora_ranks or {}

        def project(proj, node):
            return self.linear(
                f"{name}.{proj}", node, lora_rank=ranks.get(proj), merged_lora=merged_lora
            )

        hidden = self.activation(project(projections[0], data), activation)
        if len(projections) == 3:
            hidden = self.raw.create_mul_node(hidden, project(projections[1], data))
        output = project(projections[-1], hidden)
        return self.raw.create_add_node(residual, output) if residual is not None else output
