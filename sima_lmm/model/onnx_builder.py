import math
import numpy as np
import onnx
import onnx.numpy_helper

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from ml_dtypes import bfloat16
from onnx.shape_inference import infer_shapes_path
from onnxsim.onnxsim_cpp2py_export import simplify_path
from pathlib import Path
from typing import ClassVar

from sima_utils.logging.sima_logger import sima_log_dbg
from sima_lmm.utils import (
    ceil_div_row, mla_max_num_rows, mla_row_size, round_up_to_row
)


OnnxNode = onnx.ValueInfoProto | onnx.TensorProto | onnx.NodeProto


@dataclass
class OnnxBuilder:
    """Helper class to build onnx model.

    Attributes:
        IR_VERSION: IR version of the onnx model.
        OPSET_ID: Operator set id of the onnx model.
        onnx_file_name: File name of the onnx file.
        get_param_func: A function that returns a parameter tensor with the provided parameter name.
        check_param_func: A function that checks if a parameter tensor with the provided parameter name exists.
        input_nodes: Input nodes of the onnx model.
        output_nodes: Output nodes of the onnx model.
        _initializer_map: A mapping from a name to an onnx initializer.
        _node_map: A mapping from a name to an onnx node.
    """
    IR_VERSION: ClassVar[int] = 8
    OPSET_ID: ClassVar[int] = 17

    onnx_file_name: Path
    get_param_func: Callable[[str], np.ndarray] | None = field(default=None, kw_only=True)
    check_param_func: Callable[[str], bool] | None = field(default=None, kw_only=True)

    input_nodes: list[onnx.ValueInfoProto] = field(init=False, default_factory=list)
    output_nodes: list[onnx.ValueInfoProto] = field(init=False, default_factory=list)
    _initializer_map: dict[str, onnx.TensorProto] = field(init=False, default_factory=dict)
    _node_map: dict[str, onnx.NodeProto] = field(init=False, default_factory=dict)

    def create_and_save_model(self, do_simplify: bool = True):
        """Creates and saves the model.

        Args:
            do_simplify: Set true to simplify the created onnx graph.
        """
        model = self.create_model()
        self.save_model(model)

        # Infer shapes.
        infer_shapes_path(str(self.onnx_file_name), str(self.onnx_file_name), True, True, True)

        # Currently both onnxsim and its base onnxoptimizer do not work with the protobuf 4.xx.
        # Disable simplify now until the issue is resolved.
        if do_simplify and False:
            # Use simplify_path instead of simplify because simplify function does not work
            # properly with models larger than 2GB and protobuf version 4.xx.
            simplify_path(
                str(self.onnx_file_name), str(self.onnx_file_name), None, True, True,
                int(1.5 * 2**30)
            )
        sima_log_dbg(f"Created and saved {self.onnx_file_name}.")

    def create_model(self) -> onnx.ModelProto:
        """Creates the model and performs shape inference.
        """
        graph = onnx.helper.make_graph(
            name=self.onnx_file_name.name,
            nodes=list(self._node_map.values()),
            inputs=self.input_nodes,
            outputs=self.output_nodes,
            initializer=list(self._initializer_map.values())
        )
        model = onnx.helper.make_model(
            graph,
            ir_version=self.IR_VERSION,
            opset_imports=[onnx.helper.make_opsetid("", self.OPSET_ID)]
        )
        return model

    def save_model(self, model: onnx.ModelProto):
        """Saves the model to a file.

        Args:
            model: The onnx model to be saved.
        """
        assert self.onnx_file_name.suffix == ".onnx"
        self.onnx_file_name.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing external data file to prevent tensors from being appended to old content.
        external_data = self.onnx_file_name.with_name(f"{self.onnx_file_name.name}.ext_data")
        external_data.unlink(missing_ok=True)

        # Use string instead of Path because onnx save does not handle external data file names
        # correctly with Path.
        onnx.save(
            model, str(self.onnx_file_name), save_as_external_data=True, location=external_data.name
        )

    def create_input_node(self, name: str, shape: Sequence[int], dtype: type = np.float32):
        """Creates an input node with the provided name, shape and data type."""
        node = self._create_io(name, shape, dtype)
        self.input_nodes.append(node)

    def create_output_node(self, name: str, shape: Sequence[int], dtype: type = np.float32):
        """Creates an output node with the provided name, shape and data type."""
        node = self._create_io(name, shape, dtype)
        self.output_nodes.append(node)

    def get_node_output_names(self, node: OnnxNode) -> list[str]:
        """Gets a list of the output names of the given node."""
        if isinstance(node, (onnx.ValueInfoProto, onnx.TensorProto)):
            return [node.name]
        else:
            assert isinstance(node, onnx.NodeProto), type(node)
            return node.output

    def get_node_output_name(self, node: OnnxNode) -> str:
        """Gets the output name of the given node with only one output."""
        output_names = self.get_node_output_names(node)
        assert len(output_names) == 1
        return output_names[0]

    def create_initializer(
        self, name: str, value: int | float | np.ndarray | None = None,
        reshape_str: str | None = None
    ) -> OnnxNode | None:
        """Creates an initializer with the name.

        Args:
            name: Initializer name.
            value: Value of the initializer. If value is None, then look up the value using
                get_param_func; if get_param_func is None, then look up the value using the
                pre-defined file name.
            reshape_str: A string to reshape the value.
        Returns:
            Return an initializer if a valid value is found. Otherwise, return None.
        """
        if name in self._initializer_map:
            assert value is None, f"{name}: {self._initializer_map[name]}"
        elif value is None:
            assert self.get_param_func is not None
            # Load the value using the provided get_param_func.
            try:
                value = self.get_param_func(name)
            except AssertionError:
                pass
            if value is None:
                return None
            value = self.reshape_data(value, reshape_str)
            self._initializer_map[name] = onnx.numpy_helper.from_array(value, name=name)
        elif isinstance(value, int):
            self._initializer_map[name] = onnx.helper.make_tensor(
                name=name, data_type=onnx.TensorProto.INT32, dims=[1], vals=[value]
            )
        elif isinstance(value, float):
            self._initializer_map[name] = onnx.helper.make_tensor(
                name=name, data_type=onnx.TensorProto.FLOAT, dims=[1], vals=[value]
            )
        elif isinstance(value, np.ndarray):
            value = self.reshape_data(value, reshape_str)
            self._initializer_map[name] = onnx.numpy_helper.from_array(value, name=name)
        else:
            raise RuntimeError(f"value type not recognized to create initializer: {type(value)}")
        return self._initializer_map[name]

    def reshape_data(self, data: np.ndarray, reshape_str: str | None = None) -> np.ndarray:
        match reshape_str:
            case "c->nchw":
                assert data.ndim == 1
                data = data.reshape(1, -1, 1, 1)
            case "nc->nchw":
                assert data.ndim == 2
                data = data.reshape(*data.shape, 1, 1)
            case "cn->nchw":
                assert data.ndim == 2
                data = data.reshape(*data.shape, 1, 1).transpose(1, 0, 2, 3)
            case "wc->nchw":
                assert data.ndim == 2
                shape = data.shape
                data = data.transpose(1, 0).reshape(1, shape[1], 1, shape[0])
            case "ncw->nchw":
                assert data.ndim == 3, data.ndim
                n, c, w = data.shape
                data = data.reshape(n, c, 1, w)

        # Temporary workaround because TVM cannot handle the np.float16 weights with np.float32
        # inputs.
        if data.dtype == np.float16 or data.dtype == bfloat16:
            data = data.astype(np.float32)
        return data

    def build_op(
        self, base_name: str, input_nodes: Sequence[OnnxNode], op_type: str, **kwargs
    ) -> OnnxNode:
        """Builds an ONNX node.

        Args:
            base_name: Base name of the operator. This is used to create the node and the
                initializer.
            input_nodes: A list of input nodes.
            op_type: Name of the operator type.
            **kwargs: Operator attributes.
        Returns:
            Created ONNX node.
        """
        updated_input_node_names = list()
        for idx, input_node in enumerate(input_nodes):
            match input_node:
                case int() | float() | np.ndarray():
                    node = self.create_initializer(f"{base_name}.const{idx}", input_node)
                    updated_input_node_names.append(self.get_node_output_name(node))
                case str():
                    node = self.create_initializer(input_node)
                    updated_input_node_names.append(self.get_node_output_name(node))
                case list():
                    assert len(input_node) == 2, input_node
                    assert isinstance(input_node[0], OnnxNode), type(input_node[0])
                    assert isinstance(input_node[1], int), type(input_node[1])
                    node = input_node[0]
                    output_idx = input_node[1]
                    updated_input_node_names.append(self.get_node_output_names(node)[output_idx])
                case _:
                    assert isinstance(input_node, OnnxNode), type(input_node)
                    node = input_node
                    updated_input_node_names.append(self.get_node_output_name(node))

        output_names = kwargs.pop("output_names", [f"{base_name}_output"])
        op = self._create_node(
            name=base_name,
            op_type=op_type,
            inputs=updated_input_node_names,
            outputs=output_names,
            **kwargs
        )
        return op

    def build_conv(
        self, base_name: str, input_node: OnnxNode, is_fc: bool = True,
        weight_slice: tuple[int, int, int, int] | None = None, **kwargs
    ) -> OnnxNode:
        """Builds a convolution node.

        Args:
            base_name: Base name of the operator. This is used to create the node and the
                initializers.
            input_node: The input node of the convolution node.
            is_fc: Set True to indicate that the original operator is a fully-connected layer or a
                matrix multiplication where the weight needs to be reshaped to build the
                convolution.
            weight_slice: Optional specification of a sub-region of the weight tensor. Includes
                size, offset, axis and index of weight slice. If None, use the entire weight.
            **kwargs: Convolution attributes.
        Returns:
            Created convolution node.
        """
        potential_weight_name = f"{base_name}.weight"
        potential_bias_name = f"{base_name}.bias"
        reshape_str = kwargs.pop("reshape_str", "nc->nchw" if is_fc else None)
        src_weight_name = kwargs.pop("src_weight_name", potential_weight_name)
        src_bias_name = kwargs.pop("src_bias_name", potential_bias_name)
        weight_process_func = kwargs.pop("weight_process_func", lambda x: x)
        bias_process_func = kwargs.pop("bias_process_func", lambda x: x)
        q_size = kwargs.pop("q_size", None)
        kv_size = kwargs.pop("kv_size", None)

        # Some models have bundled weights with a different name for a layer.
        if not self.check_param_func(src_weight_name):
            src_weight_name, weight_process_func = find_alternate_weight(
                self.get_param_func, src_weight_name, q_size, kv_size
            )
            src_bias_name = src_weight_name.replace("weight", "bias")
            bias_process_func = weight_process_func

        weight_tensor = self.reshape_data(self.get_param_func(src_weight_name), reshape_str)
        weight_tensor = weight_process_func(weight_tensor)

        # Take part of weight tensor if slice is defined.
        slice_idx = ""
        if weight_slice is not None:
            start, size, axis, idx = weight_slice
            slice_idx = f".{idx}"
            # Normalize axis (e.g., -1 means last axis)
            axis = np.core.numeric.normalize_axis_index(axis, weight_tensor.ndim)
            weight_tensor = np.take(weight_tensor, np.arange(start, start + size), axis=axis)

        assert isinstance(weight_tensor, np.ndarray)  # Only unquantized weights are handled here
        weight = self.create_initializer(f"{base_name}.weight{slice_idx}", weight_tensor)
        conv_inputs = [self.get_node_output_name(input_node), self.get_node_output_name(weight)]

        if self.check_param_func(src_bias_name):
            bias_tensor = self.get_param_func(src_bias_name)
            bias_tensor = bias_process_func(bias_tensor)
            bias = self.create_initializer(f"{base_name}.bias", bias_tensor)
            conv_inputs.append(self.get_node_output_name(bias))

        conv = self._create_node(
            name=f"{base_name}{slice_idx}",
            op_type="Conv",
            inputs=conv_inputs,
            outputs=[f"{base_name}{slice_idx}_output"],
            **kwargs
        )
        return conv

    def _get_dense_shape_from_conv(self, conv_node: OnnxNode) -> tuple[int, int]:
        """Get the original shape of dense matrix from the converted conv node.
        """
        assert conv_node.op_type == "Conv", f"Expected Conv, but got {conv_node.op_type}"
        name = conv_node.input[1]
        assert name in self._initializer_map, f"{name} is not yet initialized."
        weight_tensor = onnx.numpy_helper.to_array(self._initializer_map[name])
        return weight_tensor.shape[0:2]

    def _build_conv_lora(
        self, base_name: str, input_node: OnnxNode, lora_shape: tuple[int, int], **kwargs
    ) -> OnnxNode:
        """Builds a convolution node for LoRA matrix multiply.

        Args:
            base_name: Base name of the operator. This is used to create the node and the
                initializers.
            input_node: The input node of the convolution node.
            lora_shape: The shape of LoRA matrix.
            **kwargs: Convolution attributes.
        Returns:
            Created convolution node.
        """
        weight_tensor = self.reshape_data(np.zeros(lora_shape, dtype=np.float32), "nc->nchw")
        weight = self.create_initializer(f"{base_name}.weight", weight_tensor)
        conv_inputs = [self.get_node_output_name(input_node), self.get_node_output_name(weight)]

        conv = self._create_node(
            name=base_name,
            op_type="Conv",
            inputs=conv_inputs,
            outputs=[f"{base_name}_output"],
            **kwargs
        )
        return conv

    def build_conv_from_dense_with_lora(
        self, base_name: str, input_node: OnnxNode, lora_rank: int | None = None,
        weight_slice: tuple[int, int, int, int] | None = None, **kwargs
    ) -> OnnxNode:
        """Builds a conv from dense op with LoRA branch.

        Args:
            base_name: The name of the layer in the base model.
            input_node: The input node.
            lora_rank: The rank of LoRA adapter.
            weight_slice: Optional specification of a sub-region of the weight tensor. Includes
                size, offset, axis and index of weight slice. If None, use the entire weight.
        Returns:
            Created conv node or merged node of a conv and LoRA branch.
        """
        if weight_slice is not None and lora_rank is not None:
            raise NotImplementedError("Conv slicing is not supported for LoRA.")

        # The dense node in the base model.
        proj = self.build_conv(base_name, input_node, weight_slice=weight_slice, **kwargs)

        # The LoRA branch.
        if lora_rank is not None:
            output_channels, input_channels = self._get_dense_shape_from_conv(proj)
            a_shape = (lora_rank, input_channels)
            b_shape = (output_channels, lora_rank)

            lora_a = self._build_conv_lora(
                f"{base_name}.lora_A", input_node, a_shape
            )
            lora_b = self._build_conv_lora(
                f"{base_name}.lora_B", lora_a, b_shape
            )
            proj = self.build_op(
                f"{base_name}_lora", [proj, lora_b], "Add"
            )
        return proj

    def build_split_and_concat(
        self, base_name: str, input_node: OnnxNode, num_splits: int, split_axis: int,
        concat_axis: int | None
    ) -> OnnxNode:
        """Builds nodes for a split-and-concat operation.

        Args:
            base_name: Base name of the operator. This is used to create the nodes.
            input_node: The input node of the split node.
            num_splits: Number of splits.
            split_axis: The axis for the input node to be split.
            concat_axis: The axis for the split nodes to be concatenated.
        Returns:
            Created split and concatenate nodes.
        """
        if num_splits == 1:
            return input_node

        node_name = f"{base_name}.split"
        split = self._create_node(
            name=node_name,
            op_type="Split",
            inputs=[self.get_node_output_name(input_node)],
            outputs=[f"{node_name}_output.{i}" for i in range(num_splits)],
            axis=split_axis
        )

        if concat_axis is None:
            return split

        node_name = f"{base_name}.concat"
        concat = self._create_node(
            name=node_name,
            op_type="Concat",
            inputs=self.get_node_output_names(split),
            outputs=[f"{node_name}_output"],
            axis=concat_axis
        )
        return concat

    def build_split_expand_concat(
        self, base_name: str, input_node: OnnxNode, num_splits: int, num_repeats: int,
        split_axis: int, concat_axis: int, concat_shape: tuple[int, ...]
    ) -> OnnxNode:
        """Builds nodes for a split-expand-concat operation.

        This is to support Group Quary Attention (GQA), where the number of KV heads
        is less than the number of attention heads. The KV tensors out of a KV cache
        need to be repeated to match attention heads.

        Args:
            base_name: Base name of the operator. This is used to create the nodes.
            input_node: The input node of the split node.
            num_splits: Number of splits.
            num_repeats: Number of repeats for each split.
            split_axis: The axis for the input node to be split.
            concat_axis: The axis for the split nodes to be concatenated.
        Returns:
            Created split, expand, and concatenate nodes.
        """
        if num_splits == 1:
            if num_repeats == 1:
                return input_node
            else:
                return self.build_op(
                    f"{base_name}.expand",
                    [input_node, np.array(concat_shape, dtype=np.int64)],
                    "Expand"
                )
        elif split_axis == concat_axis and num_repeats == 1:
            return input_node

        node_name = f"{base_name}.split"
        split = self._create_node(
            name=node_name,
            op_type="Split",
            inputs=[self.get_node_output_name(input_node)],
            outputs=[f"{node_name}_output.{i}" for i in range(num_splits)],
            axis=split_axis
        )

        inputs_to_concat = list()
        for split_output in self.get_node_output_names(split):
            inputs_to_concat += [split_output] * num_repeats

        node_name = f"{base_name}.concat"
        concat = self._create_node(
            name=node_name,
            op_type="Concat",
            inputs=inputs_to_concat,
            outputs=[f"{node_name}_output"],
            axis=concat_axis
        )
        return concat

    def build_layer_norm(
        self, base_name: str, input_node: OnnxNode, epsilon: float = 1e-5
    ) -> OnnxNode:
        """Builds nodes for layer norm operation.

        Args:
            base_name: Base name of the operator. This is used to create the nodes.
            input_node: The input node of the split node.
        Returns:
            Created layer norm nodes.
        """
        node_name = f"{base_name}.pre_transpose"
        pre_transpose = self._create_node(
            name=node_name,
            op_type="Transpose",
            inputs=self.get_node_output_names(input_node),
            outputs=[f"{node_name}_output"],
            perm=[0, 2, 3, 1]
        )

        node_name = f"{base_name}.layer_norm"
        layer_norm = self._create_node(
            name=node_name,
            op_type="LayerNormalization",
            inputs=[
                self.get_node_output_name(pre_transpose),
                self.get_node_output_name(self.create_initializer(f"{base_name}.weight")),
                self.get_node_output_name(self.create_initializer(f"{base_name}.bias"))
            ],
            outputs=[f"{node_name}_output"],
            epsilon=epsilon
        )

        node_name=f"{base_name}.post_transpose"
        post_transpose = self._create_node(
            name=node_name,
            op_type="Transpose",
            inputs=self.get_node_output_names(layer_norm),
            outputs=[f"{node_name}_output"],
            perm=[0, 3, 1, 2]
        )
        return post_transpose

    def build_rms_norm(
        self, base_name: str, input_node: OnnxNode, epsilon: float, weight_offset: float = 0.0,
        weightless: bool = False, num_channels: int = 0,
    ) -> OnnxNode:
        """Builds nodes for RMS norm operation.

        Args:
            base_name: Base name of the operator.
            input_node: Input node.
            epsilon: RMS norm epsilon.
            weight_offset: Offset added to the loaded weight. Ignored when weightless=True.
            weightless: If True, use an all-ones weight instead of loading from the HF param
                store. Requires num_channels > 0. The all-ones weight keeps the RMS norm
                mathematically identity-scaled while still producing the full
                Mul→ReduceMean→Add→Sqrt→Div→Mul(weight) graph pattern that AFE fuses into a
                single kernel (avoiding bf16 precision loss in the un-fused fallback).
            num_channels: Number of channels for the all-ones weight. Required when weightless=True.
        """
        mul1 = self.build_op(f"{base_name}.mul1", [input_node, input_node], "Mul")
        mean = self.build_op(f"{base_name}.mean", [mul1], "ReduceMean", axes=[1], keepdims=1)
        add = self.build_op(f"{base_name}.add", [mean, epsilon], "Add")
        sqrt = self.build_op(f"{base_name}.sqrt", [add], "Sqrt")
        div = self.build_op(f"{base_name}.div", [input_node, sqrt], "Div")
        if weightless:
            if num_channels <= 0:
                raise ValueError("num_channels must be > 0 when weightless=True")
            weight_tensor = np.ones(num_channels, dtype=np.float32)
        else:
            weight_tensor = self.get_param_func(f"{base_name}.weight") + weight_offset
        weight = self.create_initializer(
            f"{base_name}.weight", value=weight_tensor, reshape_str="c->nchw"
        )
        mul2 = self.build_op(f"{base_name}.mul2", [div, weight], "Mul")
        return mul2
    
    def build_logit_softcapping(self, base_name: str, input_node: OnnxNode, scalar: float) -> OnnxNode:
        """Build nodes for logit soft capping.

        Logit soft capping is used in GEMMA2 to prevent overconfident predictions.

            softcapping(x) = scalar * tanh(x/scalar)
                           = scalar * [2*sigmoid(2x/scalar)-1]
                           = (2*scalar) * sigmoid(x * (2/scalar)) - scalar
            operations: x - mul - sigmoid - mul - sub
        """
        mul1 = self.build_op(f"{base_name}.mul1", [input_node, 2.0/scalar], "Mul")
        sig = self.build_op(f"{base_name}.sigmoid", [mul1], "Sigmoid")
        mul2 = self.build_op(f"{base_name}.mul2", [sig, 2.0*scalar], "Mul")
        last = self.build_op(f"{base_name}.sub", [mul2, -scalar], "Add")
        return last

    def build_activation(self, base_name: str, input_node: OnnxNode, act_type: str) -> OnnxNode:
        """Build nodes for activation.

        LLAMA uses "silu" which uses sigmoid.
        GEMMA uses "gelu_pytorch_tanh" which uses Gaussian ERF with tanh approximation.

        Because tanh(x) = 2*sigmoid(2x)-1, gelu can also be approximated by sigmoid.
            gelu_tanh(x) = 0.5 * x * [1 + tanh(root(2/PI)*x*(1 + 0.044715 * x * x))]
                         = x * sigmoid(x*(A + B * x * x))
            where A = 2 * root(2/PI), B = A * 0.044715
        """
        match act_type:
            case "silu":
                sig = self.build_op(f"{base_name}.sigmoid", [input_node], "Sigmoid")
                last = self.build_op(f"{base_name}.mul1", [input_node, sig], "Mul")
            case "gelu":
                # gelu(x) = (x / 2) * (1 + erf(x / sqrt(2)))
                div = self.build_op(f"{base_name}.div", [input_node, np.sqrt(2)], "Div")
                erf = self.build_op(f"{base_name}.erf", [div], "Erf")
                add = self.build_op(f"{base_name}.add", [erf, 1.0], "Add")
                mul1 = self.build_op(f"{base_name}.mul1", [input_node, add], "Mul")
                last = self.build_op(f"{base_name}.mul2", [mul1, 0.5], "Mul")
            case "gelu_tanh" | "gelu_pytorch_tanh":
                value_a = 2 * math.sqrt(2 / math.pi)
                value_b = 2 * math.sqrt(2 / math.pi) * 0.044715
                const_a = self.create_initializer(f"{base_name}.const_a", value=value_a)
                square = self.build_op(f"{base_name}.mul1", [input_node, input_node], "Mul")
                const_b = self.create_initializer(f"{base_name}.const_b", value=value_b)
                b_mul = self.build_op(f"{base_name}.mul2", [square, const_b], "Mul")
                a_add = self.build_op(f"{base_name}.add1", [b_mul, const_a], "Add")
                x_mul = self.build_op(f"{base_name}.mul3", [input_node, a_add], "Mul")
                sig = self.build_op(f"{base_name}.sigmoid", [x_mul], "Sigmoid")
                last = self.build_op(f"{base_name}.mul4", [input_node, sig], "Mul")
            case "quick_gelu":
                scaled = self.build_op(f"{base_name}.mul1", [input_node, 1.702], "Mul")
                sigmoid = self.build_op(f"{base_name}.sigmoid", [scaled], "Sigmoid")
                last = self.build_op(f"{base_name}.mul2", [input_node, sigmoid], "Mul")
            case _:
                raise ValueError(f"Unsupported activation: {act_type}")
        return last

    def build_matmul_and_split_heads(
        self, base_name: str, input_node: OnnxNode, num_heads: int, seq_len: int,
        kv_len: int | None = None, post_matmul_scale: float = 1.0
    ) -> list[OnnxNode]:
        if kv_len is None:
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
            matmul = self.build_conv(
                f"{base_name}.matmul", input_node, weight_process_func=param_process_func,
                bias_process_func=param_process_func, src_weight_name=f"{base_name}.weight",
                src_bias_name=f"{base_name}.bias"
            )
            reshape = self.build_split_and_concat(f"{base_name}.reshape", matmul, num_heads, 1, 2)
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
                matmul = self.build_conv(
                    f"{base_name}.matmul.{i}", input_node, weight_process_func=param_process_func,
                    bias_process_func=param_process_func, src_weight_name=f"{base_name}.weight",
                    src_bias_name=f"{base_name}.bias"
                )
                heads.append(matmul)
            return heads

    def build_merge_heads_and_matmul(
        self, base_name: str, input_nodes: list[OnnxNode], num_heads: int
    ) -> OnnxNode:
        if len(input_nodes) == 1:
            matmul_input = input_nodes[0]
        else:
            assert len(input_nodes) == num_heads
            # Heads are not merged. Concatenate first.
            matmul_input = self.build_op(f"{base_name}.concat", input_nodes, "Concat", axis=2)

        reshape = self.build_split_and_concat(f"{base_name}.reshape", matmul_input, num_heads, 2, 1)

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
        return self.build_conv(base_name, reshape, weight_process_func=param_process_func)

        # def param_process_func(x: np.ndarray) -> np.ndarray:
        #     assert x.shape[1] % num_heads == 0
        #     assert x.shape[2] == x.shape[3] == 1
        #     head_dim = x.shape[1] // num_heads
        #     x = x.reshape(x.shape[0], num_heads, head_dim, 1).transpose(0, 2, 1, 3)
        #     if len(input_nodes) == 1 and head_dim % mla_row_size:
        #         rounded_head_dim = round_up_to_row(head_dim)
        #         x = np.pad(x, ((0, 0), (0, rounded_head_dim - head_dim), (0, 0), (0, 0)))
        #     return x

        # return self.build_conv(base_name, matmul_input, weight_process_func=param_process_func)

    def build_attention(
        self,
        base_name: str,
        input_nodes: list[OnnxNode],
        num_heads: int,
        head_dim: int,
        seq_len: int,
        kv_len: int | None = None,
        skip_kv_projs_and_split_head: bool = False,
        mask_node: OnnxNode | None = None,
        output_kv_projs: bool = False,
    ) -> list[OnnxNode]:
        if len(input_nodes) == 1:
            input_nodes = input_nodes * 3

        scaled_q_projs = self.build_matmul_and_split_heads(
            f"{base_name}.q_proj", input_nodes[0], num_heads, seq_len, kv_len,
            post_matmul_scale=head_dim ** -0.5
        )
        if skip_kv_projs_and_split_head:
            if len(scaled_q_projs) == 1:
                k_projs = [input_nodes[1]]
                v_projs = [input_nodes[2]]
            else:
                k_split = self.build_op(
                    f"{base_name}.k.split", [input_nodes[1]], "Split", axis=2,
                    output_names=[f"{base_name}.k.split_output.{i}" for i in range(num_heads)]
                )
                v_split = self.build_op(
                    f"{base_name}.v.split", [input_nodes[2]], "Split", axis=2,
                    output_names=[f"{base_name}.v.split_output.{i}" for i in range(num_heads)]
                )
                k_projs = [(k_split, i) for i in range(num_heads)]
                v_projs = [(v_split, i) for i in range(num_heads)]
        else:
            k_projs = self.build_matmul_and_split_heads(
                f"{base_name}.k_proj", input_nodes[1], num_heads, seq_len, kv_len
            )
            v_projs = self.build_matmul_and_split_heads(
                f"{base_name}.v_proj", input_nodes[2], num_heads, seq_len, kv_len
            )

        attn_outputs = list()
        for i, scaled_q_proj, k_proj, v_proj in zip(
            range(num_heads), scaled_q_projs, k_projs, v_projs
        ):
            attn_weights = self.build_op(
                f"{base_name}.attn_weights.{i}", [scaled_q_proj, k_proj], "Einsum",
                equation="nchw,nchq->nqhw"
            )
            if mask_node is not None:
                attn_weights = self.build_op(
                    f"{base_name}.masked_attn_weights.{i}", [attn_weights, mask_node], "Add"
                )
            softmax = self.build_op(f"{base_name}.softmax.{i}", [attn_weights], "Softmax", axis=1)
            attn_outputs.append(
                self.build_op(
                    f"{base_name}.attn_output.{i}", [softmax, v_proj], "Einsum",
                    equation="nchw,nqhc->nqhw"
                )
            )
        o_proj = self.build_merge_heads_and_matmul(
            f"{base_name}.out_proj", attn_outputs, num_heads
        )
        if output_kv_projs:
            # Currently split head is not supported when kv projs need to be returned.
            assert len(k_projs) == len(v_projs) == 1
            return [o_proj, k_projs[0], v_projs[0]]
        else:
            return [o_proj]

    def build_encoder_decoder_mlp(
        self, base_name: str, input_node: OnnxNode, act_type: str
    ) -> OnnxNode:
        fc1 = self.build_conv(f"{base_name}.fc1", input_node)
        act = self.build_activation(f"{base_name}.{act_type}", fc1, act_type)
        fc2 = self.build_conv(f"{base_name}.fc2", act)
        return fc2

    def _create_io(self, name: str, shape: Sequence[int], dtype: type) -> onnx.ValueInfoProto:
        """Creates an input or output node based on the provided name, shape and data type.

        Args:
            name: Name of the input and output node.
            shape: Shape of the node.
            dtype: Data type of the node.
        Return:
            Created input or output node.
        """
        match dtype:
            case np.float32:
                onnx_dtype = onnx.TensorProto.FLOAT
            case np.int64:
                onnx_dtype = onnx.TensorProto.INT64
            case _:
                raise RuntimeError(f"Conversion from dtype={dtype} to onnx type is not supported")
        return onnx.helper.make_tensor_value_info(name=name, elem_type=onnx_dtype, shape=shape)

    def _create_node(self, **kwargs) -> onnx.NodeProto:
        """Creates a node based on the provided arguments."""
        node = onnx.helper.make_node(**kwargs)
        self._node_map[node.name] = node
        return node


def _get_array_partition(
    count: int, index: int, span: int
) -> Callable[[np.ndarray | tuple[np.ndarray, np.ndarray]], np.ndarray | tuple[np.ndarray, np.ndarray]]:
    """
    Get a function that divides tensor data into parts along
    axis 0 and returns one of the parts.

    This is a helper function for find_alternate_weight.

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

    def get(a: np.ndarray | tuple[np.ndarray, np.ndarray]) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if isinstance(a, tuple):
            return slice_array(a[0]), slice_array(a[1])
        # else
        return slice_array(a)

    return get


def find_alternate_weight(
    get_param_func: Callable[[str], np.ndarray | tuple[np.ndarray, np.ndarray]],
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
        get_param_func: Retrieves weight tensors by name.  Returns the tensor value.
            For GGUF models, it may take and return a tuple of scales and quantized weights.
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
    weight_data = get_param_func(new_weight_name)

    # For quantized tensors, use the shape of the quantized value array
    size = (weight_data[1] if isinstance(weight_data, tuple) else weight_data).shape[0]

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

    new_process_func = _get_array_partition(parts, index, span)

    return new_weight_name, new_process_func
