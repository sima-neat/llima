from types import SimpleNamespace

import numpy as np
import pytest
from ml_dtypes import int4

from afe.backends.backends import Backend
from afe.ir.defines import Status, TensorValue, TupleValue, get_expected_tensor_value
from afe.ir.execute import create_node_executor, create_node_quant_executor
from afe.ir.operations import BatchMatmulOp, ConvAddActivationOp, StridedSliceOp
from afe.ir.serializer import load_awesomenet
from afe.ir.tensor_type import ScalarType, TensorType
from sima_lmm.model.base import BaseModel
from sima_lmm.model.model_graph import ModelGraph, tensor_type, activation_dtype, activation_type

pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


def _source(params=None, path=None):
    params = {} if params is None else params
    return SimpleNamespace(
        model_name="component",
        sima_model_sdk_path=path,
        get_hf_param=params.__getitem__,
        check_hf_param=params.__contains__,
    )


def _run(net, inputs):
    result = net.run(inputs, node_callable=create_node_executor(False))
    return result[0] if isinstance(result, (tuple, list)) else result


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize("use_jax", [False, True])
def test_run_reuses_graph_with_named_inputs_and_ordered_outputs(monkeypatch, quantizable, use_jax):
    shape = (1, 1, 2, 32)
    graph = ModelGraph(_source(), {"hidden": shape, "mask": shape, "cache": shape}, quantizable,
                       input_dtypes={"cache": np.int8})
    total = graph.add(graph.inputs["hidden"], graph.inputs["mask"])
    graph.finish([graph.inputs["cache"], total, graph.argmax(total)])
    selected = []

    def executor(**kwargs):
        selected.append(kwargs)
        return create_node_quant_executor(**kwargs)

    monkeypatch.setattr("sima_lmm.model.model_graph.create_node_quant_executor", executor)
    hidden = np.arange(64).reshape(shape).astype(activation_dtype(quantizable))
    mask = np.ones(shape, dtype=hidden.dtype)
    cache = np.full(shape, 7, dtype=np.int8)
    # Keyword order differs from declaration order; new values reuse the same graph.
    for offset in (0, 2):
        values = hidden + np.array(offset, dtype=hidden.dtype)
        result = graph.run(cache=cache, mask=mask, hidden=values, use_jax=use_jax)
        assert isinstance(result, list)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], cache)
        np.testing.assert_array_equal(result[1], (values + mask).astype(np.float32))
        np.testing.assert_array_equal(result[2], np.full((1, 1, 2, 1), 31, dtype=np.int32))
    assert selected == [{"fast_mode": True, "use_jax": use_jax}] * 2


@pytest.mark.parametrize("weight_dtype,group_size", [(np.int8, 32), (int4, 16)])
@pytest.mark.parametrize("use_jax", [False, True])
def test_run_executes_prequantized_projections(weight_dtype, group_size, use_jax):
    shape = (1, 1, 2, 32)
    weights = np.full((16, 32), 2, dtype=weight_dtype)
    scales = np.full((16, 32 // group_size), 0.5, dtype=np.float32)
    graph = ModelGraph(_source({"proj.weight": (scales, weights, group_size)}), {"x": shape}, False)
    graph.finish([graph.linear("proj", graph.inputs["x"])])
    x = np.full(shape, 0.25, dtype=activation_dtype(False))
    np.testing.assert_array_equal(graph.run(x=x, use_jax=use_jax)[0], np.full((1, 1, 2, 16), 8.0))


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize("finish_first", [True, False])
def test_run_and_save_share_finalization(tmp_path, quantizable, finish_first):
    shape = (1, 1, 2, 32)
    graph = ModelGraph(_source(path=tmp_path), {"x": shape}, quantizable)
    output = graph.mul(graph.inputs["x"], graph.constant([2.0]))
    x = np.ones(shape, dtype=activation_dtype(quantizable))
    if finish_first:
        graph.finish([output])
        graph.run(x=x)
        graph.save()
    else:
        graph.save([output])
    np.testing.assert_array_equal(graph.run(x=x)[0], x.astype(np.float32) * 2)
    net = load_awesomenet("component" + (".fp32" if quantizable else ""), str(tmp_path))
    actual = net.run({"x": x}, node_callable=create_node_quant_executor(fast_mode=True))
    np.testing.assert_array_equal(actual[0], x.astype(np.float32) * 2)
    with pytest.raises(RuntimeError, match="already finished"):
        graph.finish([output])
    with pytest.raises(ValueError, match="without outputs"):
        graph.save([output])


@pytest.mark.parametrize(
    "inputs,error,message",
    [
        ({}, ValueError, "missing.*x"),
        ({"x": np.ones((1, 1, 1, 32), np.float32), "extra": np.zeros(1)}, ValueError, "unexpected.*extra"),
        ({"x": np.ones((1, 1, 2, 32), np.float32)}, ValueError, "x: expected shape"),
        ({"x": np.ones((1, 1, 1, 32), np.float64)}, TypeError, "x: expected dtype float32"),
        ({"x": [1]}, TypeError, "x: expected a NumPy array"),
        ({"x": np.ones((1, 1, 1, 32), np.float32), "use_jax": "yes"}, TypeError, "use_jax must be a bool"),
    ],
)
def test_run_rejects_invalid_inputs(inputs, error, message):
    graph = ModelGraph(_source(), {"x": (1, 1, 1, 32)}, True)
    graph.finish([graph.inputs["x"]])
    with pytest.raises(error, match=message):
        graph.run(**inputs)


def test_execution_requires_explicit_finalization():
    graph = ModelGraph(_source(), {"x": (1, 1, 1, 32)}, True)
    with pytest.raises(RuntimeError, match="before run"):
        graph.run(x=np.ones((1, 1, 1, 32), np.float32))
    with pytest.raises(ValueError, match="Supply outputs"):
        graph.save()


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize("multiple_outputs", [False, True])
def test_model_graph_preserves_inputs_and_selected_output_types(quantizable, multiple_outputs):
    shape = (1, 1, 2, 32)
    specs = {"hidden": shape, "cache": shape}
    graph = ModelGraph(_source(), specs, quantizable, input_dtypes={"cache": np.int8})
    assert graph.dtype == np.dtype(activation_dtype(quantizable))
    builder, inputs = graph, graph.inputs
    outputs = [inputs["cache"], inputs["hidden"]] if multiple_outputs else [inputs["hidden"]]
    # Finishing must select the requested outputs, not the last node created.
    builder.create_constant_node(np.array([99], np.float32))
    transformed = []
    net = graph.finish(outputs, transform_subnet=transformed.append)
    mla = net.nodes["MLA_0"].ir
    output_type = mla.nodes[mla.output_node_name].get_type().output
    assert isinstance(output_type, TupleValue if multiple_outputs else TensorValue)

    assert net.status == (Status.RELAY if quantizable else Status.SIMA_QUANTIZED)
    assert list(net.input_node_names) == list(specs)
    assert list(mla.input_node_names) == [f"MLA_0/{name}" for name in specs]
    assert transformed == [mla]
    for name, scalar in (("hidden", activation_type(quantizable)), ("cache", ScalarType.int8)):
        outer_type = get_expected_tensor_value(net.nodes[name].get_type().output)
        inner_type = get_expected_tensor_value(mla.nodes[f"MLA_0/{name}"].get_type().output)
        assert outer_type == inner_type == TensorType(scalar, shape)

    values = {
        "hidden": np.full(shape, 0.25, dtype=activation_dtype(quantizable)),
        "cache": np.full(shape, 7, dtype=np.int8),
    }
    execute = (
        create_node_executor(False) if quantizable else create_node_quant_executor(False, False)
    )
    result = net.run(values, node_callable=execute)
    result = list(result) if isinstance(result, (tuple, list)) else [result]
    expected = [values["hidden"].astype(np.float32)]
    if multiple_outputs:
        expected.insert(0, values["cache"])
    assert len(result) == len(expected)
    for actual, reference in zip(result, expected):
        assert actual.dtype == reference.dtype
        np.testing.assert_array_equal(actual, reference)
    casts = [node for node in net.nodes.values() if node.name.startswith("cast_")]
    assert len(casts) == (0 if quantizable else 1)
    assert all(node.ir.backend == Backend.EV for node in casts)


@pytest.mark.parametrize(
    "specs,input_dtypes,error",
    [
        ({"": (1, 1, 1, 32)}, None, "Invalid model input"),
        ({"input": (1, 1, 0, 32)}, None, "Invalid model input"),
        ({"input": (1, 1, 1, 32)}, {"missing": np.int8}, "unknown inputs"),
        ({"input": (1, 1, 1, 32)}, {"input": np.complex64}, "unsupported input dtype"),
        ({"input": TensorType(ScalarType.int8, (1, 1, 1, 32))}, {"input": np.int8}, "not both"),
    ],
)
def test_model_graph_rejects_invalid_inputs(specs, input_dtypes, error):
    with pytest.raises(ValueError, match=error):
        ModelGraph(_source(), specs, quantizable=True, input_dtypes=input_dtypes)


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize(
    "begin,end,axis,operator",
    [
        (0, 16, 3, StridedSliceOp),
        (0, 40, 3, ConvAddActivationOp),
        (40, 80, -1, ConvAddActivationOp),
        (1, 3, 2, StridedSliceOp),
    ],
)
def test_slice_preserves_values_and_handles_channel_alignment(
    quantizable, begin, end, axis, operator
):
    shape = (1, 1, 3, 80)
    graph = ModelGraph(_source(), {"x": shape}, quantizable)
    outputs = [
        graph.slice(graph.inputs["x"], [begin], [end], [1], [axis]),
        graph.slice(graph.inputs["x"], start=begin, stop=end, axis=axis),
    ]
    assert all(isinstance(output.ir.operation, operator) for output in outputs)
    x = np.arange(np.prod(shape)).reshape(shape).astype(activation_dtype(quantizable))
    selection = [slice(None)] * 4
    selection[axis] = slice(begin, end)
    execute = (
        create_node_executor(False) if quantizable else create_node_quant_executor(False, False)
    )
    actual = graph.finish(outputs).run({"x": x}, node_callable=execute)
    assert len(actual) == 2
    for value in actual:
        np.testing.assert_array_equal(value, x[tuple(selection)].astype(np.float32))


@pytest.mark.parametrize("quantizable", [True, False])
def test_slice_keeps_integer_channels_on_native_path(quantizable):
    shape = (1, 1, 2, 80)
    graph = ModelGraph(_source(), {"x": TensorType(ScalarType.int8, shape)}, quantizable)
    outputs = [
        graph.slice(graph.inputs["x"], [0], [40], [1], [3]),
        graph.slice(graph.inputs["x"], stop=40, axis=-1),
    ]
    for output in outputs:
        assert isinstance(output.ir.operation, StridedSliceOp)
        assert tensor_type(output) == TensorType(ScalarType.int8, (1, 1, 2, 40))


def test_model_graph_requires_an_output():
    graph = ModelGraph(_source(), {"input": (1, 1, 1, 32)}, quantizable=True)
    with pytest.raises(ValueError, match="at least one output"):
        graph.finish([])


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize(
    "transpose_a,transpose_b",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_matmul_lowers_to_mla_batch_matmul(quantizable, transpose_a, transpose_b):
    lhs_shape = (1, 2, 16, 3) if transpose_a else (1, 2, 3, 16)
    rhs_shape = (1, 2, 5, 16) if transpose_b else (1, 2, 16, 5)
    graph = ModelGraph(_source(), {"lhs": lhs_shape, "rhs": rhs_shape}, quantizable)
    output = graph.matmul(
        graph.inputs["lhs"], graph.inputs["rhs"], transpose_a=transpose_a, transpose_b=transpose_b
    )
    assert isinstance(output.ir.operation, BatchMatmulOp)
    assert output.ir.backend == Backend.MLA
    assert (output.ir.attrs.transpose_a, output.ir.attrs.transpose_b) == (transpose_a, transpose_b)
    assert tensor_type(output).shape == (1, 2, 3, 5)
    net = graph.finish([output])
    rng = np.random.default_rng(9)
    inputs = {
        name: rng.normal(0, 0.2, shape).astype(activation_dtype(quantizable))
        for name, shape in (("lhs", lhs_shape), ("rhs", rhs_shape))
    }
    execute = (
        create_node_executor(False) if quantizable else create_node_quant_executor(False, False)
    )
    actual = net.run(inputs, node_callable=execute)
    lhs, rhs = inputs["lhs"].astype(np.float32), inputs["rhs"].astype(np.float32)
    expected = np.matmul(
        lhs.swapaxes(-1, -2) if transpose_a else lhs,
        rhs.swapaxes(-1, -2) if transpose_b else rhs,
    )
    actual = actual[0] if isinstance(actual, (tuple, list)) else actual
    if quantizable:
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    else:
        np.testing.assert_allclose(actual, expected, rtol=0.03, atol=0.002)


@pytest.mark.parametrize(
    "lhs,rhs,error",
    [
        ((1, 2, 3, 16), (2, 2, 5, 16), "equal batches"),
        ((1, 3, 3, 16), (1, 2, 5, 16), "divisible head counts"),
        ((1, 2, 3, 16), (1, 2, 5, 32), "contraction dimensions"),
        ((1, 3, 16), (1, 2, 5, 16), "rank-four"),
        (TensorType(ScalarType.int8, (1, 2, 3, 16)), (1, 2, 5, 16), "matching FP32/BF16"),
    ],
)
def test_matmul_rejects_invalid_types_and_shapes(lhs, rhs, error):
    graph = ModelGraph(_source(), {"lhs": lhs, "rhs": rhs}, True)
    with pytest.raises(ValueError, match=error):
        graph.matmul(graph.inputs["lhs"], graph.inputs["rhs"], transpose_b=True)


@pytest.mark.parametrize("block_size", [None, 32])
def test_linear_preserves_quantized_weight_layout_and_metadata(block_size):
    # 48 channels deliberately do not align to G32. Per-channel has one scale.
    weights = np.arange(16 * 48).reshape(16, 48) % 15 - 7
    weights = weights.astype(int4 if block_size else np.int8)
    groups = 2 if block_size else 1
    scales = np.arange(1, 16 * groups + 1, dtype=np.float32).reshape(16, groups) / 100
    params = {
        "proj.weight": (scales, weights, block_size) if block_size else (scales, weights),
        "proj.bias": np.arange(16, dtype=np.float32),
    }
    graph = ModelGraph(_source(params), {"x": (1, 1, 2, 48)}, False)
    output = graph.linear(
        "proj",
        graph.inputs["x"],
        relocatable=True,
        lora_rank=4 if block_size else None,
        merged_lora=True,
    )
    attrs = output.ir.quant_attrs
    np.testing.assert_array_equal(attrs.weight_quant_data.reshape(48, 16), weights.T)
    np.testing.assert_array_equal(attrs.bias_quant_data, params["proj.bias"])
    assert attrs.conv_attrs.reloc_name == "proj.weight"
    assert attrs.c_block_size == (block_size or 48)
    # SiMa IR stores scales as [input groups, output channels].
    np.testing.assert_array_equal(attrs.requant.sc_correction, scales.T)


@pytest.mark.parametrize("proportional", [False, True])
def test_rope_preserves_nonrotary_channels(proportional):
    specs = {"x": (1, 2, 3, 64), "cos": (1, 1, 3, 16), "sin": (1, 1, 3, 16)}
    graph = ModelGraph(_source(), specs, True)
    output = graph.rope(
        graph.inputs["x"], graph.inputs["cos"], graph.inputs["sin"], 32, proportional=proportional
    )
    net = graph.finish([output])
    rng = np.random.default_rng(3)
    inputs = {name: rng.normal(size=shape).astype(np.float32) for name, shape in specs.items()}
    expected = inputs["x"].copy()
    start = 32 if proportional else 16
    real, imag = inputs["x"][..., :16], inputs["x"][..., start : start + 16]
    expected[..., :16] = real * inputs["cos"] - imag * inputs["sin"]
    expected[..., start : start + 16] = real * inputs["sin"] + imag * inputs["cos"]
    np.testing.assert_array_equal(_run(net, inputs), expected)


@pytest.mark.parametrize("quantizable", [True, False])
def test_context_saves_constants_and_infers_tessellation_defaults(tmp_path, quantizable):
    graph = ModelGraph(_source(path=tmp_path), {"x": (1, 1, 2, 32)}, quantizable)
    constant = graph.constant([0.5])
    assert tensor_type(constant).scalar == activation_type(quantizable)
    integer = graph.constant([1], dtype=np.int32)
    assert tensor_type(integer).scalar == ScalarType.int32
    graph.save([graph.mul(graph.inputs["x"], constant), integer])
    net = load_awesomenet("component" + (".fp32" if quantizable else ""), str(tmp_path))
    assert net.status == (Status.RELAY if quantizable else Status.SIMA_QUANTIZED)
    # Components with ordinary layouts need no boilerplate tessellation overrides.
    assert BaseModel.get_mla_input_tessellate_params(_source()) == {}
    assert BaseModel.get_mla_output_tessellate_params(_source()) == {}


def test_cross_attention_projection_uses_basic_linear_and_head_split():
    params = {"proj.weight": np.zeros((64, 64), np.float32)}
    graph = ModelGraph(_source(params), {"audio": (1, 1, 1500, 64)}, True)
    heads = graph.split_heads(graph.linear("proj", graph.inputs["audio"]), 4)
    assert tensor_type(heads).shape == (1, 4, 1500, 16)


def test_split_and_merge_heads_preserve_tokens():
    shape = (1, 1, 3, 64)
    graph = ModelGraph(_source(), {"x": shape}, True)
    split = graph.split_heads(graph.inputs["x"], 4)
    assert tensor_type(split).shape == (1, 4, 3, 16)
    output = graph.merge_heads(split)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(_run(graph.finish([output]), {"x": x}), x)


def test_split_heads_repeats_each_head():
    shape = (1, 1, 3, 32)
    graph = ModelGraph(_source(), {"x": shape}, True)
    output = graph.split_heads(graph.inputs["x"], 2, repeat=3)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = np.repeat(x.reshape(1, 3, 2, 16).transpose(0, 2, 1, 3), 3, axis=1)
    np.testing.assert_array_equal(_run(graph.finish([output]), {"x": x}), expected)


def test_unaligned_heads_split_and_merge_preserve_values():
    shape = (1, 1, 3, 24)
    graph = ModelGraph(_source(), {"x": shape}, True)
    heads = graph.split_heads(graph.inputs["x"], 4)
    assert tensor_type(heads).shape == (1, 4, 3, 6)
    output = graph.merge_heads(heads)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(_run(graph.finish([output]), {"x": x}), x)


@pytest.mark.parametrize("block_size", [None, 32])
def test_linear_scaling_preserves_packed_weights_and_scales_bias(block_size):
    weights = (np.arange(16 * 48).reshape(16, 48) % 15 - 7).astype(int4 if block_size else np.int8)
    scales = np.ones((16, 2 if block_size else 1), dtype=np.float32)
    bias = np.arange(16, dtype=np.float32)
    params = {
        "proj.weight": (scales, weights, block_size) if block_size else (scales, weights),
        "proj.bias": bias,
    }
    graph = ModelGraph(_source(params), {"x": (1, 1, 2, 48)}, False)
    output = graph.linear("proj", graph.inputs["x"], scale=0.125)
    attrs = output.ir.quant_attrs
    np.testing.assert_array_equal(attrs.weight_quant_data.reshape(48, 16), weights.T)
    np.testing.assert_array_equal(attrs.requant.sc_correction, scales.T * 0.125)
    np.testing.assert_array_equal(attrs.bias_quant_data, bias * 0.125)
    assert attrs.c_block_size == (block_size or 48)


@pytest.mark.parametrize(
    "projection,start,end", [("q_proj", 0, 32), ("k_proj", 32, 48), ("v_proj", 48, 64)]
)
@pytest.mark.parametrize("precision", ["float32", "int8", "int4"])
def test_bundled_linear_preserves_scaling_and_caller_transforms(projection, start, end, precision):
    weights = (np.arange(64 * 48).reshape(64, 48) % 15 - 7).astype(
        np.float32 if precision == "float32" else np.int8 if precision == "int8" else int4
    )
    block_size = 32 if precision == "int4" else None
    scales = np.arange(1, 64 * (2 if block_size else 1) + 1, dtype=np.float32).reshape(64, -1)
    bias = np.arange(64, dtype=np.float32)
    source_weights = weights
    if precision != "float32":
        source_weights = (scales, weights, block_size) if block_size else (scales, weights)
    params = {
        "attn.qkv_proj.weight": source_weights,
        "attn.qkv_proj.bias": bias,
    }
    shape = (1, 1, 1, 48)
    graph = ModelGraph(_source(params), {"x": shape}, precision == "float32")
    output = graph.linear(
        f"attn.{projection}",
        graph.inputs["x"],
        scale=0.125,
        q_size=32,
        kv_size=16,
        weight_process_func=lambda x: x[::-1],
        scale_process_func=lambda x: x[::-1] * 2,
        bias_process_func=lambda x: x[::-1] + 1,
    )
    expected_weights = weights[start:end][::-1]
    expected_bias = (bias[start:end][::-1] + 1) * 0.125
    if precision == "float32":
        x = np.ones(shape, np.float32)
        expected = x @ expected_weights.T * 0.125 + expected_bias
        np.testing.assert_array_equal(_run(graph.finish([output]), {"x": x}), expected)
    else:
        attrs = output.ir.quant_attrs
        np.testing.assert_array_equal(
            attrs.weight_quant_data.reshape(48, end - start), expected_weights.T
        )
        np.testing.assert_array_equal(
            attrs.requant.sc_correction, scales[start:end][::-1].T * 0.25
        )
        np.testing.assert_array_equal(attrs.bias_quant_data, expected_bias)
        assert attrs.c_block_size == (block_size or 48)


@pytest.mark.parametrize("quantizable", [True, False])
def test_softcap_matches_tanh_and_preserves_zero(quantizable):
    shape = (1, 1, 1, 32)
    graph = ModelGraph(_source(), {"x": shape}, quantizable)
    output = graph.softcap(graph.inputs["x"], 30.0)
    x = np.linspace(-30, 30, 32).reshape(shape).astype(activation_dtype(quantizable))
    x[..., 16] = 0
    execute = create_node_executor(False) if quantizable else create_node_quant_executor(False, False)
    actual = graph.finish([output]).run({"x": x}, node_callable=execute)
    actual = actual[0] if isinstance(actual, (tuple, list)) else actual
    expected = 30 * np.tanh(x.astype(np.float32) / 30)
    np.testing.assert_allclose(
        actual, expected, rtol=2e-6 if quantizable else 0.03,
        atol=2e-6 if quantizable else 0.25,
    )
    assert actual[..., 16].item() == 0


@pytest.mark.parametrize("split_heads", [False, True])
@pytest.mark.parametrize("mask_shape", [(1, 2, 3, 5), (1, 1, 1, 5), (5,), (1,)])
def test_attention_preserves_broadcast_values_and_mask(monkeypatch, split_heads, mask_shape):
    import sima_lmm.model.model_graph as graph_module

    if split_heads:
        monkeypatch.setattr(graph_module, "mla_max_num_rows", 1)
    specs = {"q": (1, 2, 3, 16), "k": (1, 2, 5, 16), "v": (1, 1, 5, 16), "mask": mask_shape}
    graph = ModelGraph(_source(), specs, True)
    output = graph.attention(
        graph.inputs["q"],
        graph.inputs["k"],
        graph.inputs["v"],
        mask=graph.inputs["mask"],
        score_scale=0.25,
    )
    net = graph.finish([output])
    matmuls = [
        node
        for node in net.nodes["MLA_0"].ir.nodes.values()
        if isinstance(node.ir.operation, BatchMatmulOp)
    ]
    assert len(matmuls) == (4 if split_heads else 2)
    if split_heads:
        assert all(tensor_type(node).shape[1] == 1 for node in matmuls)
    rng = np.random.default_rng(5)
    inputs = {name: rng.normal(size=shape).astype(np.float32) for name, shape in specs.items()}
    scores = np.einsum("nhtc,nhsc->nhts", inputs["q"], inputs["k"]) * 0.25 + inputs["mask"]
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    expected = np.matmul(probs, inputs["v"])
    np.testing.assert_allclose(_run(net, inputs), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("split_heads", [False, True])
@pytest.mark.parametrize(
    "mask_shape",
    [(1, 3, 3, 5), (3, 3, 5), (3, 5), (2, 2, 3, 5), (1, 2, 4, 5), (1, 2, 3, 6), (1, 1, 2, 3, 5)],
)
def test_attention_rejects_masks_that_do_not_broadcast_to_scores(monkeypatch, split_heads, mask_shape):
    import sima_lmm.model.model_graph as graph_module

    if split_heads:
        monkeypatch.setattr(graph_module, "mla_max_num_rows", 1)
    specs = {"q": (1, 2, 3, 16), "k": (1, 2, 5, 16), "v": (1, 1, 5, 16), "mask": mask_shape}
    graph = ModelGraph(_source(), specs, True)
    with pytest.raises(ValueError, match="attention mask .* broadcast to score shape"):
        graph.attention(*(graph.inputs[name] for name in ("q", "k", "v")), mask=graph.inputs["mask"])


def test_rope2d_preserves_axis_pairing_with_unaligned_quarters():
    specs = {"x": (1, 2, 3, 96)}
    specs.update({name: (1, 1, 3, 24) for name in ("cos_x", "sin_x", "cos_y", "sin_y")})
    graph = ModelGraph(_source(), specs, True)
    output = graph.rope2d(*(graph.inputs[name] for name in specs))
    rng = np.random.default_rng(4)
    inputs = {name: rng.normal(size=shape).astype(np.float32) for name, shape in specs.items()}
    xr, xi, yr, yi = np.split(inputs["x"], 4, axis=-1)
    expected = np.concatenate(
        [
            xr * inputs["cos_x"] - xi * inputs["sin_x"],
            xr * inputs["sin_x"] + xi * inputs["cos_x"],
            yr * inputs["cos_y"] - yi * inputs["sin_y"],
            yr * inputs["sin_y"] + yi * inputs["cos_y"],
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(_run(graph.finish([output]), inputs), expected)


def test_attention_scales_scores_before_adding_mask():
    specs = {name: (1, 2, 3, 16) for name in ("q", "k", "v")}
    specs["mask"] = (1, 1, 3, 3)
    graph = ModelGraph(_source(), specs, True)
    output = graph.attention(
        graph.inputs["q"],
        graph.inputs["k"],
        graph.inputs["v"],
        mask=graph.inputs["mask"],
        score_scale=0.125,
    )
    rng = np.random.default_rng(4)
    inputs = {name: rng.normal(size=shape).astype(np.float32) for name, shape in specs.items()}
    inputs["mask"][..., -1] = -20
    scores = np.einsum("nhtc,nhsc->nhts", inputs["q"], inputs["k"]) * 0.125 + inputs["mask"]
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    expected = np.einsum("nhts,nhsc->nhtc", probs, inputs["v"])
    np.testing.assert_allclose(_run(graph.finish([output]), inputs), expected, rtol=2e-5, atol=2e-6)


def test_space_to_depth_preserves_spatial_block_order():
    shape = (1, 4, 6, 16)
    graph = ModelGraph(_source(), {"x": shape}, True)
    output = graph.space_to_depth(graph.inputs["x"], 2)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = x.reshape(1, 2, 2, 3, 2, 16).transpose(0, 1, 3, 2, 4, 5).reshape(1, 2, 3, 64)
    np.testing.assert_array_equal(_run(graph.finish([output]), {"x": x}), expected)


def test_conv_transforms_rank_five_grouped_weights_without_changing_metadata():
    weight = (np.arange(16 * 24).reshape(16, 3, 2, 2, 2) % 15 - 7).astype(int4)
    scales = np.arange(1, 33, dtype=np.float32).reshape(16, 2) / 100
    graph = ModelGraph(_source({"patch.weight": (scales, weight, 16)}), {"x": (1, 1, 2, 24)}, False)
    output = graph.conv(
        "patch",
        graph.inputs["x"],
        weight_process_func=lambda w: w.reshape(16, 24, 1, 1),
        scale_process_func=lambda s: s,
    )
    attrs = output.ir.quant_attrs
    assert attrs.c_block_size == 16
    np.testing.assert_array_equal(attrs.weight_quant_data.reshape(24, 16), weight.reshape(16, 24).T)
    np.testing.assert_array_equal(attrs.requant.sc_correction, scales.T)


@pytest.mark.parametrize("repeat", [0, -1, 1.5])
def test_split_heads_rejects_invalid_repetition(repeat):
    graph = ModelGraph(_source(), {"x": (1, 1, 2, 32)}, True)
    with pytest.raises(ValueError, match="positive integer repeat"):
        graph.split_heads(graph.inputs["x"], 2, repeat=repeat)


def test_rope2d_rejects_incompatible_tables():
    graph = ModelGraph(_source(), {"x": (1, 2, 3, 96), "table": (1, 1, 4, 24)}, True)
    with pytest.raises(ValueError, match="tables must broadcast"):
        graph.rope2d(graph.inputs["x"], *(graph.inputs["table"] for _ in range(4)))


def test_space_to_depth_rejects_partial_spatial_blocks():
    graph = ModelGraph(_source(), {"x": (1, 3, 4, 16)}, True)
    with pytest.raises(ValueError, match="H/W divisible"):
        graph.space_to_depth(graph.inputs["x"], 2)


@pytest.mark.parametrize("per_token", [True, False])
def test_dynamic_quantization_infers_scale_shape_and_roundtrips(per_token):
    shape = (1, 1, 2, 32)
    graph = ModelGraph(_source(), {"x": shape}, False)
    quantized, scale = graph.quant(graph.inputs["x"], per_token=per_token)
    assert tensor_type(quantized) == TensorType(ScalarType.int8, shape)
    assert tensor_type(scale).shape == ((1, 1, 2, 1) if per_token else (1, 1, 1, 1))
    output = graph.dequant(quantized, scale)
    x = np.linspace(-0.5, 0.5, np.prod(shape)).reshape(shape).astype(activation_dtype(False))
    actual = graph.finish([output]).run(
        {"x": x}, node_callable=create_node_quant_executor(False, False)
    )
    actual = actual[0] if isinstance(actual, (tuple, list)) else actual
    np.testing.assert_allclose(actual, x.astype(np.float32), atol=0.004, rtol=0)


@pytest.mark.parametrize("gated", [True, False])
def test_shared_mlp_matches_dense_reference(gated):
    rng = np.random.default_rng(8)
    projections = ("gate_proj", "up_proj", "down_proj") if gated else ("fc1", "fc2")
    weights = {name: rng.normal(0, 0.05, (32, 32)).astype(np.float32) for name in projections}
    graph = ModelGraph(
        _source({f"mlp.{name}.weight": w for name, w in weights.items()}),
        {"x": (1, 1, 2, 32)},
        True,
    )
    x = rng.normal(0, 0.1, (1, 1, 2, 32)).astype(np.float32)
    hidden = x @ weights[projections[0]].T
    hidden = hidden / (1 + np.exp(-hidden))
    if gated:
        hidden *= x @ weights[projections[1]].T
    expected = x + hidden @ weights[projections[-1]].T
    output = graph.mlp(
        "mlp", graph.inputs["x"], "silu", projections=projections, residual=graph.inputs["x"]
    )
    np.testing.assert_allclose(
        _run(graph.finish([output]), {"x": x}), expected, atol=2e-8, rtol=2e-6
    )


def test_rms_norm_supports_weight_offset_and_inferred_weightless_channels():
    shape = (1, 1, 2, 32)
    weight = np.linspace(-0.2, 0.2, 32).astype(np.float32)
    source = _source({"norm.weight": weight})
    source.cfg = SimpleNamespace(lm_cfg=SimpleNamespace(rms_norm_eps=1e-6, rms_norm_unit_offset=True))
    graph = ModelGraph(source, {"x": shape}, True)
    weighted = graph.rms_norm("norm", graph.inputs["x"], epsilon=1e-6, weight_offset=1.0)
    weightless = graph.rms_norm(None, graph.inputs["x"])
    language_norm = graph.rms_norm("norm", graph.inputs["x"])
    explicit_norm = graph.rms_norm("norm", graph.inputs["x"], epsilon=1e-4)
    x = np.linspace(-1, 1, np.prod(shape)).reshape(shape).astype(np.float32)
    normalized = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-6))
    result = graph.finish([weighted, weightless, language_norm, explicit_norm]).run(
        {"x": x}, node_callable=create_node_executor(False)
    )
    np.testing.assert_allclose(result[0], normalized * (weight + 1), rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(result[1], normalized, rtol=2e-6, atol=2e-7)
    np.testing.assert_array_equal(result[2], result[0])
    explicit = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-4))
    np.testing.assert_allclose(result[3], explicit * weight, rtol=2e-6, atol=2e-7)

    graph = ModelGraph(_source(), {"x": shape}, True)
    with pytest.raises(ValueError, match="requires epsilon"):
        graph.rms_norm(None, graph.inputs["x"])


def test_tessellation_defaults_and_explicit_cache_layout_overrides():
    from afe.apis.defines import TensorDRAMLayout, TensorTessellateParameters
    from sima_lmm.model.sima_analysis import get_tessellate_parameters

    graph = ModelGraph(_source(), {"x": (1, 1, 2, 32), "cache": (1, 2, 2, 32)}, True)
    net = graph.finish([graph.inputs["x"], graph.inputs["cache"]])
    custom = TensorTessellateParameters(
        tile_shape=(1, 1, 1, 32),
        enable_mla=True,
        dram_layout=TensorDRAMLayout.HWC16,
        dram_shape=(1, 2, 128, 32),
    )
    params = get_tessellate_parameters(SimpleNamespace(_net=net), {1: custom}, {-1: custom})
    mla = net.nodes["MLA_0"]
    assert params[mla.input_names[0]].tile_shape == (0, 0, 0, 0)
    assert params[mla.input_names[1]].dram_shape == custom.dram_shape
    output = mla.ir.nodes[mla.ir.output_node_name].input_node_names[-1]
    assert params[f"{output}_output"].dram_shape == custom.dram_shape
    assert params[f"{output}_output"].persistent_mem_name.startswith("output_1/")


def test_unknown_weight_override_is_rejected():
    graph = ModelGraph(_source(), {"x": (1, 1, 1, 32)}, True)
    with pytest.raises(ValueError, match="Unknown weight options"):
        graph.linear("proj", graph.inputs["x"], src_weights_name="typo.weight")


def test_matmul_repeats_kv_heads_for_gqa():
    specs = {"query": (1, 8, 3, 16), "key": (1, 2, 5, 16)}
    graph = ModelGraph(_source(), specs, True)
    output = graph.matmul(graph.inputs["query"], graph.inputs["key"], transpose_b=True)
    rng = np.random.default_rng(2)
    inputs = {name: rng.normal(0, 0.1, shape).astype(np.float32) for name, shape in specs.items()}
    keys = np.repeat(inputs["key"], 4, axis=1)
    expected = np.matmul(inputs["query"], keys.swapaxes(-1, -2))
    np.testing.assert_allclose(_run(graph.finish([output]), inputs), expected, atol=2e-8, rtol=2e-6)
