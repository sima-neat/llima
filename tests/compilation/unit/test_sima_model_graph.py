from types import SimpleNamespace

import numpy as np
import pytest
from ml_dtypes import int4

from afe.backends.backends import Backend
from afe.ir.defines import Status, get_expected_tensor_value
from afe.ir.execute import create_node_executor, create_node_quant_executor
from afe.ir.operations import BatchMatmulOp
from afe.ir.serializer import load_awesomenet
from afe.ir.tensor_type import ScalarType, TensorType
from sima_lmm.model.base import BaseModel
from sima_lmm.model.model_graph import ModelGraph, tensor_type
from sima_lmm.model.sima_builder import activation_dtype, activation_type

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
@pytest.mark.parametrize("multiple_outputs", [False, True])
def test_model_graph_preserves_inputs_and_selected_output_types(quantizable, multiple_outputs):
    shape = (1, 1, 2, 32)
    specs = {"hidden": shape, "cache": TensorType(ScalarType.int8, shape)}
    graph = ModelGraph(_source(), specs, quantizable)
    builder, inputs = graph.raw, graph.inputs
    outputs = [inputs["cache"], inputs["hidden"]] if multiple_outputs else [inputs["hidden"]]
    # Finishing must select the requested outputs, not the last node created.
    builder.create_constant_node(np.array([99], np.float32))
    transformed = []
    net = graph.finish(outputs, transform_subnet=transformed.append)
    mla = net.nodes["MLA_0"].ir

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


@pytest.mark.parametrize("specs", [{"": (1, 1, 1, 32)}, {"input": (1, 1, 0, 32)}])
def test_model_graph_rejects_invalid_inputs(specs):
    with pytest.raises(ValueError, match="Invalid model input"):
        ModelGraph(_source(), specs, quantizable=True)


def test_model_graph_requires_an_output():
    graph = ModelGraph(_source(), {"input": (1, 1, 1, 32)}, quantizable=True)
    with pytest.raises(ValueError, match="at least one output"):
        graph.finish([])


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize(
    "transpose_a,transpose_b,equation",
    [
        (False, False, "nhwc,nhcq->nhwq"),
        (False, True, "nhwc,nhqc->nhwq"),
        (True, False, "nhcw,nhcq->nhwq"),
        (True, True, "nhcw,nhqc->nhwq"),
    ],
)
def test_einsum_lowers_to_mla_batch_matmul(quantizable, transpose_a, transpose_b, equation):
    lhs_shape = (1, 2, 16, 3) if transpose_a else (1, 2, 3, 16)
    rhs_shape = (1, 2, 5, 16) if transpose_b else (1, 2, 16, 5)
    graph = ModelGraph(_source(), {"lhs": lhs_shape, "rhs": rhs_shape}, quantizable)
    output = graph.einsum(equation, graph.inputs["lhs"], graph.inputs["rhs"])
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
    expected = np.einsum(
        equation, inputs["lhs"].astype(np.float32), inputs["rhs"].astype(np.float32)
    )
    actual = actual[0] if isinstance(actual, (tuple, list)) else actual
    if quantizable:
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    else:
        np.testing.assert_allclose(actual, expected, rtol=0.03, atol=0.002)


@pytest.mark.parametrize(
    "equation",
    [
        "bad",
        "nhwc,nhwc->nhwc",
        "nhwc,nhwc->",
        "...wc,...cq->...wq",
        "nnwc,nnqc->nnwq",
        "1234,1254->1235",
    ],
)
def test_einsum_rejects_unsupported_equations(equation):
    graph = ModelGraph(_source(), {"x": (1, 1, 16, 16)}, True)
    with pytest.raises(ValueError, match="Unsupported einsum"):
        graph.einsum(equation, graph.inputs["x"], graph.inputs["x"])


@pytest.mark.parametrize(
    "lhs,rhs,error",
    [
        ((1, 2, 3, 16), (2, 2, 5, 16), "equal batches"),
        ((1, 2, 3, 16), (1, 4, 5, 16), "singleton heads"),
        ((1, 2, 3, 16), (1, 2, 5, 32), "contraction dimensions"),
        ((1, 3, 16), (1, 2, 5, 16), "rank-four"),
        (TensorType(ScalarType.int8, (1, 2, 3, 16)), (1, 2, 5, 16), "matching FP32/BF16"),
    ],
)
def test_einsum_rejects_invalid_types_and_shapes(lhs, rhs, error):
    graph = ModelGraph(_source(), {"lhs": lhs, "rhs": rhs}, True)
    with pytest.raises(ValueError, match=error):
        graph.einsum("nhwc,nhqc->nhwq", graph.inputs["lhs"], graph.inputs["rhs"])


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
        "proj", graph.inputs["x"], relocatable=True,
        lora_rank=4 if block_size else None, merged_lora=True,
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
    graph.save([graph.raw.create_mul_node(graph.inputs["x"], constant), integer])
    net = load_awesomenet("component" + (".fp32" if quantizable else ""), str(tmp_path))
    assert net.status == (Status.RELAY if quantizable else Status.SIMA_QUANTIZED)
    # Components with ordinary layouts need no boilerplate tessellation overrides.
    assert BaseModel.get_mla_input_tessellate_params(_source()) == {}
    assert BaseModel.get_mla_output_tessellate_params(_source()) == {}


def test_cross_attention_projection_uses_query_length_for_packing():
    params = {"proj.weight": np.zeros((64, 64), np.float32)}
    graph = ModelGraph(_source(params), {"audio": (1, 1, 1500, 64)}, True)
    heads = graph.project_heads("proj", graph.inputs["audio"], 4, kv_len=1500, query_len=1)
    assert len(heads) == 1
    assert tensor_type(heads[0]).shape == (1, 4, 1500, 16)


def test_split_and_merge_heads_preserve_tokens():
    shape = (1, 1, 3, 64)
    graph = ModelGraph(_source(), {"x": shape}, True)
    split = graph.split_heads(graph.inputs["x"], 4)
    assert tensor_type(split).shape == (1, 4, 3, 16)
    output = graph.merge_heads(split)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(_run(graph.finish([output]), {"x": x}), x)


@pytest.mark.parametrize("per_token", [True, False])
def test_dynamic_quantization_infers_scale_shape_and_roundtrips(per_token):
    shape = (1, 1, 2, 32)
    graph = ModelGraph(_source(), {"x": shape}, False)
    quantized, scale = graph.quantize(graph.inputs["x"], per_token=per_token)
    assert tensor_type(quantized) == TensorType(ScalarType.int8, shape)
    assert tensor_type(scale).shape == ((1, 1, 2, 1) if per_token else (1, 1, 1, 1))
    output = graph.dequantize(quantized, scale)
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
    graph = ModelGraph(_source({"norm.weight": weight}), {"x": shape}, True)
    weighted = graph.rms_norm("norm", graph.inputs["x"], epsilon=1e-6, weight_offset=1.0)
    weightless = graph.rms_norm(None, graph.inputs["x"], epsilon=1e-6)
    x = np.linspace(-1, 1, np.prod(shape)).reshape(shape).astype(np.float32)
    normalized = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-6))
    result = graph.finish([weighted, weightless]).run(
        {"x": x}, node_callable=create_node_executor(False)
    )
    np.testing.assert_allclose(result[0], normalized * (weight + 1), rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(result[1], normalized, rtol=2e-6, atol=2e-7)


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


def test_einsum_accepts_renamed_labels_and_singleton_heads():
    specs = {"query": (1, 2, 3, 16), "key": (1, 1, 5, 16)}
    graph = ModelGraph(_source(), specs, True)
    output = graph.einsum(
        " b h t d, b h s d -> b h t s ", graph.inputs["query"], graph.inputs["key"]
    )
    rng = np.random.default_rng(2)
    inputs = {name: rng.normal(0, 0.1, shape).astype(np.float32) for name, shape in specs.items()}
    expected = np.einsum("bhtd,bhsd->bhts", inputs["query"], inputs["key"])
    np.testing.assert_allclose(_run(graph.finish([output]), inputs), expected, atol=2e-8, rtol=2e-6)
