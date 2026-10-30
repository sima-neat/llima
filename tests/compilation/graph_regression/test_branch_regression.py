import hashlib
import warnings
from pathlib import Path

import numpy as np
import pytest

from afe.ir.defines import data_value_elements, get_expected_tensor_value
from afe.ir.execute import create_node_executor
from afe.ir.net import AwesomeNet
from afe.ir.serializer import load_awesomenet
from afe.ir.tensor_type import ScalarType

from tests.compilation.cases import GRAPH_REGRESSION_CASES, GraphRegressionCase
from tests.compilation.helpers.paths import require_readable_path


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_graph_regression]

ACTIVE_CASES = tuple(case for case in GRAPH_REGRESSION_CASES if case.mode != "disabled")


def _case_parameter(case: GraphRegressionCase):
    marks = []
    if case.target_model_folder is not None:
        marks.extend((pytest.mark.serial, pytest.mark.high_memory))
    return pytest.param(case, id=case.id, marks=marks)


def _graph_signature(net: AwesomeNet) -> tuple:
    inputs = tuple(
        (name, spec.scalar, tuple(spec.shape))
        for name in net.input_node_names
        for spec in [get_expected_tensor_value(net.nodes[name].get_type().output)]
    )
    outputs = tuple(
        (str(index), spec.scalar, tuple(spec.shape))
        for index, spec in enumerate(
            data_value_elements(net.nodes[net.output_node_name].get_type().output)
        )
    )
    return inputs, outputs


def _validate_graph(path: Path) -> tuple:
    net = load_awesomenet(path.name, str(path.parent))
    signature = _graph_signature(net)
    inputs, outputs = signature
    assert inputs, f"native graph has no runtime inputs: {path}"
    assert outputs, f"native graph has no runtime outputs: {path}"
    assert len({item[0] for item in inputs}) == len(inputs)
    assert len({item[0] for item in outputs}) == len(outputs)
    assert all(all(dim > 0 for dim in item[2]) for item in inputs + outputs)
    return signature


def _make_inputs(case: GraphRegressionCase, signature: tuple) -> dict[str, np.ndarray]:
    digest = hashlib.sha256(case.id.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    feeds = {}
    for name, elem_type, dimensions in signature[0]:
        dtype = np.dtype(ScalarType.numpy_type(elem_type))
        shape = dimensions
        if np.issubdtype(dtype, np.floating):
            value = (
                np.ones(shape, dtype=dtype)
                if name == "linear_valid_mask"
                else rng.uniform(-1.0, 1.0, shape).astype(dtype)
            )
        elif np.issubdtype(dtype, np.integer):
            value = rng.integers(0, 4, shape, dtype=dtype)
        elif np.issubdtype(dtype, np.bool_):
            value = rng.integers(0, 2, shape).astype(dtype)
        else:
            raise TypeError(f"Unsupported native graph input dtype for {case.id}: {dtype}")
        feeds[name] = value
    return feeds


def _run_graph(path: Path, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    net = load_awesomenet(path.name, str(path.parent))
    outputs = net.run(feeds, node_callable=create_node_executor(False))
    return list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]


def _validate_runtime_outputs(
    case: GraphRegressionCase,
    signature: tuple,
    outputs: list[np.ndarray],
) -> None:
    expected_outputs = signature[1]
    assert len(outputs) == len(expected_outputs), (
        f"native graph output count differs from its graph interface for {case.id}: "
        f"runtime={len(outputs)}, graph={len(expected_outputs)}"
    )
    for output, (name, elem_type, dimensions) in zip(
        outputs, expected_outputs, strict=True
    ):
        expected_dtype = np.dtype(ScalarType.numpy_type(elem_type))
        assert output.dtype == expected_dtype, (
            f"native graph runtime output {name} has dtype {output.dtype}; "
            f"graph declares {expected_dtype}"
        )
        assert output.ndim == len(dimensions), (
            f"native graph runtime output {name} has rank {output.ndim}; "
            f"graph declares rank {len(dimensions)}"
        )
        assert np.all(np.isfinite(output)), f"Non-finite output {name} for {case.id}"
        for actual, expected in zip(output.shape, dimensions, strict=True):
            if isinstance(expected, int) and expected > 0:
                assert actual == expected, (
                    f"native graph runtime output {name} has shape {output.shape}; "
                    f"graph declares {dimensions}"
                )


def _manifest_case(
    manifest: dict,
    case: GraphRegressionCase,
    source: str,
) -> dict:
    manifest_case = manifest.get("cases", {}).get(case.id)
    if manifest_case is None:
        raise KeyError(
            f"{source} native graph manifest is missing required case {case.id}"
        )
    return manifest_case


def _case_paths(
    root: Path,
    manifest: dict,
    case: GraphRegressionCase,
    source: str,
) -> list[Path]:
    manifest_case = _manifest_case(manifest, case, source)
    status = manifest_case.get("status", "available")
    if status != "available":
        raise RuntimeError(
            f"{source} native graph for {case.id} has unexpected status {status}: "
            f"{manifest_case.get('reason', 'no reason recorded')}"
        )
    relative_paths = manifest_case.get("graph_paths")
    if not isinstance(relative_paths, list) or not relative_paths:
        raise RuntimeError(
            f"{source} native graph manifest has no artifact paths for {case.id}"
        )
    return [
        require_readable_path(
            root / relative_path,
            f"{source} generated native graph artifact {index} for {case.id}",
        )
        for index, relative_path in enumerate(relative_paths)
    ]


def _report_regression(case: GraphRegressionCase, message: str) -> None:
    if case.mode == "required":
        pytest.fail(message)
    warnings.warn(
        f"Informative native graph regression for {case.id}: {message}", stacklevel=2
    )


@pytest.mark.parametrize("case", [_case_parameter(case) for case in ACTIVE_CASES])
def test_branch_relative_graph_regression(
    case: GraphRegressionCase,
    graph_validation_mode: str,
    candidate_graph_root: Path,
    candidate_graph_manifest: dict,
    base_graph_root: Path | None,
    base_graph_manifest: dict | None,
):
    candidate_paths = _case_paths(
        candidate_graph_root, candidate_graph_manifest, case, "candidate"
    )
    candidate_results = []
    for path in candidate_paths:
        signature = _validate_graph(path)
        feeds = _make_inputs(case, signature)
        outputs = _run_graph(path, feeds)
        _validate_runtime_outputs(case, signature, outputs)
        candidate_results.append((signature, feeds, outputs))

    if graph_validation_mode == "candidate-only":
        return

    assert base_graph_root is not None
    assert base_graph_manifest is not None
    base_manifest_case = _manifest_case(base_graph_manifest, case, "baseline")
    if base_manifest_case.get("status", "available") == "unavailable":
        _report_regression(
            case,
            "Baseline compiler could not generate this native graph case: "
            f"{base_manifest_case.get('reason', 'no reason recorded')}",
        )
        return
    base_paths = _case_paths(
        base_graph_root, base_graph_manifest, case, "baseline"
    )
    if len(candidate_paths) != len(base_paths):
        _report_regression(case, "Candidate and baseline graph counts differ")
        return
    candidate_layers = _manifest_case(
        candidate_graph_manifest, case, "candidate"
    ).get("layer_indices")
    if candidate_layers != base_manifest_case.get("layer_indices"):
        _report_regression(case, "Candidate and baseline vision layer selections differ")
        return

    differences = []
    for graph_index, (base_path, candidate_result) in enumerate(
        zip(base_paths, candidate_results, strict=True)
    ):
        candidate_signature, feeds, candidate_outputs = candidate_result
        base_signature = _validate_graph(base_path)
        if candidate_signature[0] != base_signature[0]:
            differences.append(
                f"graph {graph_index} input interface differs: "
                f"candidate={candidate_signature[0]}, base={base_signature[0]}"
            )
            continue
        base_outputs = _run_graph(base_path, feeds)
        _validate_runtime_outputs(case, base_signature, base_outputs)
        if len(candidate_outputs) != len(base_outputs):
            differences.append(f"graph {graph_index} output counts differ")
            continue

        for index, (candidate, base) in enumerate(
            zip(candidate_outputs, base_outputs, strict=True)
        ):
            if candidate.shape != base.shape or candidate.dtype != base.dtype:
                differences.append(
                    f"graph {graph_index} output {index} interface: "
                    f"candidate={candidate.shape}/{candidate.dtype}, "
                    f"base={base.shape}/{base.dtype}"
                )
                continue
            if not np.allclose(candidate, base, rtol=case.rtol, atol=case.atol):
                max_difference = float(
                    np.max(np.abs(candidate.astype(np.float64) - base.astype(np.float64)))
                )
                differences.append(
                    f"graph {graph_index} output {index} values: max_difference={max_difference:.6e}, "
                    f"rtol={case.rtol}, atol={case.atol}"
                )

    if differences:
        _report_regression(case, "; ".join(differences))
