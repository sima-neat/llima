import hashlib
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from tests.compilation.cases import ONNX_REGRESSION_CASES, OnnxRegressionCase
from tests.compilation.helpers.paths import require_readable_path


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_onnx_regression]

ACTIVE_CASES = tuple(
    case for case in ONNX_REGRESSION_CASES if case.mode != "disabled"
)


def _case_parameter(case: OnnxRegressionCase):
    marks = []
    if case.target_model_folder is not None:
        marks.extend((pytest.mark.serial, pytest.mark.high_memory))
    return pytest.param(case, id=case.id, marks=marks)


def _graph_signature(model: onnx.ModelProto) -> tuple:
    def value_signature(value_info):
        tensor_type = value_info.type.tensor_type
        dimensions = tuple(
            dimension.dim_param or dimension.dim_value
            for dimension in tensor_type.shape.dim
        )
        return value_info.name, tensor_type.elem_type, dimensions

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = tuple(
        value_signature(value)
        for value in model.graph.input
        if value.name not in initializer_names
    )
    outputs = tuple(value_signature(value) for value in model.graph.output)
    return inputs, outputs


def _validate_graph(path: Path) -> tuple:
    onnx.checker.check_model(str(path))
    model = onnx.load(str(path), load_external_data=False)
    signature = _graph_signature(model)
    inputs, outputs = signature
    assert inputs, f"ONNX has no runtime inputs: {path}"
    assert outputs, f"ONNX has no runtime outputs: {path}"
    assert len({item[0] for item in inputs}) == len(inputs)
    assert len({item[0] for item in outputs}) == len(outputs)
    assert all(item[1] != onnx.TensorProto.UNDEFINED for item in inputs + outputs)
    return signature


def _input_shape(dimensions: tuple) -> tuple[int, ...]:
    return tuple(
        dimension if isinstance(dimension, int) and dimension > 0 else 1
        for dimension in dimensions
    )


def _make_inputs(case: OnnxRegressionCase, signature: tuple) -> dict[str, np.ndarray]:
    digest = hashlib.sha256(case.id.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    feeds = {}
    for name, elem_type, dimensions in signature[0]:
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type))
        shape = _input_shape(dimensions)
        if np.issubdtype(dtype, np.floating):
            value = rng.uniform(-1.0, 1.0, shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            value = rng.integers(0, 4, shape, dtype=dtype)
        elif np.issubdtype(dtype, np.bool_):
            value = rng.integers(0, 2, shape).astype(dtype)
        else:
            raise TypeError(f"Unsupported ONNX input dtype for {case.id}: {dtype}")
        feeds[name] = value
    return feeds


def _run_onnx(path: Path, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    return session.run([], feeds)


def _case_path(root: Path, manifest: dict, case: OnnxRegressionCase) -> Path:
    manifest_case = manifest.get("cases", {}).get(case.id)
    if manifest_case is None:
        raise KeyError(f"Generated ONNX manifest is missing required case {case.id}")
    return require_readable_path(
        root / manifest_case["onnx_path"],
        f"generated ONNX for {case.id}",
    )


def _report_regression(case: OnnxRegressionCase, message: str) -> None:
    if case.mode == "required":
        pytest.fail(message)
    warnings.warn(f"Informative ONNX regression for {case.id}: {message}", stacklevel=2)


@pytest.mark.parametrize("case", [_case_parameter(case) for case in ACTIVE_CASES])
def test_branch_relative_onnx_regression(
    case: OnnxRegressionCase,
    candidate_onnx_root: Path,
    candidate_onnx_manifest: dict,
    base_onnx_root: Path,
    base_onnx_manifest: dict,
):
    candidate_path = _case_path(
        candidate_onnx_root, candidate_onnx_manifest, case
    )
    base_path = _case_path(base_onnx_root, base_onnx_manifest, case)

    candidate_signature = _validate_graph(candidate_path)
    base_signature = _validate_graph(base_path)
    if candidate_signature != base_signature:
        _report_regression(
            case,
            "ONNX input/output interface differs between candidate and base: "
            f"candidate={candidate_signature}, base={base_signature}",
        )
        return

    feeds = _make_inputs(case, base_signature)
    candidate_outputs = _run_onnx(candidate_path, feeds)
    base_outputs = _run_onnx(base_path, feeds)
    if len(candidate_outputs) != len(base_outputs):
        _report_regression(
            case,
            "ONNX output count differs: "
            f"candidate={len(candidate_outputs)}, base={len(base_outputs)}",
        )
        return

    differences = []
    for index, (candidate, base) in enumerate(
        zip(candidate_outputs, base_outputs, strict=True)
    ):
        if candidate.shape != base.shape or candidate.dtype != base.dtype:
            differences.append(
                f"output {index} interface: candidate={candidate.shape}/{candidate.dtype}, "
                f"base={base.shape}/{base.dtype}"
            )
            continue
        if not np.allclose(
            candidate,
            base,
            rtol=case.rtol,
            atol=case.atol,
            equal_nan=True,
        ):
            max_difference = float(
                np.max(np.abs(candidate.astype(np.float64) - base.astype(np.float64)))
            )
            differences.append(
                f"output {index} values: max_difference={max_difference:.6e}, "
                f"rtol={case.rtol}, atol={case.atol}"
            )

    if differences:
        _report_regression(case, "; ".join(differences))
