from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from sima_lmm.model.model_graph import ModelGraph
from tests.compilation.cases import GraphRegressionCase
from tests.compilation.graph_regression import generate
from tests.compilation.graph_regression import test_branch_regression as regression


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


def _save_graph(root, name, input_name="x", tokens=1, offset=0.0):
    root.mkdir(exist_ok=True)
    model = SimpleNamespace(model_name=name, sima_model_sdk_path=root)
    graph = ModelGraph(model, {input_name: (1, 1, tokens, 16)}, quantizable=True)
    output = graph.add(
        graph.inputs[input_name], graph.constant(np.array([offset], np.float32))
    )
    graph.save([output])
    return f"{name}.fp32.sima"


def _manifest(case, paths, layers=None):
    entry = {"status": "available", "graph_paths": paths}
    if layers is not None:
        entry["layer_indices"] = layers
    return {"cases": {case.id: entry}}


@pytest.mark.parametrize("validation_mode", ["candidate-only", "compare"])
def test_regression_executes_vision_layers_independently(tmp_path, validation_mode):
    case = GraphRegressionCase("synthetic", "vision")
    candidate_root, base_root = tmp_path / "candidate", tmp_path / "base"
    paths = [
        _save_graph(candidate_root, "first"),
        _save_graph(candidate_root, "middle", "hidden", 2),
    ]
    base_paths = [
        _save_graph(base_root, "first"),
        _save_graph(base_root, "middle", "hidden", 2),
    ]
    regression.test_branch_relative_graph_regression(
        case,
        validation_mode,
        candidate_root,
        _manifest(case, paths, [0, 2]),
        base_root,
        _manifest(case, base_paths, [0, 2]),
    )


def test_informative_regression_requires_finite_candidate_outputs(tmp_path):
    case = GraphRegressionCase("synthetic", "pre", mode="informative")
    path = _save_graph(tmp_path, "invalid", offset=np.inf)
    with pytest.raises(AssertionError, match="Non-finite output"):
        regression.test_branch_relative_graph_regression(
            case, "candidate-only", tmp_path, _manifest(case, [path]), None, None,
        )


@pytest.mark.parametrize("mode", ["required", "informative"])
@pytest.mark.parametrize("failure", ["unavailable", "values", "layers"])
def test_regression_enforces_baseline_policy(tmp_path, mode, failure):
    case = GraphRegressionCase("synthetic", "vision", mode=mode)
    candidate_root, base_root = tmp_path / "candidate", tmp_path / "base"
    candidate_path = _save_graph(candidate_root, "candidate")
    base_path = _save_graph(
        base_root, "base", offset=1.0 if failure == "values" else 0.0
    )
    candidate_manifest = _manifest(case, [candidate_path], [0])
    base_manifest = _manifest(case, [base_path], [1] if failure == "layers" else [0])
    if failure == "unavailable":
        base_manifest["cases"][case.id] = {
            "status": "unavailable", "reason": "No baseline support"
        }
    expected = (
        pytest.raises(pytest.fail.Exception)
        if mode == "required" else pytest.warns(UserWarning)
    )
    with expected:
        regression.test_branch_relative_graph_regression(
            case, "compare", candidate_root, candidate_manifest, base_root, base_manifest,
        )


def test_vision_regression_selects_same_intermediate_layer_for_both_revisions(monkeypatch):
    case = GraphRegressionCase("synthetic", "vision")
    source = SimpleNamespace(
        cfg=SimpleNamespace(num_vision_layers=26),
        vision_model_name="vision",
        sima_path="unused",
        hf_model=None,
    )
    vision = Mock(cfg=source.cfg)
    vision._get_part_model.side_effect = lambda index: SimpleNamespace(layer_idx=index)
    monkeypatch.setattr(generate, "VisionModel", Mock(return_value=vision))
    first, _ = generate._standard_models(case, source)
    second, _ = generate._standard_models(case, source)
    assert [model.layer_idx for model in first] == [model.layer_idx for model in second]
    assert first[0].layer_idx == 0
    assert 0 < first[1].layer_idx < source.cfg.num_vision_layers - 1
