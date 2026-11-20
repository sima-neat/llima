from dataclasses import dataclass
import logging
import multiprocessing
import os
from pathlib import Path
import time

import pytest

pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


@dataclass
class _CompilationJob:
    kind: str
    root: Path

    @property
    def model_name(self):
        return self.kind

    def get_gen_file_name(self, mode):
        return self.root / f"{self.kind}.sima"

    def gen_files(self, mode, **kwargs):
        if self.kind == "slow":
            (self.root / "started").touch()
            time.sleep(15)
        elif self.kind in ("fail", "interrupt", "crash"):
            deadline = time.monotonic() + 10
            while not (self.root / "started").exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("The other worker did not start")
                time.sleep(0.01)
            if self.kind == "interrupt":
                raise KeyboardInterrupt("original interrupt")
            if self.kind == "crash":
                os._exit(17)
            raise ValueError("original failure")
        self.get_gen_file_name(mode).touch()
        return True


def test_parallel_compilation_stops_failed_workers_and_can_restart(tmp_path):
    # Import compiler code only in the parent; spawned jobs stay lightweight.
    from concurrent.futures.process import BrokenProcessPool
    from sima_lmm.model.base import BaseModel

    for kind, error in (("fail", ValueError), ("interrupt", KeyboardInterrupt),
                        ("crash", BrokenProcessPool), ("success", None)):
        root = tmp_path / kind
        root.mkdir()
        jobs = ([_CompilationJob("one", root), _CompilationJob("two", root)]
                if error is None else [_CompilationJob("slow", root), _CompilationJob(kind, root)])
        start = time.monotonic()
        if error is None:
            BaseModel.gen_files_from_model_list(None, [(job, {}) for job in jobs], None, 2, logging.ERROR, False)
            assert all(job.get_gen_file_name(None).is_file() for job in jobs)
        else:
            with pytest.raises(error) as caught:
                BaseModel.gen_files_from_model_list(None, [(job, {}) for job in jobs], None, 2, logging.ERROR, False)
            if kind != "crash":
                assert str(caught.value) == f"original {'failure' if kind == 'fail' else 'interrupt'}"
        assert time.monotonic() - start < 12
        assert not multiprocessing.active_children()


@dataclass
class _PatchJob:
    name: str
    root: Path
    weight_patch_record_dir: Path | None = None
    weight_patch_template_dir: Path | None = None

    @property
    def model_name(self):
        return self.name

    def get_gen_file_name(self, mode):
        return self.root / f"{self.name}.sima"

    def gen_files(self, mode, **kwargs):
        if self.weight_patch_record_dir is not None:
            time.sleep(1)
            self.weight_patch_record_dir.mkdir()
        if self.weight_patch_template_dir is not None:
            # The model recording the template must have finished before this one starts.
            assert self.weight_patch_template_dir.is_dir()
        self.get_gen_file_name(mode).touch()
        return True


@pytest.mark.parametrize("num_processes", [1, 3])
def test_weight_patched_models_start_after_their_template(tmp_path, num_processes):
    from sima_lmm.model.base import BaseModel

    template = tmp_path / "template"
    jobs = [
        _PatchJob("patched_0", tmp_path, weight_patch_template_dir=template),
        _PatchJob("independent", tmp_path),
        _PatchJob("template", tmp_path, weight_patch_record_dir=template),
        _PatchJob("patched_1", tmp_path, weight_patch_template_dir=template),
    ]
    BaseModel.gen_files_from_model_list(
        None, [(job, {}) for job in jobs], None, num_processes, logging.ERROR, False
    )
    assert all(job.get_gen_file_name(None).is_file() for job in jobs)


@pytest.mark.parametrize(
    "num_dependents, num_processes, cap",
    [(30, 20, 15), (31, 20, 16), (20, 20, 20), (5, 20, 5), (30, 1, 1), (40, 20, 20)],
)
def test_patched_concurrency_cap_keeps_the_number_of_waves(num_dependents, num_processes, cap):
    from sima_lmm.model.base import _patched_concurrency_cap

    assert _patched_concurrency_cap(num_dependents, num_processes) == cap
    waves = -(-num_dependents // num_processes)
    assert -(-num_dependents // cap) == waves


@dataclass
class _TimedPatchJob:
    name: str
    root: Path
    weight_patch_record_dir: Path | None = None
    weight_patch_template_dir: Path | None = None

    @property
    def model_name(self):
        return self.name

    def get_gen_file_name(self, mode):
        return self.root / f"{self.name}.sima"

    def gen_files(self, mode, **kwargs):
        start = time.monotonic()
        if self.weight_patch_record_dir is not None:
            self.weight_patch_record_dir.mkdir()
        time.sleep(0.5)
        self.get_gen_file_name(mode).write_text(f"{start} {time.monotonic()}")
        return True


def test_patched_models_run_at_most_cap_at_a_time(tmp_path):
    from sima_lmm.model.base import BaseModel, _patched_concurrency_cap

    num_processes, num_dependents = 4, 5
    template = tmp_path / "template"
    dependents = [
        _TimedPatchJob(f"patched_{i}", tmp_path, weight_patch_template_dir=template)
        for i in range(num_dependents)
    ]
    jobs = [_TimedPatchJob("template", tmp_path, weight_patch_record_dir=template), *dependents]
    BaseModel.gen_files_from_model_list(
        None, [(job, {}) for job in jobs], None, num_processes, logging.ERROR, False
    )
    spans = [
        tuple(map(float, job.get_gen_file_name(None).read_text().split())) for job in dependents
    ]
    most_at_once = max(sum(s <= t < e for s, e in spans) for t, _ in spans)
    assert most_at_once <= _patched_concurrency_cap(num_dependents, num_processes) == 3


@dataclass
class _GraphModel:
    model_name: str
    _layer_base_name: str
    graph_dir: Path

    def get_gen_file_name(self, mode):
        return self.graph_dir / f"{self.model_name}.sima"


def test_graph_fingerprint_masks_layer_names_only(tmp_path):
    from sima_lmm.model.language_model import _graph_fingerprint

    def model(layer, extra_input=""):
        m = _GraphModel(f"lm_n1_post_layer{layer}", f"model.layers.{layer}", tmp_path)
        graph = (
            f'{{"name": "{m.model_name}", "weights": ["{m._layer_base_name}.mlp.up_proj"]'
            f'{extra_input}}}'
        )
        Path(f"{m.get_gen_file_name(None)}.json").write_text(graph)
        return m

    plain = [_graph_fingerprint(model(layer)) for layer in (3, 4, 17)]
    assert plain[0] is not None and len(set(plain)) == 1
    assert _graph_fingerprint(model(0, ', "inputs": ["input_scale"]')) != plain[0]
    missing = _GraphModel("lm_n1_post_layer9", "model.layers.9", tmp_path / "absent")
    assert _graph_fingerprint(missing) is None
