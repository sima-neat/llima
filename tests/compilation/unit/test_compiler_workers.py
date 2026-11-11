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
