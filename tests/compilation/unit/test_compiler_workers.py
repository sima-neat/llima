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
