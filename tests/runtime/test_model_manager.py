import hashlib
import threading
from pathlib import Path

import pytest

from sima_lmm.devkit import model_manager
from sima_lmm.devkit.model_manager import (
    ArtifactInfo,
    ArtifactDownloader,
    DownloadCancellation,
    DownloadConfig,
    DownloadCancelled,
    HuggingFaceClient,
    INCOMPLETE_MARKER,
    LocalModelStore,
    ModelManager,
    ModelSnapshot,
    RetryableDownloadError,
)


pytestmark = pytest.mark.no_dispatcher

MODEL_NAME = "test-model"
REVISION = "a" * 40
TEST_MODELS_ENV_VAR = "LLIMA_TEST_MODELS_PATH"


class FakeHuggingFaceClient:
    organization = "simaai"

    def __init__(self, contents: dict[str, bytes]) -> None:
        self._contents = contents

    def snapshot(self, _model_name: str) -> ModelSnapshot:
        return ModelSnapshot(
            revision=REVISION,
            artifacts=tuple(
                ArtifactInfo(
                    filename=filename,
                    size=len(content),
                    checksum=hashlib.sha256(content).hexdigest(),
                    checksum_type="sha256",
                )
                for filename, content in self._contents.items()
            ),
        )


def _manager(
    monkeypatch,
    root: Path,
    contents: dict[str, bytes],
    *,
    workers: int = 1,
    max_attempts: int = 4,
) -> tuple[ModelManager, ArtifactDownloader, LocalModelStore]:
    monkeypatch.delenv(TEST_MODELS_ENV_VAR, raising=False)
    client = FakeHuggingFaceClient(contents)
    config = DownloadConfig(
        workers=workers,
        max_attempts=max_attempts,
        retry_base_seconds=0.0,
        retry_max_seconds=0.0,
    )
    downloader = ArtifactDownloader(client, config)
    store = LocalModelStore(
        primary_root=root,
        env_var=TEST_MODELS_ENV_VAR,
        organization=client.organization,
    )
    return ModelManager(client, downloader, store), downloader, store


def _write_artifact(
    destination: Path,
    content: bytes,
    progress,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    progress(len(content))


def test_pull_downloads_artifacts_concurrently(monkeypatch, tmp_path):
    contents = {
        "devkit/config.json": b"config",
        "elf_files/model-0.elf": b"model-0",
        "elf_files/model-1.elf": b"model-1",
        "tokenizer.json": b"tokenizer",
    }
    manager, downloader, _store = _manager(
        monkeypatch, tmp_path, contents, workers=2
    )
    barrier = threading.Barrier(2, timeout=2)
    state_lock = threading.Lock()
    active_downloads = 0
    max_active_downloads = 0

    def fetch_artifact(
        _model_name,
        _revision,
        destination,
        artifact,
        progress,
        _cancellation,
    ):
        nonlocal active_downloads, max_active_downloads
        with state_lock:
            active_downloads += 1
            max_active_downloads = max(max_active_downloads, active_downloads)
        try:
            barrier.wait()
            _write_artifact(destination, contents[artifact.filename], progress)
        finally:
            with state_lock:
                active_downloads -= 1

    monkeypatch.setattr(downloader, "_fetch_artifact", fetch_artifact)

    model_dir = manager.pull(MODEL_NAME)

    assert max_active_downloads == 2
    assert not (model_dir / INCOMPLETE_MARKER).exists()
    for filename, content in contents.items():
        assert (model_dir / filename).read_bytes() == content


def test_pull_retry_reuses_completed_artifacts(monkeypatch, tmp_path):
    contents = {
        "devkit/config.json": b"config-data",
        "elf_files/model.elf": b"model-data",
        "tokenizer.json": b"x",
    }
    manager, downloader, _store = _manager(monkeypatch, tmp_path, contents)
    first_attempt: list[str] = []

    def fail_last_artifact(
        _model_name,
        _revision,
        destination,
        artifact,
        progress,
        _cancellation,
    ):
        first_attempt.append(artifact.filename)
        if artifact.filename == "tokenizer.json":
            raise RuntimeError("injected download failure")
        _write_artifact(destination, contents[artifact.filename], progress)

    monkeypatch.setattr(downloader, "_fetch_artifact", fail_last_artifact)

    with pytest.raises(RuntimeError, match="injected download failure"):
        manager.pull(MODEL_NAME)

    model_dir = tmp_path / MODEL_NAME
    assert first_attempt == list(contents)
    assert (model_dir / INCOMPLETE_MARKER).exists()
    assert (model_dir / "devkit/config.json").read_bytes() == b"config-data"
    assert (model_dir / "elf_files/model.elf").read_bytes() == b"model-data"

    retry_downloads: list[str] = []

    def download_missing_artifact(
        _model_name,
        _revision,
        destination,
        artifact,
        progress,
        _cancellation,
    ):
        retry_downloads.append(artifact.filename)
        _write_artifact(destination, contents[artifact.filename], progress)

    monkeypatch.setattr(
        downloader,
        "_fetch_artifact",
        download_missing_artifact,
    )

    assert manager.pull(MODEL_NAME) == model_dir
    assert retry_downloads == ["tokenizer.json"]
    assert not (model_dir / INCOMPLETE_MARKER).exists()


def test_pull_retries_transient_artifact_failures(monkeypatch, tmp_path):
    contents = {
        "devkit/config.json": b"config",
        "elf_files/model.elf": b"model",
    }
    manager, downloader, _store = _manager(monkeypatch, tmp_path, contents)
    attempts: dict[str, int] = {}

    def fetch_artifact(
        _model_name,
        _revision,
        destination,
        artifact,
        progress,
        _cancellation,
    ):
        attempts[artifact.filename] = attempts.get(artifact.filename, 0) + 1
        should_fail = (
            artifact.filename == "devkit/config.json"
            and attempts[artifact.filename] < 3
        )
        if should_fail:
            progress(1)
            raise RetryableDownloadError("injected short response")
        _write_artifact(destination, contents[artifact.filename], progress)

    monkeypatch.setattr(downloader, "_fetch_artifact", fetch_artifact)

    model_dir = manager.pull(MODEL_NAME)

    assert model_dir == tmp_path / MODEL_NAME
    assert attempts["devkit/config.json"] == 3
    assert attempts["elf_files/model.elf"] == 1


def test_pull_schedules_largest_artifacts_first(monkeypatch, tmp_path):
    contents = {
        "devkit/config.json": b"x",
        "elf_files/model.elf": b"12345",
        "tokenizer.json": b"123",
    }
    manager, downloader, _store = _manager(monkeypatch, tmp_path, contents)
    download_order: list[str] = []

    def fetch_artifact(
        _model_name,
        _revision,
        destination,
        artifact,
        progress,
        _cancellation,
    ):
        download_order.append(artifact.filename)
        _write_artifact(destination, contents[artifact.filename], progress)

    monkeypatch.setattr(downloader, "_fetch_artifact", fetch_artifact)

    manager.pull(MODEL_NAME)

    assert download_order == [
        "elf_files/model.elf",
        "tokenizer.json",
        "devkit/config.json",
    ]


def test_pull_cancels_other_workers_on_interrupt(monkeypatch, tmp_path):
    contents = {
        "devkit/config.json": b"config",
        "elf_files/model.elf": b"model",
    }
    manager, downloader, _store = _manager(
        monkeypatch, tmp_path, contents, workers=2
    )
    barrier = threading.Barrier(2, timeout=2)
    waiting_worker_stopped = threading.Event()

    def fetch_artifact(
        _model_name,
        _revision,
        _destination,
        artifact,
        _progress,
        cancellation,
    ):
        barrier.wait()
        if artifact.filename == "devkit/config.json":
            raise KeyboardInterrupt
        try:
            cancellation.wait(5)
        finally:
            waiting_worker_stopped.set()

    monkeypatch.setattr(downloader, "_fetch_artifact", fetch_artifact)

    with pytest.raises(KeyboardInterrupt):
        manager.pull(MODEL_NAME)

    assert waiting_worker_stopped.wait(1)


def test_download_cancellation_closes_active_response(monkeypatch, tmp_path):
    read_started = threading.Event()
    response_closed = threading.Event()

    class BlockingResponse:
        status = 200
        headers = {"Content-Length": "1"}

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            self.close()

        def read(self, _size):
            read_started.set()
            response_closed.wait(2)
            return b""

        def close(self):
            response_closed.set()

    response = BlockingResponse()
    monkeypatch.setattr(
        model_manager.urllib.request,
        "urlopen",
        lambda _url, timeout: response,
    )
    client = HuggingFaceClient()
    downloader = ArtifactDownloader(client)
    artifact = ArtifactInfo(
        filename="model.elf",
        size=1,
        checksum=hashlib.sha256(b"x").hexdigest(),
        checksum_type="sha256",
    )
    cancellation = DownloadCancellation()
    errors: list[BaseException] = []

    def download() -> None:
        try:
            downloader._fetch_artifact(
                MODEL_NAME,
                REVISION,
                tmp_path / "model.elf",
                artifact,
                lambda _size: None,
                cancellation,
            )
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=download)
    worker.start()
    assert read_started.wait(1)

    cancellation.cancel()

    worker.join(timeout=1)
    assert response_closed.is_set()
    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], DownloadCancelled)


def test_model_lock_serializes_access(monkeypatch, tmp_path):
    monkeypatch.delenv(TEST_MODELS_ENV_VAR, raising=False)
    store = LocalModelStore(tmp_path, TEST_MODELS_ENV_VAR)
    second_lock_acquired = threading.Event()

    def acquire_same_lock() -> None:
        with store.lock_model(tmp_path, MODEL_NAME):
            second_lock_acquired.set()

    with store.lock_model(tmp_path, MODEL_NAME):
        waiter = threading.Thread(target=acquire_same_lock)
        waiter.start()
        assert not second_lock_acquired.wait(0.05)

    assert second_lock_acquired.wait(1)
    waiter.join(timeout=1)
    assert not waiter.is_alive()


def test_snapshot_payload_is_validated_and_converted(monkeypatch):
    client = HuggingFaceClient()
    payload = {
        "sha": REVISION,
        "siblings": [
            {
                "rfilename": "devkit/config.json",
                "size": 6,
                "lfs": {"sha256": hashlib.sha256(b"config").hexdigest()},
            }
        ],
    }
    monkeypatch.setattr(client, "_fetch_json", lambda _url: payload)

    snapshot = client.snapshot(MODEL_NAME)

    assert snapshot.revision == REVISION
    assert snapshot.artifacts == (
        ArtifactInfo(
            "devkit/config.json",
            6,
            hashlib.sha256(b"config").hexdigest(),
            "sha256",
        ),
    )

    payload["siblings"][0]["size"] = True
    with pytest.raises(RuntimeError, match="valid size"):
        client.snapshot(MODEL_NAME)


def test_incomplete_model_is_hidden_but_removable(monkeypatch, tmp_path):
    monkeypatch.delenv(TEST_MODELS_ENV_VAR, raising=False)
    store = LocalModelStore(tmp_path, TEST_MODELS_ENV_VAR)
    model_dir = tmp_path / MODEL_NAME
    (model_dir / "devkit").mkdir(parents=True)
    (model_dir / "elf_files").mkdir()
    (model_dir / INCOMPLETE_MARKER).touch()

    assert store.resolve_runnable(MODEL_NAME) is None
    assert store.iter_installed() == []
    assert store.remove(MODEL_NAME)
    assert not model_dir.exists()
