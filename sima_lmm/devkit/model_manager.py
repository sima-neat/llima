import fcntl
import hashlib
import http.client
import json
import os
import random
import shutil
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Protocol,
    TypeVar,
    TypedDict,
    cast,
)


DEFAULT_HF_ORG = "simaai"
DEFAULT_HF_API_BASE = "https://huggingface.co/api"
DEFAULT_HF_RESOLVE_BASE = "https://huggingface.co"
DEFAULT_HF_COLLECTIONS = (
    "large-language-models",
    "vision-language-models",
    "speech-and-audio-models",
)
DEFAULT_MODELS_ROOT = Path("/media/nvme/llima/models")
DEFAULT_MODELS_ENV_VAR = "LLIMA_MODELS_PATH"
INCOMPLETE_MARKER = ".llima-incomplete"
RETRYABLE_HTTP_STATUS = frozenset((408, 429))

ChecksumType = Literal["sha256", "git-sha1"]
T = TypeVar("T")


class _HFLfsPayload(TypedDict, total=False):
    sha256: str


class _HFArtifactPayload(TypedDict, total=False):
    rfilename: str
    size: int
    blobId: str
    lfs: _HFLfsPayload


class _HFModelPayload(TypedDict, total=False):
    sha: str
    siblings: list[_HFArtifactPayload]


class _HFSearchItemPayload(TypedDict, total=False):
    modelId: str
    downloads: int
    likes: int


class _HFSearchPagePayload(TypedDict, total=False):
    models: list[_HFSearchItemPayload]
    next: str


class _HFCollectionItemPayload(TypedDict, total=False):
    type: str
    id: str


class _HFCollectionPayload(TypedDict, total=False):
    items: list[_HFCollectionItemPayload]


class _Hasher(Protocol):
    def update(self, data: bytes) -> None: ...

    def hexdigest(self) -> str: ...


class _Closable(Protocol):
    def close(self) -> None: ...


class _StreamCancellation(Protocol):
    def raise_if_cancelled(self) -> None: ...

    def register(self, response: object) -> None: ...

    def unregister(self, response: object) -> None: ...


class RetryableDownloadError(RuntimeError):
    pass


class DownloadCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    downloads: int | None = None
    likes: int | None = None


@dataclass(frozen=True)
class ArtifactInfo:
    filename: str
    size: int
    checksum: str
    checksum_type: ChecksumType


@dataclass(frozen=True)
class ModelSnapshot:
    revision: str
    artifacts: tuple[ArtifactInfo, ...]


@dataclass(frozen=True)
class DownloadConfig:
    chunk_size: int = 8 * 1024 * 1024
    workers: int = 8
    max_attempts: int = 4
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 30.0
    io_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.workers <= 0:
            raise ValueError("workers must be greater than zero")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        if self.retry_base_seconds < 0 or self.retry_max_seconds < 0:
            raise ValueError("retry delays must not be negative")
        if self.io_timeout_seconds <= 0:
            raise ValueError("io_timeout_seconds must be greater than zero")


@dataclass(frozen=True)
class ArtifactDownload:
    artifact: ArtifactInfo
    staging_path: Path
    destination_path: Path


class ArtifactStream:
    def __init__(
        self,
        response: object,
        chunk_size: int,
        cancellation: _StreamCancellation,
    ) -> None:
        self._response = response
        self._chunk_size = chunk_size
        self._cancellation = cancellation

    @property
    def status(self) -> int:
        return int(getattr(self._response, "status"))

    @property
    def content_length(self) -> int | None:
        headers = getattr(self._response, "headers")
        value = headers.get("Content-Length")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def iter_chunks(self) -> Iterator[bytes]:
        read = getattr(self._response, "read")
        while True:
            self._cancellation.raise_if_cancelled()
            chunk = read(self._chunk_size)
            self._cancellation.raise_if_cancelled()
            if not chunk:
                return
            yield chunk


class HuggingFaceClient:
    def __init__(
        self,
        organization: str = DEFAULT_HF_ORG,
        api_base: str = DEFAULT_HF_API_BASE,
        resolve_base: str = DEFAULT_HF_RESOLVE_BASE,
        collections: tuple[str, ...] = DEFAULT_HF_COLLECTIONS,
    ) -> None:
        self.organization = organization
        self._api_base = api_base.rstrip("/")
        self._resolve_base = resolve_base.rstrip("/")
        self._collections = collections

    def search(self, term: str) -> list[ModelInfo]:
        results: list[ModelInfo] = []
        cursor: str | None = None
        while True:
            params = {"author": self.organization}
            if term:
                params["search"] = term
            if cursor:
                params["cursor"] = cursor
            url = f"{self._api_base}/models?{urllib.parse.urlencode(params)}"
            data, headers = self._fetch_json_with_headers(url)
            items, cursor = self._parse_search_page(data, headers)
            results.extend(self._parse_search_item(item) for item in items)
            if not cursor:
                break

        allowed_ids = self._collection_model_ids()
        return [info for info in results if info.model_id in allowed_ids]

    def snapshot(self, model_name: str) -> ModelSnapshot:
        encoded_name = urllib.parse.quote(model_name, safe="")
        url = (
            f"{self._api_base}/models/{self.organization}/"
            f"{encoded_name}?blobs=true"
        )
        data = self._fetch_json(url)
        if not isinstance(data, dict):
            raise RuntimeError(f"Invalid model metadata for '{model_name}'.")
        revision = data.get("sha")
        siblings = data.get("siblings")
        if not isinstance(revision, str) or not revision:
            raise RuntimeError(
                f"Model metadata is missing a revision for '{model_name}'."
            )
        if not isinstance(siblings, list):
            raise RuntimeError(
                f"Model metadata is missing artifacts for '{model_name}'."
            )

        artifacts: list[ArtifactInfo] = []
        filenames: set[str] = set()
        for entry in siblings:
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"Invalid artifact metadata for '{model_name}'."
                )
            artifact = self._parse_artifact(entry)
            if artifact.filename in filenames:
                raise RuntimeError(
                    "Duplicate artifact in model metadata: "
                    f"'{artifact.filename}'"
                )
            filenames.add(artifact.filename)
            artifacts.append(artifact)
        return ModelSnapshot(revision, tuple(artifacts))

    @contextmanager
    def open_artifact_stream(
        self,
        model_name: str,
        revision: str,
        filename: str,
        *,
        chunk_size: int,
        timeout: float,
        cancellation: _StreamCancellation,
    ) -> Iterator[ArtifactStream]:
        encoded_name = urllib.parse.quote(model_name, safe="")
        encoded_revision = urllib.parse.quote(revision, safe="")
        encoded_filename = urllib.parse.quote(filename, safe="/")
        url = (
            f"{self._resolve_base}/{self.organization}/{encoded_name}/"
            f"resolve/{encoded_revision}/{encoded_filename}"
        )
        cancellation.raise_if_cancelled()
        response_context = urllib.request.urlopen(url, timeout=timeout)
        with response_context as response:
            cancellation.register(response)
            try:
                yield ArtifactStream(response, chunk_size, cancellation)
            finally:
                cancellation.unregister(response)

    def strip_organization(self, model_id: str) -> str:
        prefix = f"{self.organization}/"
        if model_id.startswith(prefix):
            return model_id[len(prefix):]
        return model_id

    def _collection_model_ids(self) -> set[str]:
        ids: set[str] = set()
        for slug in self._collections:
            url = (
                f"{self._api_base}/collections/"
                f"{self.organization}/{urllib.parse.quote(slug, safe='')}"
            )
            data = self._fetch_json(url)
            if not isinstance(data, dict):
                raise RuntimeError(
                    f"Invalid collection metadata for '{slug}'."
                )
            items = data.get("items")
            if not isinstance(items, list):
                raise RuntimeError(
                    f"Collection metadata is missing items for '{slug}'."
                )
            for item in items:
                if not isinstance(item, dict) or item.get("type") != "model":
                    continue
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id:
                    ids.add(self.strip_organization(model_id))
        return ids

    def _parse_search_page(
        self,
        data: object,
        headers: dict[str, str],
    ) -> tuple[list[dict[str, object]], str | None]:
        if isinstance(data, dict):
            raw_items = data.get("models", [])
            raw_cursor = data.get("next")
            cursor = (
                raw_cursor
                if isinstance(raw_cursor, str) and raw_cursor
                else self._parse_next_cursor(headers)
            )
        elif isinstance(data, list):
            raw_items = data
            cursor = self._parse_next_cursor(headers)
        else:
            raise RuntimeError("Invalid model search response.")
        if not isinstance(raw_items, list) or not all(
            isinstance(item, dict) for item in raw_items
        ):
            raise RuntimeError("Invalid model search results.")
        return raw_items, cursor

    def _parse_search_item(self, item: dict[str, object]) -> ModelInfo:
        model_id = item.get("modelId")
        if not isinstance(model_id, str) or not model_id:
            raise RuntimeError("Model search result is missing a model ID.")
        return ModelInfo(
            model_id=self.strip_organization(model_id),
            downloads=self._optional_non_negative_int(
                item.get("downloads"), "downloads", model_id
            ),
            likes=self._optional_non_negative_int(
                item.get("likes"), "likes", model_id
            ),
        )

    @staticmethod
    def _optional_non_negative_int(
        value: object,
        field: str,
        model_id: str,
    ) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError(
                f"Model search result has invalid {field} for '{model_id}'."
            )
        return value

    @staticmethod
    def _parse_artifact(entry: dict[str, object]) -> ArtifactInfo:
        filename = entry.get("rfilename")
        size = entry.get("size")
        if not isinstance(filename, str) or not filename:
            raise RuntimeError(
                "Model metadata contains an artifact without a filename."
            )
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RuntimeError(
                f"Model metadata is missing a valid size for '{filename}'."
            )
        lfs = entry.get("lfs")
        if isinstance(lfs, dict) and isinstance(lfs.get("sha256"), str):
            checksum = lfs["sha256"].lower()
            checksum_type: ChecksumType = "sha256"
            checksum_length = 64
        else:
            checksum = str(entry.get("blobId", "")).lower()
            checksum_type = "git-sha1"
            checksum_length = 40
        if len(checksum) != checksum_length or any(
            character not in "0123456789abcdef" for character in checksum
        ):
            raise RuntimeError(
                f"Model metadata is missing a valid checksum for '{filename}'."
            )
        return ArtifactInfo(filename, size, checksum, checksum_type)

    @staticmethod
    def _parse_next_cursor(headers: dict[str, str]) -> str | None:
        for key in ("X-Next-Cursor", "X-Next-Page"):
            value = headers.get(key)
            if value:
                return value
        link = headers.get("Link")
        if not link:
            return None
        for part in link.split(","):
            if 'rel="next"' not in part or "cursor=" not in part:
                continue
            segment = part.split("cursor=", 1)[1]
            cursor = segment.split("&", 1)[0].strip(" >")
            if cursor:
                return cursor
        return None

    def _fetch_json_with_headers(
        self,
        url: str,
    ) -> tuple[object, dict[str, str]]:
        with urllib.request.urlopen(url) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} for {url}")
            payload = response.read()
            headers = dict(response.headers.items())
        return json.loads(payload.decode("utf-8")), headers

    def _fetch_json(self, url: str) -> object:
        data, _headers = self._fetch_json_with_headers(url)
        return data


class DownloadCancellation:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._responses: dict[int, _Closable] = {}
        self._lock = threading.Lock()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise DownloadCancelled("Model download cancelled.")

    def wait(self, timeout: float) -> None:
        if self._event.wait(timeout):
            raise DownloadCancelled("Model download cancelled.")

    def register(self, response: object) -> None:
        if not hasattr(response, "close"):
            raise TypeError("Download response must be closable.")
        closable = cast(_Closable, response)
        with self._lock:
            if self._event.is_set():
                closable.close()
                raise DownloadCancelled("Model download cancelled.")
            self._responses[id(response)] = closable

    def unregister(self, response: object) -> None:
        with self._lock:
            self._responses.pop(id(response), None)

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            responses = tuple(self._responses.values())
        for response in responses:
            try:
                response.close()
            except OSError:
                pass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    base_delay_seconds: float
    max_delay_seconds: float

    def run(
        self,
        operation: Callable[[int], T],
        *,
        cancellation: DownloadCancellation,
        on_retry: Callable[[int, float, BaseException], None],
    ) -> T:
        for attempt in range(1, self.max_attempts + 1):
            cancellation.raise_if_cancelled()
            try:
                return operation(attempt)
            except BaseException as error:
                if (
                    attempt == self.max_attempts
                    or not self._is_retryable(error)
                ):
                    raise
                delay = self._delay(error, attempt)
                on_retry(attempt, delay, error)
                cancellation.wait(delay)
        raise AssertionError(
            "Retry loop exhausted without returning or raising."
        )

    @staticmethod
    def _is_retryable(error: BaseException) -> bool:
        if isinstance(error, DownloadCancelled):
            return False
        if isinstance(error, urllib.error.HTTPError):
            return RetryPolicy._is_retryable_http_status(error.code)
        return isinstance(
            error,
            (
                RetryableDownloadError,
                urllib.error.URLError,
                http.client.IncompleteRead,
                http.client.RemoteDisconnected,
                TimeoutError,
                ConnectionError,
            ),
        )

    @staticmethod
    def _is_retryable_http_status(status: int) -> bool:
        return status in RETRYABLE_HTTP_STATUS or 500 <= status < 600

    def _delay(self, error: BaseException, attempt: int) -> float:
        retry_after: float | None = None
        if isinstance(error, urllib.error.HTTPError):
            value = error.headers.get("Retry-After") if error.headers else None
            if value is not None:
                try:
                    retry_after = max(0.0, float(value))
                except ValueError:
                    pass
        base_delay = min(
            self.max_delay_seconds,
            self.base_delay_seconds * (2 ** (attempt - 1)),
        )
        delay = retry_after if retry_after is not None else base_delay
        jitter = random.uniform(0.0, min(1.0, delay * 0.25))
        return min(self.max_delay_seconds, delay + jitter)


class _DownloadProgress:
    def __init__(
        self,
        artifacts: Iterable[ArtifactInfo],
        max_attempts: int,
    ) -> None:
        artifacts = tuple(artifacts)
        self._artifact_count = len(artifacts)
        self._total_bytes = sum(artifact.size for artifact in artifacts)
        self._max_attempts = max_attempts
        self._completed_artifacts = 0
        self._downloaded_bytes = 0
        self._printed = False
        self._lock = threading.Lock()

    def update(self, chunk_size: int) -> None:
        with self._lock:
            self._downloaded_bytes += chunk_size
            self._render()

    def discard(self, byte_count: int) -> None:
        with self._lock:
            self._downloaded_bytes = max(
                0, self._downloaded_bytes - byte_count
            )

    def retry(
        self,
        artifact: ArtifactInfo,
        attempt: int,
        delay: float,
        error: BaseException,
    ) -> None:
        with self._lock:
            if self._printed:
                print(flush=True)
            print(
                f"Retrying {artifact.filename} after attempt {attempt}/"
                f"{self._max_attempts} failed ({self._error_summary(error)}); "
                f"waiting {delay:.1f}s.",
                flush=True,
            )
            self._render()

    def artifact_complete(self) -> None:
        with self._lock:
            self._completed_artifacts += 1
            self._render()

    def finish(self) -> None:
        with self._lock:
            self._downloaded_bytes = self._total_bytes
            self._completed_artifacts = self._artifact_count
            self._render(end="\n")

    def abort(self) -> None:
        with self._lock:
            if self._printed:
                print(flush=True)

    def _render(self, end: str = "") -> None:
        if self._total_bytes:
            percent = min(
                100.0,
                self._downloaded_bytes / self._total_bytes * 100,
            )
            progress = f"{percent:5.1f}%"
        else:
            progress = "100.0%"
        print(
            f"\rDownloading model artifacts: {progress} "
            f"({self._completed_artifacts}/{self._artifact_count} files)",
            end=end,
            flush=True,
        )
        self._printed = True

    @staticmethod
    def _error_summary(error: BaseException) -> str:
        if isinstance(error, urllib.error.HTTPError):
            return f"HTTP {error.code}"
        if isinstance(error, urllib.error.URLError):
            return f"{type(error.reason).__name__}: {error.reason}"
        return f"{type(error).__name__}: {error}"


class ArtifactDownloader:
    def __init__(
        self,
        client: HuggingFaceClient,
        config: DownloadConfig | None = None,
    ) -> None:
        self._client = client
        self.config = config or DownloadConfig()
        self._retry_policy = RetryPolicy(
            max_attempts=self.config.max_attempts,
            base_delay_seconds=self.config.retry_base_seconds,
            max_delay_seconds=self.config.retry_max_seconds,
        )

    def download(
        self,
        model_name: str,
        revision: str,
        downloads: list[ArtifactDownload],
    ) -> None:
        downloads = sorted(
            downloads,
            key=lambda item: (-item.artifact.size, item.artifact.filename),
        )
        if not downloads:
            return
        progress = _DownloadProgress(
            (download.artifact for download in downloads),
            self.config.max_attempts,
        )
        cancellation = DownloadCancellation()

        def download_one(download: ArtifactDownload) -> None:
            artifact = download.artifact

            def operation(_attempt: int) -> None:
                attempt_bytes = 0

                def update_progress(chunk_size: int) -> None:
                    nonlocal attempt_bytes
                    attempt_bytes += chunk_size
                    progress.update(chunk_size)

                try:
                    self._fetch_artifact(
                        model_name,
                        revision,
                        download.staging_path,
                        artifact,
                        update_progress,
                        cancellation,
                    )
                except BaseException:
                    progress.discard(attempt_bytes)
                    raise

            self._retry_policy.run(
                operation,
                cancellation=cancellation,
                on_retry=lambda attempt, delay, error: progress.retry(
                    artifact, attempt, delay, error
                ),
            )
            download.destination_path.parent.mkdir(parents=True, exist_ok=True)
            download.staging_path.replace(download.destination_path)
            progress.artifact_complete()

        futures = []
        try:
            worker_count = min(self.config.workers, len(downloads))
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix=f"llima-pull-{model_name}",
            ) as executor:
                futures = [
                    executor.submit(download_one, download)
                    for download in downloads
                ]
                try:
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    cancellation.cancel()
                    for future in futures:
                        future.cancel()
                    raise
        except BaseException:
            progress.abort()
            raise
        progress.finish()

    def _fetch_artifact(
        self,
        model_name: str,
        revision: str,
        destination: Path,
        artifact: ArtifactInfo,
        progress: Callable[[int], None],
        cancellation: DownloadCancellation,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._client.open_artifact_stream(
            model_name,
            revision,
            artifact.filename,
            chunk_size=self.config.chunk_size,
            timeout=self.config.io_timeout_seconds,
            cancellation=cancellation,
        ) as stream:
            if stream.status != 200:
                error_type = (
                    RetryableDownloadError
                    if RetryPolicy._is_retryable_http_status(stream.status)
                    else RuntimeError
                )
                raise error_type(
                    f"HTTP {stream.status} while downloading "
                    f"'{artifact.filename}'"
                )
            if (
                stream.content_length is not None
                and stream.content_length != artifact.size
            ):
                raise RetryableDownloadError(
                    f"Invalid size for '{artifact.filename}': expected "
                    f"{artifact.size} bytes, server reported "
                    f"{stream.content_length} bytes"
                )

            downloaded = 0
            hasher = self.new_hasher(artifact.checksum_type, artifact.size)
            with open(destination, "wb") as artifact_file:
                for chunk in stream.iter_chunks():
                    artifact_file.write(chunk)
                    hasher.update(chunk)
                    downloaded += len(chunk)
                    progress(len(chunk))
            if downloaded != artifact.size:
                raise RetryableDownloadError(
                    f"Invalid size for '{artifact.filename}': expected "
                    f"{artifact.size} bytes, downloaded {downloaded} bytes"
                )
            checksum = hasher.hexdigest()
            if checksum != artifact.checksum:
                raise RetryableDownloadError(
                    f"Checksum mismatch for '{artifact.filename}': expected "
                    f"{artifact.checksum}, got {checksum}"
                )

    @staticmethod
    def new_hasher(checksum_type: ChecksumType, size: int) -> _Hasher:
        if checksum_type == "sha256":
            return hashlib.sha256()
        if checksum_type == "git-sha1":
            try:
                hasher = hashlib.sha1(usedforsecurity=False)
            except TypeError:
                hasher = hashlib.sha1()
            hasher.update(f"blob {size}\0".encode())
            return hasher
        raise ValueError(f"Unsupported checksum type: {checksum_type}")


class LocalModelStore:
    def __init__(
        self,
        primary_root: Path = DEFAULT_MODELS_ROOT,
        env_var: str = DEFAULT_MODELS_ENV_VAR,
        organization: str = DEFAULT_HF_ORG,
    ) -> None:
        self._primary_root = primary_root
        self._env_var = env_var
        self._organization = organization

    def resolve_runnable(self, model: str) -> Path | None:
        explicit = Path(model)
        if self._is_runnable(explicit):
            return explicit.resolve()
        stripped = self._strip_organization(model)
        for root in self.roots():
            for name in dict.fromkeys((model, stripped)):
                candidate = self.model_dir(root, name)
                if self._is_runnable(candidate):
                    return candidate.resolve()
        return None

    def iter_installed(self) -> list[Path]:
        models: list[Path] = []
        for root in self.roots():
            if not root.exists():
                continue
            children = sorted(
                root.iterdir(), key=lambda path: path.name.lower()
            )
            for child in children:
                if self._is_runnable(child):
                    models.append(child)
        return models

    def remove(self, model: str) -> bool:
        removed = False
        for root, model_name, target in self.find_removal_targets(model):
            with self.lock_model(root, model_name):
                if target.is_dir():
                    shutil.rmtree(target)
                    removed = True
        return removed

    def find_removal_targets(
        self,
        model: str,
    ) -> list[tuple[Path, str, Path]]:
        explicit = Path(model)
        if explicit.is_dir():
            return [(explicit.parent, explicit.name, explicit)]
        model_name = self._strip_organization(model)
        if not self._is_model_name(model_name):
            return []
        targets: list[tuple[Path, str, Path]] = []
        for root in self.roots():
            target = self.model_dir(root, model_name)
            if target.is_dir():
                targets.append((root, model_name, target))
        return targets

    def roots(self) -> Iterable[Path]:
        yield self._primary_root
        env_root = self._env_root()
        if env_root is not None and env_root != self._primary_root:
            yield env_root

    def select_download_root(self) -> Path:
        env_root = self._env_root()
        if env_root is not None:
            root = self._ensure_writable_root(env_root)
            if root is not None:
                return root
            raise RuntimeError(f"{self._env_var} is not writable: {env_root}")
        root = self._ensure_writable_root(self._primary_root)
        if root is not None:
            return root
        raise RuntimeError(
            f"No writable models path. Could not create {self._primary_root} "
            f"and {self._env_var} is not set."
        )

    @contextmanager
    def lock_model(self, root: Path, model_name: str) -> Iterator[None]:
        lock_dir = root / ".llima-locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_key = hashlib.sha256(model_name.encode("utf-8")).hexdigest()
        lock_path = lock_dir / f"{lock_key}.lock"
        with open(lock_path, "a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def validate_model_name(model_name: str) -> str:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("model_id must not be empty")
        if not LocalModelStore._is_model_name(model_name):
            raise ValueError(
                "model_id should be a model name without org prefix"
            )
        return model_name

    @staticmethod
    def model_dir(root: Path, model_name: str) -> Path:
        return root / model_name

    @staticmethod
    def artifact_path(root: Path, filename: str) -> Path:
        relative = PurePosixPath(filename)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in (".", "..") for part in relative.parts)
            or relative.parts[0] == INCOMPLETE_MARKER
        ):
            raise RuntimeError(
                f"Unsafe artifact path in model metadata: '{filename}'"
            )
        resolved_root = root.resolve()
        path = resolved_root.joinpath(*relative.parts)
        if not path.resolve(strict=False).is_relative_to(resolved_root):
            raise RuntimeError(
                f"Unsafe artifact path in model metadata: '{filename}'"
            )
        return path

    @staticmethod
    def artifact_matches(path: Path, artifact: ArtifactInfo) -> bool:
        try:
            if not path.is_file() or path.is_symlink():
                return False
            if path.stat().st_size != artifact.size:
                return False
            with open(path, "rb") as artifact_file:
                hasher = hashlib.file_digest(
                    artifact_file,
                    lambda: cast(
                        Any,
                        ArtifactDownloader.new_hasher(
                            artifact.checksum_type, artifact.size
                        ),
                    ),
                )
            return hasher.hexdigest() == artifact.checksum
        except OSError:
            return False

    @staticmethod
    def validate_runnable(model_dir: Path) -> None:
        if not (model_dir / "devkit").is_dir() or not (
            model_dir / "elf_files"
        ).is_dir():
            raise RuntimeError(
                "Model directory missing required 'devkit' or "
                "'elf_files' directories."
            )

    @staticmethod
    def mark_incomplete(model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / INCOMPLETE_MARKER).touch()

    @staticmethod
    def mark_complete(model_dir: Path) -> None:
        (model_dir / INCOMPLETE_MARKER).unlink(missing_ok=True)

    @staticmethod
    def _is_model_name(model_name: str) -> bool:
        return (
            bool(model_name)
            and "/" not in model_name
            and "\\" not in model_name
            and model_name not in (".", "..")
        )

    def _strip_organization(self, model_id: str) -> str:
        prefix = f"{self._organization}/"
        if model_id.startswith(prefix):
            return model_id[len(prefix):]
        return model_id

    @staticmethod
    def _is_runnable(path: Path) -> bool:
        return (
            path.is_dir()
            and not (path / INCOMPLETE_MARKER).exists()
            and (path / "devkit").is_dir()
            and (path / "elf_files").is_dir()
        )

    def _env_root(self) -> Path | None:
        value = os.getenv(self._env_var)
        return Path(value) if value else None

    @staticmethod
    def _ensure_writable_root(path: Path) -> Path | None:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        if path.is_dir() and os.access(path, os.W_OK):
            return path
        return None


class ModelManager:
    def __init__(
        self,
        client: HuggingFaceClient | None = None,
        downloader: ArtifactDownloader | None = None,
        store: LocalModelStore | None = None,
        download_config: DownloadConfig | None = None,
    ) -> None:
        self._client = client or HuggingFaceClient()
        self._store = store or LocalModelStore(
            organization=self._client.organization
        )
        self._downloader = downloader or ArtifactDownloader(
            self._client, download_config
        )

    def search(self, term: str) -> list[ModelInfo]:
        return self._client.search(term)

    def pull(self, model_id: str) -> Path:
        model_name = self._store.validate_model_name(model_id)
        snapshot = self._client.snapshot(model_name)
        root = self._store.select_download_root()
        with self._store.lock_model(root, model_name):
            return self._pull_to_root(
                root,
                model_name,
                snapshot.revision,
                snapshot.artifacts,
            )

    def list(self) -> list[Path]:
        return self._store.iter_installed()

    def remove(self, model_id: str) -> bool:
        return self._store.remove(model_id)

    def resolve(self, model: str) -> Path | None:
        return self._store.resolve_runnable(model)

    def _pull_to_root(
        self,
        root: Path,
        model_name: str,
        revision: str,
        artifacts: tuple[ArtifactInfo, ...],
    ) -> Path:
        model_dir = self._store.model_dir(root, model_name)
        local_artifacts = [
            (
                artifact,
                self._store.artifact_path(model_dir, artifact.filename),
            )
            for artifact in artifacts
        ]
        missing_or_invalid = [
            (artifact, path)
            for artifact, path in local_artifacts
            if not self._store.artifact_matches(path, artifact)
        ]
        if not missing_or_invalid:
            self._store.validate_runnable(model_dir)
            self._store.mark_complete(model_dir)
            return model_dir

        self._store.mark_incomplete(model_dir)
        try:
            with tempfile.TemporaryDirectory(
                prefix=f".{model_name}.download-",
                dir=root,
            ) as temporary_directory:
                staging_root = Path(temporary_directory)
                (staging_root / INCOMPLETE_MARKER).touch()
                downloads = [
                    ArtifactDownload(
                        artifact=artifact,
                        staging_path=self._store.artifact_path(
                            staging_root, artifact.filename
                        ),
                        destination_path=destination,
                    )
                    for artifact, destination in missing_or_invalid
                ]
                self._downloader.download(model_name, revision, downloads)

            self._store.validate_runnable(model_dir)
            self._store.mark_complete(model_dir)
            return model_dir
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f"Failed to pull model '{model_name}': {error}. "
                "The model remains incomplete; "
                f"retry with `llima pull {model_name}`."
            ) from error
