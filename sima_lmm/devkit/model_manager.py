import hashlib
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Literal, Protocol


HF_ORG = "simaai"
HF_API_BASE = "https://huggingface.co/api"
HF_RESOLVE_BASE = "https://huggingface.co"
HF_COLLECTIONS = (
    "large-language-models",
    "vision-language-models",
    "speech-and-audio-models",
)

NVME_MODELS_ROOT = Path("/media/nvme/llima/models")
ENV_MODELS_VAR = "LLIMA_MODELS_PATH"
INCOMPLETE_MARKER = ".llima-incomplete"
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
ChecksumType = Literal["sha256", "git-sha1"]


class _Hasher(Protocol):
    def update(self, data: bytes) -> None: ...

    def hexdigest(self) -> str: ...


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


def _env_models_root() -> Path | None:
    value = os.getenv(ENV_MODELS_VAR)
    if not value:
        return None
    return Path(value)


def _iter_model_roots() -> Iterable[Path]:
    yield NVME_MODELS_ROOT
    env_root = _env_models_root()
    if env_root is not None:
        yield env_root


def _safe_model_dir(root: Path, model_id: str) -> Path:
    return root / model_id


def _is_incomplete_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / INCOMPLETE_MARKER).exists()


def resolve_model_path(model: str) -> Path | None:
    candidate = Path(model)
    if candidate.exists() and not _is_incomplete_model_dir(candidate):
        return candidate.resolve()

    stripped = _strip_org(model)

    for root in _iter_model_roots():
        candidate = _safe_model_dir(root, model)
        if candidate.exists() and not _is_incomplete_model_dir(candidate):
            return candidate.resolve()
        if stripped != model:
            candidate = _safe_model_dir(root, stripped)
            if candidate.exists() and not _is_incomplete_model_dir(candidate):
                return candidate.resolve()

    return None


def _fetch_json_with_headers(url: str) -> tuple[object, dict[str, str]]:
    with urllib.request.urlopen(url) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} for {url}")
        payload = response.read()
        headers = dict(response.headers.items())
    return json.loads(payload.decode("utf-8")), headers


def _fetch_json(url: str) -> object:
    data, _headers = _fetch_json_with_headers(url)
    return data


def _parse_next_cursor(headers: dict[str, str]) -> str | None:
    for key in ("X-Next-Cursor", "X-Next-Page"):
        value = headers.get(key)
        if value:
            return value
    link = headers.get("Link")
    if not link:
        return None
    for part in link.split(","):
        if 'rel=\"next\"' not in part:
            continue
        if "cursor=" not in part:
            continue
        segment = part.split("cursor=", 1)[1]
        cursor = segment.split("&", 1)[0].strip(" >")
        if cursor:
            return cursor
    return None


def _strip_org(model_id: str) -> str:
    prefix = f"{HF_ORG}/"
    if model_id.startswith(prefix):
        return model_id[len(prefix) :]
    return model_id


def _collection_model_ids() -> set[str]:
    ids: set[str] = set()
    for slug in HF_COLLECTIONS:
        data = _fetch_json(f"{HF_API_BASE}/collections/{HF_ORG}/{slug}")
        for item in data.get("items", []):
            if item.get("type") != "model":
                continue
            model_id = item.get("id")
            if model_id:
                ids.add(_strip_org(model_id))
    return ids


def search_models(term: str) -> list[ModelInfo]:
    params = f"author={HF_ORG}"
    if term:
        params += f"&search={term}"

    results: list[ModelInfo] = []
    cursor: str | None = None
    while True:
        cursor_param = f"&cursor={cursor}" if cursor else ""
        url = f"{HF_API_BASE}/models?{params}{cursor_param}"
        data, headers = _fetch_json_with_headers(url)
        items: list[dict] = []
        if isinstance(data, dict):
            items = data.get("models", [])
            cursor = data.get("next", None) or _parse_next_cursor(headers)
        elif isinstance(data, list):
            items = data
            cursor = _parse_next_cursor(headers)
        else:
            items = []
            cursor = None

        for item in items:
            model_id = item.get("modelId", "")
            results.append(
                ModelInfo(
                    model_id=_strip_org(model_id),
                    downloads=item.get("downloads"),
                    likes=item.get("likes"),
                )
            )

        if not cursor:
            break

    allowed_ids = _collection_model_ids()
    return [info for info in results if info.model_id in allowed_ids]


def _new_hasher(checksum_type: ChecksumType, size: int) -> _Hasher:
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


def _artifact_checksum(path: Path, artifact: ArtifactInfo) -> str:
    with open(path, "rb") as f:
        hasher = hashlib.file_digest(
            f, lambda: _new_hasher(artifact.checksum_type, artifact.size)
        )
    return hasher.hexdigest()


def _artifact_matches(path: Path, artifact: ArtifactInfo) -> bool:
    try:
        if not path.is_file() or path.is_symlink():
            return False
        if path.stat().st_size != artifact.size:
            return False
        return _artifact_checksum(path, artifact) == artifact.checksum
    except OSError:
        return False


def _download_file(url: str, dest: Path, artifact: ArtifactInfo) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} for {url}")
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        if total_bytes is not None and total_bytes != artifact.size:
            raise RuntimeError(
                f"Invalid size for '{artifact.filename}': expected {artifact.size} bytes, "
                f"server reported {total_bytes} bytes"
            )
        downloaded = 0
        hasher = _new_hasher(artifact.checksum_type, artifact.size)
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                hasher.update(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = (downloaded / total_bytes) * 100
                    print(
                        f"\rDownloading {artifact.filename}: {pct:5.1f}%", end="", flush=True
                    )
                else:
                    print(
                        f"\rDownloading {artifact.filename}: {downloaded} bytes",
                        end="",
                        flush=True,
                    )
        if downloaded != artifact.size:
            raise RuntimeError(
                f"Invalid size for '{artifact.filename}': expected {artifact.size} bytes, "
                f"downloaded {downloaded} bytes"
            )
        checksum = hasher.hexdigest()
        if checksum != artifact.checksum:
            raise RuntimeError(
                f"Checksum mismatch for '{artifact.filename}': expected {artifact.checksum}, "
                f"got {checksum}"
            )
        if total_bytes:
            print(f"\rDownloading {artifact.filename}: 100.0%", flush=True)
        else:
            print(f"\rDownloading {artifact.filename}: done", flush=True)


def _artifact_info(entry: dict[str, object]) -> ArtifactInfo:
    filename = entry.get("rfilename")
    size = entry.get("size")
    if not isinstance(filename, str) or not filename:
        raise RuntimeError("Model metadata contains an artifact without a filename.")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise RuntimeError(f"Model metadata is missing a valid size for '{filename}'.")

    lfs = entry.get("lfs")
    if isinstance(lfs, dict) and isinstance(lfs.get("sha256"), str):
        checksum = lfs["sha256"].lower()
        checksum_type: ChecksumType = "sha256"
        checksum_length = 64
    else:
        checksum = str(entry.get("blobId", "")).lower()
        checksum_type = "git-sha1"
        checksum_length = 40

    if len(checksum) != checksum_length or any(c not in "0123456789abcdef" for c in checksum):
        raise RuntimeError(f"Model metadata is missing a valid checksum for '{filename}'.")
    return ArtifactInfo(filename, size, checksum, checksum_type)


def _safe_artifact_path(root: Path, filename: str) -> Path:
    relative = PurePosixPath(filename)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in (".", "..") for part in relative.parts)
        or relative.parts[0] == INCOMPLETE_MARKER
    ):
        raise RuntimeError(f"Unsafe artifact path in model metadata: '{filename}'")

    root = root.resolve()
    path = root.joinpath(*relative.parts)
    if not path.resolve(strict=False).is_relative_to(root):
        raise RuntimeError(f"Unsafe artifact path in model metadata: '{filename}'")
    return path


def _ensure_writable_models_root(path: Path) -> Path | None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    if path.is_dir() and os.access(path, os.W_OK):
        return path
    return None


def _select_download_root() -> Path:
    env_root = _env_models_root()
    if env_root is not None:
        root = _ensure_writable_models_root(env_root)
        if root is not None:
            return root
        raise RuntimeError(f"LLIMA_MODELS_PATH is not writable: {env_root}")

    root = _ensure_writable_models_root(NVME_MODELS_ROOT)
    if root is not None:
        return root

    raise RuntimeError(
        "No writable models path. Could not create /media/nvme/llima/models "
        "and LLIMA_MODELS_PATH is not set."
    )


def pull_model(model_id: str) -> Path:
    model_id = model_id.strip()
    if not model_id:
        raise ValueError("model_id must not be empty")
    if "/" in model_id or "\\" in model_id or model_id in (".", ".."):
        raise ValueError("model_id should be a model name without org prefix")
    model_name = model_id

    encoded_model_name = urllib.parse.quote(model_name, safe="")
    url = f"{HF_API_BASE}/models/{HF_ORG}/{encoded_model_name}?blobs=true"
    data = _fetch_json(url)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid model metadata for '{model_name}'.")
    revision = data.get("sha")
    siblings = data.get("siblings")
    if not isinstance(revision, str) or not revision:
        raise RuntimeError(f"Model metadata is missing a revision for '{model_name}'.")
    if not isinstance(siblings, list):
        raise RuntimeError(f"Model metadata is missing artifacts for '{model_name}'.")

    artifacts: list[ArtifactInfo] = []
    filenames: set[str] = set()
    for entry in siblings:
        if not isinstance(entry, dict):
            raise RuntimeError(f"Invalid artifact metadata for '{model_name}'.")
        artifact = _artifact_info(entry)
        if artifact.filename in filenames:
            raise RuntimeError(f"Duplicate artifact in model metadata: '{artifact.filename}'")
        filenames.add(artifact.filename)
        artifacts.append(artifact)

    root = _select_download_root()
    model_dir = _safe_model_dir(root, model_name)
    local_artifacts = [
        (artifact, _safe_artifact_path(model_dir, artifact.filename))
        for artifact in artifacts
    ]
    missing_or_invalid = [
        (artifact, path)
        for artifact, path in local_artifacts
        if not _artifact_matches(path, artifact)
    ]

    marker = model_dir / INCOMPLETE_MARKER
    if not missing_or_invalid:
        _validate_model_dir(model_dir)
        marker.unlink(missing_ok=True)
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    marker.touch()
    encoded_revision = urllib.parse.quote(revision, safe="")

    try:
        with tempfile.TemporaryDirectory(prefix=f".{model_name}.download-", dir=root) as tmp:
            download_dir = Path(tmp)
            (download_dir / INCOMPLETE_MARKER).touch()
            downloaded: list[tuple[Path, Path]] = []
            for artifact, dest in missing_or_invalid:
                staged = _safe_artifact_path(download_dir, artifact.filename)
                encoded_filename = urllib.parse.quote(artifact.filename, safe="/")
                file_url = (
                    f"{HF_RESOLVE_BASE}/{HF_ORG}/{encoded_model_name}/resolve/"
                    f"{encoded_revision}/{encoded_filename}"
                )
                _download_file(file_url, staged, artifact)
                downloaded.append((staged, dest))

            for staged, dest in downloaded:
                dest.parent.mkdir(parents=True, exist_ok=True)
                staged.replace(dest)

        _validate_model_dir(model_dir)
        marker.unlink(missing_ok=True)
        return model_dir
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"Failed to pull model '{model_name}': {exc}. The model remains incomplete; "
            f"retry with `llima pull {model_name}`."
        ) from exc


def _validate_model_dir(model_dir: Path) -> None:
    devkit = model_dir / "devkit"
    elf_files = model_dir / "elf_files"
    if not devkit.is_dir() or not elf_files.is_dir():
        raise RuntimeError(
            "Model directory missing required 'devkit' or 'elf_files' directories."
        )


def list_models() -> list[Path]:
    models: list[Path] = []
    for root in _iter_model_roots():
        if not root.exists():
            continue
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            if (
                not _is_incomplete_model_dir(child)
                and (child / "devkit").is_dir()
                and (child / "elf_files").is_dir()
            ):
                models.append(child)
    return models


def rm_model(model_id: str) -> bool:
    candidate = Path(model_id)
    if candidate.exists():
        shutil.rmtree(candidate)
        return True

    if model_id.startswith(f"{HF_ORG}/"):
        model_id = model_id.split("/", 1)[1]

    removed = False
    for root in _iter_model_roots():
        target = _safe_model_dir(root, model_id)
        if target.exists():
            shutil.rmtree(target)
            removed = True
    return removed
