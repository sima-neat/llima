import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    downloads: int | None = None
    likes: int | None = None


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


def resolve_model_path(model: str) -> Path | None:
    candidate = Path(model)
    if candidate.exists():
        return candidate.resolve()

    stripped = _strip_org(model)

    for root in _iter_model_roots():
        candidate = _safe_model_dir(root, model)
        if candidate.exists():
            return candidate.resolve()
        if stripped != model:
            candidate = _safe_model_dir(root, stripped)
            if candidate.exists():
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


def _download_file(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} for {url}")
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = (downloaded / total_bytes) * 100
                    print(f"\rDownloading {label}: {pct:5.1f}%", end="", flush=True)
                else:
                    print(f"\rDownloading {label}: {downloaded} bytes", end="", flush=True)
        if total_bytes:
            print(f"\rDownloading {label}: 100.0%", flush=True)
        else:
            print(f"\rDownloading {label}: done", flush=True)


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
    if "/" in model_id:
        raise ValueError("model_id should be a model name without org prefix")
    model_name = model_id

    url = f"{HF_API_BASE}/models/{HF_ORG}/{model_name}"
    data = _fetch_json(url)
    siblings = data.get("siblings", [])

    root = _select_download_root()
    model_dir = _safe_model_dir(root, model_name)

    for entry in siblings:
        filename = entry.get("rfilename")
        if not filename:
            continue
        file_url = f"{HF_RESOLVE_BASE}/{HF_ORG}/{model_name}/resolve/main/{filename}"
        dest = model_dir / filename
        if dest.exists():
            continue
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        _download_file(file_url, tmp, filename)
        tmp.replace(dest)

    _validate_model_dir(model_dir)
    return model_dir


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
            if (child / "devkit").is_dir() and (child / "elf_files").is_dir():
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
