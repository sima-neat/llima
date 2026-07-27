#!/usr/bin/env python3
"""Materialize verified LLiMa model inputs from the Vulcan cache."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


CHUNK_SIZE = 1024 * 1024
DOWNLOAD_ATTEMPTS = 3
MIN_DISK_HEADROOM = 2 * 1024**3
RESOLVED_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SELECTION_POLICY = json.loads(
    (
        Path(__file__).parents[1]
        / "hf-safetensors"
        / "selection-policy.json"
    ).read_text(encoding="utf-8")
)
CONFIG_PATTERNS = tuple(SELECTION_POLICY["config"])
SELECTION_PATTERNS = {
    "config": CONFIG_PATTERNS,
    "safetensors": (*SELECTION_POLICY["safetensors"], *CONFIG_PATTERNS),
}


class PreparationError(RuntimeError):
    """Raised when cached model inputs cannot be prepared safely."""


@dataclass(frozen=True)
class SourceSpec:
    repo_id: str
    revision: str
    payload_format: str
    cache_root: str
    file_patterns: tuple[str, ...] = ()

    @property
    def model_folder(self) -> str:
        return f"models--{self.repo_id.replace('/', '--')}"


@dataclass(frozen=True)
class CachedFile:
    relative_path: PurePosixPath
    s3_key: str
    size: int
    sha256: str
    download_url: str


@dataclass(frozen=True)
class ModelPlan:
    source: SourceSpec
    resolved_revision: str
    selection_sha256: str
    files: tuple[CachedFile, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and verify all active LLiMa model inputs from the Vulcan cache."
        )
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tools/hf-safetensors/manifest.txt"),
    )
    parser.add_argument(
        "--config-manifest",
        type=Path,
        default=Path("tools/hf-safetensors/config-manifest.txt"),
    )
    parser.add_argument(
        "--gguf-manifest",
        type=Path,
        default=Path("tools/hf-safetensors/gguf-manifest.txt"),
    )
    parser.add_argument("--cache-root", default="llima-safetensors")
    parser.add_argument("--config-cache-root", default="llima-hf-config")
    parser.add_argument("--gguf-cache-root", default="llima-gguf")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--provenance-output", type=Path)
    return parser.parse_args()


def validate_repo_id(repo_id: str) -> str:
    repo_id = repo_id.strip().strip("/")
    parts = repo_id.split("/")
    valid_part = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    if len(parts) != 2 or any(not valid_part.fullmatch(part) for part in parts):
        raise PreparationError(
            f"Expected repository id 'namespace/model', got {repo_id!r}"
        )
    return repo_id


def validate_cache_root(cache_root: str) -> str:
    cache_root = cache_root.strip().strip("/")
    if not cache_root or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", cache_root):
        raise PreparationError(f"Invalid Vulcan cache root: {cache_root!r}")
    if ".." in cache_root.split("/"):
        raise PreparationError(f"Invalid Vulcan cache root: {cache_root!r}")
    return cache_root


def parse_source_manifest(
    path: Path,
    *,
    payload_format: str,
    cache_root: str,
    allow_filenames: bool = False,
) -> list[SourceSpec]:
    if not path.is_file():
        raise PreparationError(f"Source manifest does not exist: {path}")

    specs: list[SourceSpec] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if not allow_filenames and len(parts) > 2:
            raise PreparationError(
                f"{path}:{line_number}: expected '<repo_id> [revision]'"
            )
        repo_id = validate_repo_id(parts[0])
        revision = parts[1] if len(parts) >= 2 else "main"
        file_patterns = tuple(parts[2:]) if allow_filenames else ()
        if any(not pattern.lower().endswith(".gguf") for pattern in file_patterns):
            raise PreparationError(
                f"{path}:{line_number}: explicit GGUF filenames must end in .gguf"
            )
        key = f"{repo_id}@{revision}"
        if key in seen:
            raise PreparationError(f"{path}:{line_number}: duplicate entry {key}")
        seen.add(key)
        specs.append(
            SourceSpec(
                repo_id=repo_id,
                revision=revision,
                payload_format=payload_format,
                cache_root=validate_cache_root(cache_root),
                file_patterns=file_patterns,
            )
        )

    if not specs:
        raise PreparationError(f"Source manifest is empty: {path}")
    return specs


def normalize_base_url(raw_url: str) -> str:
    base_url = raw_url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(base_url)
    is_local_http = parsed.scheme == "http" and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }
    if parsed.scheme != "https" and not is_local_http:
        raise PreparationError(
            "Vulcan artifact base URL must use HTTPS "
            "(HTTP is allowed only for local validation)"
        )
    if not parsed.netloc or parsed.query or parsed.fragment:
        raise PreparationError(f"Invalid Vulcan artifact base URL: {raw_url!r}")
    return base_url


def build_url(base_url: str, key: str) -> str:
    return f"{base_url}/{urllib.parse.quote(key, safe='/')}"


def selection_fingerprint(payload_format: str, file_patterns: tuple[str, ...]) -> str:
    selection = {
        "format": payload_format,
        "allow_patterns": sorted(file_patterns),
    }
    payload = json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def open_internal_url(url: str, *, timeout: int):
    response = urllib.request.urlopen(url, timeout=timeout)
    requested = urllib.parse.urlsplit(url)
    resolved = urllib.parse.urlsplit(response.geturl())
    if (resolved.scheme, resolved.netloc) != (requested.scheme, requested.netloc):
        response.close()
        raise PreparationError(
            f"Refusing cross-origin redirect from {url} to {resolved.geturl()}"
        )
    return response


def fetch_json(url: str) -> dict[str, Any]:
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with open_internal_url(url, timeout=60) as response:
                payload = response.read()
            break
        except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
            retryable = not isinstance(exc, urllib.error.HTTPError) or (
                exc.code in {408, 429} or exc.code >= 500
            )
            if not retryable or attempt == DOWNLOAD_ATTEMPTS:
                raise PreparationError(
                    f"Unable to download cache manifest {url}: {exc}"
                ) from exc
            delay = attempt * 5
            print(
                f"Manifest download attempt {attempt} failed for {url}; "
                f"retrying in {delay}s: {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Cache manifest is not valid JSON: {url}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise PreparationError(f"Cache manifest root must be an object: {url}")
    return parsed


def validate_relative_path(raw_path: object, *, context: str) -> PurePosixPath:
    if not isinstance(raw_path, str) or not raw_path:
        raise PreparationError(f"{context}: file path must be a non-empty string")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PreparationError(f"{context}: unsafe file path {raw_path!r}")
    return path


def require_string(manifest: dict[str, Any], key: str, *, context: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise PreparationError(f"{context}: {key} must be a non-empty string")
    return value


def resolve_model_plan(base_url: str, source: SourceSpec) -> ModelPlan:
    prefix = f"{source.cache_root}/{source.repo_id}/latest"
    manifest_url = build_url(base_url, f"{prefix}/manifest.json")
    print(f"Resolving cached inputs for {source.repo_id}@{source.revision}", flush=True)
    manifest = fetch_json(manifest_url)
    context = f"{source.repo_id} cache manifest"

    if require_string(manifest, "repo_id", context=context) != source.repo_id:
        raise PreparationError(f"{context}: repository id does not match source manifest")
    if require_string(manifest, "requested_revision", context=context) != source.revision:
        raise PreparationError(
            f"{context}: requested revision does not match source manifest"
        )
    if require_string(manifest, "format", context=context) != source.payload_format:
        raise PreparationError(f"{context}: payload format does not match cache root")

    resolved_revision = require_string(
        manifest, "resolved_revision", context=context
    ).lower()
    if not RESOLVED_REVISION_PATTERN.fullmatch(resolved_revision):
        raise PreparationError(f"{context}: invalid resolved Hugging Face revision")
    selection_sha256 = require_string(
        manifest, "selection_sha256", context=context
    ).lower()
    if not SHA256_PATTERN.fullmatch(selection_sha256):
        raise PreparationError(f"{context}: invalid selection SHA-256")
    expected_patterns = source.file_patterns or SELECTION_PATTERNS.get(
        source.payload_format, ()
    )
    if expected_patterns and selection_sha256 != selection_fingerprint(
        source.payload_format, expected_patterns
    ):
        raise PreparationError(
            f"{context}: cached file selection does not match source manifest"
        )

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise PreparationError(f"{context}: files must be a non-empty list")

    files: list[CachedFile] = []
    seen_paths: set[PurePosixPath] = set()
    for index, raw_file in enumerate(raw_files):
        file_context = f"{context} file {index}"
        if not isinstance(raw_file, dict):
            raise PreparationError(f"{file_context}: entry must be an object")
        relative_path = validate_relative_path(
            raw_file.get("path"), context=file_context
        )
        if relative_path in seen_paths:
            raise PreparationError(f"{file_context}: duplicate path {relative_path}")
        seen_paths.add(relative_path)

        size = raw_file.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise PreparationError(f"{file_context}: size must be a non-negative integer")
        sha256 = raw_file.get("sha256")
        if not isinstance(sha256, str) or not SHA256_PATTERN.fullmatch(sha256.lower()):
            raise PreparationError(f"{file_context}: invalid SHA-256")
        expected_key = f"{prefix}/{relative_path.as_posix()}"
        if raw_file.get("s3_key") != expected_key:
            raise PreparationError(
                f"{file_context}: object key is outside the expected cache prefix"
            )

        files.append(
            CachedFile(
                relative_path=relative_path,
                s3_key=expected_key,
                size=size,
                sha256=sha256.lower(),
                download_url=build_url(base_url, expected_key),
            )
        )

    return ModelPlan(
        source=source,
        resolved_revision=resolved_revision,
        selection_sha256=selection_sha256,
        files=tuple(files),
    )


def nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise PreparationError(f"No existing parent found for output directory: {path}")
        candidate = candidate.parent
    return candidate


def validate_output_directory(path: Path) -> Path:
    output_dir = path.expanduser().resolve()
    forbidden = {Path("/"), Path.home().resolve(), Path.cwd().resolve()}
    if output_dir in forbidden or output_dir.parent == Path("/"):
        raise PreparationError(f"Refusing unsafe output directory: {output_dir}")
    return output_dir


def ensure_disk_capacity(output_dir: Path, total_size: int) -> None:
    existing_parent = nearest_existing_parent(output_dir.parent)
    free_bytes = shutil.disk_usage(existing_parent).free
    headroom = max(total_size // 10, MIN_DISK_HEADROOM)
    required = total_size + headroom
    print(
        "Model input disk requirement: "
        f"payload={total_size} bytes, headroom={headroom} bytes, "
        f"available={free_bytes} bytes",
        flush=True,
    )
    if free_bytes < required:
        raise PreparationError(
            f"Insufficient disk space: need {required} bytes, have {free_bytes} bytes"
        )


def download_file(cached_file: CachedFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(
        f".{destination.name}.partial-{os.getpid()}-"
        f"{hashlib.sha256(cached_file.s3_key.encode()).hexdigest()[:12]}"
    )
    try:
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            partial.unlink(missing_ok=True)
            digest = hashlib.sha256()
            byte_count = 0
            try:
                with open_internal_url(
                    cached_file.download_url, timeout=300
                ) as response, partial.open("wb") as handle:
                    while chunk := response.read(CHUNK_SIZE):
                        handle.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)
            except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
                if attempt == DOWNLOAD_ATTEMPTS:
                    raise PreparationError(
                        f"Unable to download {cached_file.download_url} after "
                        f"{DOWNLOAD_ATTEMPTS} attempts: {exc}"
                    ) from exc
                delay = attempt * 5
                print(
                    f"Download attempt {attempt} failed for {cached_file.s3_key}; "
                    f"retrying in {delay}s: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
                continue

            if byte_count != cached_file.size:
                if attempt < DOWNLOAD_ATTEMPTS:
                    delay = attempt * 5
                    print(
                        f"Size mismatch for {cached_file.s3_key}; "
                        f"retrying in {delay}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                raise PreparationError(
                    f"Size mismatch for {cached_file.s3_key}: "
                    f"expected {cached_file.size}, got {byte_count}"
                )

            actual_sha256 = digest.hexdigest()
            if actual_sha256 != cached_file.sha256:
                raise PreparationError(
                    f"SHA-256 mismatch for {cached_file.s3_key}: "
                    f"expected {cached_file.sha256}, got {actual_sha256}"
                )
            os.replace(partial, destination)
            print(f"Verified {cached_file.s3_key}", flush=True)
            return
    finally:
        partial.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def provenance_for(plans: list[ModelPlan], output_dir: Path) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_root": str(output_dir),
        "model_count": len(plans),
        "file_count": sum(len(plan.files) for plan in plans),
        "total_size": sum(file.size for plan in plans for file in plan.files),
        "models": [
            {
                "repo_id": plan.source.repo_id,
                "requested_revision": plan.source.revision,
                "resolved_revision": plan.resolved_revision,
                "payload_format": plan.source.payload_format,
                "selection_sha256": plan.selection_sha256,
                "model_folder": plan.source.model_folder,
                "files": [
                    {
                        "path": file.relative_path.as_posix(),
                        "size": file.size,
                        "sha256": file.sha256,
                        "s3_key": file.s3_key,
                    }
                    for file in plan.files
                ],
            }
            for plan in plans
        ],
    }


def load_sources(args: argparse.Namespace) -> list[SourceSpec]:
    sources = [
        *parse_source_manifest(
            args.manifest,
            payload_format="safetensors",
            cache_root=args.cache_root,
        ),
        *parse_source_manifest(
            args.config_manifest,
            payload_format="config",
            cache_root=args.config_cache_root,
        ),
        *parse_source_manifest(
            args.gguf_manifest,
            payload_format="gguf",
            cache_root=args.gguf_cache_root,
            allow_filenames=True,
        ),
    ]
    target_folders: set[str] = set()
    for source in sources:
        if source.model_folder in target_folders:
            raise PreparationError(
                f"Multiple manifest entries target {source.model_folder}"
            )
        target_folders.add(source.model_folder)
    return sources


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.jobs < 1:
        raise PreparationError("--jobs must be at least 1")
    base_url = normalize_base_url(args.base_url)
    output_dir = validate_output_directory(args.output_dir)
    if output_dir.exists():
        raise PreparationError(
            f"Output directory already exists; refusing to replace it: {output_dir}"
        )
    sources = load_sources(args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        plans = list(
            executor.map(lambda source: resolve_model_plan(base_url, source), sources)
        )

    total_size = sum(file.size for plan in plans for file in plan.files)
    ensure_disk_capacity(output_dir, total_size)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent)
    )
    try:
        download_tasks: list[tuple[CachedFile, Path]] = []
        for plan in plans:
            model_dir = staging_dir / plan.source.model_folder
            for cached_file in plan.files:
                destination = model_dir.joinpath(*cached_file.relative_path.parts)
                download_tasks.append((cached_file, destination))

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(download_file, cached_file, destination)
                for cached_file, destination in download_tasks
            ]
            try:
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

        os.replace(staging_dir, output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    provenance = provenance_for(plans, output_dir)
    provenance_output = args.provenance_output or (
        output_dir.parent / "llima-model-input-provenance.json"
    )
    write_json_atomic(provenance_output, provenance)
    print(json.dumps(provenance, indent=2, sort_keys=True))
    print(f"Prepared all model inputs under {output_dir}")
    print(f"Wrote provenance to {provenance_output.expanduser().resolve()}")
    return provenance


def main() -> int:
    try:
        prepare(parse_args())
    except PreparationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
