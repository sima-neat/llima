#!/usr/bin/env python3
"""Cache Hugging Face safetensors and GGUF inputs in Vulcan artifacts."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


CONFIG_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "preprocessor_config.json",
    "processor_config.json",
    "image_processor_config.json",
]

SAFETENSORS_ALLOW_PATTERNS = [
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    *CONFIG_ALLOW_PATTERNS,
]


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str
    payload_format: str = "safetensors"
    file_patterns: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache Hugging Face safetensors and GGUF payloads in Vulcan S3."
    )
    parser.add_argument("--manifest", default="tools/hf-safetensors/manifest.txt")
    parser.add_argument(
        "--config-manifest", default="tools/hf-safetensors/config-manifest.txt"
    )
    parser.add_argument(
        "--gguf-manifest", default="tools/hf-safetensors/gguf-manifest.txt"
    )
    parser.add_argument("--repo-id", default="", help="Optional single model override.")
    parser.add_argument("--revision", default="main", help="Revision for --repo-id.")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--cache-root", default="llima-safetensors")
    parser.add_argument("--config-cache-root", default="llima-hf-config")
    parser.add_argument("--gguf-cache-root", default="llima-gguf")
    parser.add_argument("--kms-key-id", default="")
    parser.add_argument("--cloudfront-distribution-id", default="")
    parser.add_argument("--work-dir", default="_hf-safetensors-cache")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--publish-probe",
        action="store_true",
        help="During dry-run, verify S3/KMS write, read, and delete permissions under the cache root.",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Resolve Hugging Face revisions without checking S3, downloading, or publishing.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(message)


def validate_repo_id(repo_id: str) -> str:
    repo_id = repo_id.strip().strip("/")
    if not repo_id:
        fail("repo_id is empty.")
    if repo_id.startswith(".") or ".." in repo_id.split("/"):
        fail(f"Invalid Hugging Face repo_id: {repo_id!r}")
    parts = repo_id.split("/")
    if len(parts) != 2:
        fail(f"Expected repo_id to be 'namespace/model', got: {repo_id!r}")
    valid = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    for part in parts:
        if not valid.match(part):
            fail(f"Invalid repo_id path component {part!r} in {repo_id!r}")
    return "/".join(parts)


def parse_manifest(path: Path, payload_format: str = "safetensors") -> list[ModelSpec]:
    if not path.exists():
        fail(f"Manifest does not exist: {path}")

    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) > 2:
            fail(f"{path}:{line_no}: expected '<repo_id> [revision]', got: {raw_line!r}")
        repo_id = validate_repo_id(parts[0])
        revision = parts[1] if len(parts) == 2 else "main"
        key = f"{repo_id}@{revision}"
        if key in seen:
            fail(f"{path}:{line_no}: duplicate model entry: {key}")
        seen.add(key)
        specs.append(
            ModelSpec(
                repo_id=repo_id,
                revision=revision,
                payload_format=payload_format,
            )
        )

    if not specs:
        fail(f"Manifest did not contain any model entries: {path}")
    return specs


def parse_gguf_manifest(path: Path) -> list[ModelSpec]:
    if not path.exists():
        fail(f"Manifest does not exist: {path}")

    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for line_no, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        repo_id = validate_repo_id(parts[0])
        revision = parts[1] if len(parts) >= 2 else "main"
        file_patterns = tuple(parts[2:])
        if any(not pattern.lower().endswith(".gguf") for pattern in file_patterns):
            fail(
                f"{path}:{line_no}: explicit GGUF filenames must end in .gguf: "
                f"{raw_line!r}"
            )
        key = f"{repo_id}@{revision}"
        if key in seen:
            fail(f"{path}:{line_no}: duplicate model entry: {key}")
        seen.add(key)
        specs.append(
            ModelSpec(
                repo_id=repo_id,
                revision=revision,
                payload_format="gguf",
                file_patterns=file_patterns,
            )
        )

    if not specs:
        fail(f"Manifest did not contain any model entries: {path}")
    return specs


def resolve_specs(args: argparse.Namespace) -> list[ModelSpec]:
    if args.repo_id.strip():
        repo_id = validate_repo_id(args.repo_id)
        config_repositories = {
            spec.repo_id
            for spec in parse_manifest(
                Path(args.config_manifest), payload_format="config"
            )
        }
        gguf_specs = {
            spec.repo_id: spec
            for spec in parse_gguf_manifest(Path(args.gguf_manifest))
        }
        if repo_id.lower().endswith("-gguf"):
            payload_format = "gguf"
            file_patterns = gguf_specs.get(
                repo_id, ModelSpec(repo_id, "main", "gguf")
            ).file_patterns
        elif repo_id in config_repositories:
            payload_format = "config"
            file_patterns = ()
        else:
            payload_format = "safetensors"
            file_patterns = ()
        return [
            ModelSpec(
                repo_id,
                args.revision.strip() or "main",
                payload_format=payload_format,
                file_patterns=file_patterns,
            )
        ]
    return [
        *parse_manifest(Path(args.manifest)),
        *parse_manifest(Path(args.config_manifest), payload_format="config"),
        *parse_gguf_manifest(Path(args.gguf_manifest)),
    ]


def run(command: list[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=check, text=True, capture_output=capture)


def s3_cp_args(kms_key_id: str) -> list[str]:
    args = ["--sse", "aws:kms"]
    if kms_key_id:
        args.extend(["--sse-kms-key-id", kms_key_id])
    return args


def content_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_fingerprint(payload_format: str, allow_patterns: list[str]) -> str:
    selection = {
        "format": payload_format,
        "allow_patterns": sorted(allow_patterns),
    }
    payload = json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def cloudfront_url(base_url: str, s3_key: str) -> str:
    return f"{base_url.rstrip('/')}/{s3_key}"


def dry_run_probe_prefix(cache_root: str) -> str:
    run_id = os.environ.get("GITHUB_RUN_ID") or "local"
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT") or "0"
    return f"{cache_root.strip('/')}/_dry-run/{run_id}-{run_attempt}"


def cache_prefix(cache_root: str, repo_id: str) -> str:
    return f"{cache_root.strip('/')}/{repo_id}/latest"


def select_gguf_file(info: object, repo_id: str) -> str:
    filenames = [
        sibling.rfilename
        for sibling in (getattr(info, "siblings", None) or [])
        if getattr(sibling, "rfilename", "").lower().endswith(".gguf")
    ]
    for quantization in ("q4_0", "q8_0"):
        matches = sorted(
            filename
            for filename in filenames
            if Path(filename).stem.lower().endswith(quantization)
        )
        root_matches = [filename for filename in matches if "/" not in filename]
        if root_matches:
            matches = root_matches
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            fail(
                f"Multiple {quantization.upper()} GGUF files found for {repo_id}: "
                f"{', '.join(matches)}"
            )
    fail(f"No Q4_0 or Q8_0 GGUF file found for {repo_id}")


def validate_gguf_files(
    info: object, repo_id: str, requested_files: tuple[str, ...]
) -> list[str]:
    available_files = {
        sibling.rfilename
        for sibling in (getattr(info, "siblings", None) or [])
        if getattr(sibling, "rfilename", "").lower().endswith(".gguf")
    }
    missing_files = [
        filename for filename in requested_files if filename not in available_files
    ]
    if missing_files:
        fail(
            f"Requested GGUF files are missing for {repo_id}: "
            f"{', '.join(missing_files)}"
        )
    return list(requested_files)


def fetch_cached_manifest(bucket: str, prefix: str) -> dict | None:
    result = run(
        ["aws", "s3", "cp", f"s3://{bucket}/{prefix}/manifest.json", "-"],
        capture=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"No existing cache manifest found at s3://{bucket}/{prefix}/manifest.json")
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        fail(f"Existing cache manifest is not valid JSON: {exc}")


def find_downloaded_files(snapshot_path: Path) -> list[Path]:
    return [
        path
        for path in sorted(snapshot_path.rglob("*"))
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(snapshot_path).parts)
    ]


def has_safetensors(files: list[Path]) -> bool:
    return any(
        fnmatch.fnmatch(path.name, "*.safetensors")
        or fnmatch.fnmatch(path.name, "*.safetensors.index.json")
        for path in files
    )


def build_manifest(
    *,
    spec: ModelSpec,
    resolved_revision: str,
    prefix: str,
    files: list[Path],
    snapshot_path: Path,
    base_url: str,
    payload_format: str,
    selection_sha256: str,
) -> dict:
    entries = []
    for path in files:
        rel = path.relative_to(snapshot_path)
        rel_key = "/".join(rel.parts)
        s3_key = f"{prefix}/{rel_key}"
        entries.append(
            {
                "path": rel_key,
                "s3_key": s3_key,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "url": cloudfront_url(base_url, s3_key),
            }
        )

    return {
        "source": "hf",
        "format": payload_format,
        "selection_sha256": selection_sha256,
        "repo_id": spec.repo_id,
        "requested_revision": spec.revision,
        "resolved_revision": resolved_revision,
        "cache_policy": "latest-only",
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": {
            "repository": os.environ.get("GITHUB_REPOSITORY", ""),
            "run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
            "sha": os.environ.get("GITHUB_SHA", ""),
        },
        "files": entries,
    }


def publish_model(
    *,
    args: argparse.Namespace,
    spec: ModelSpec,
    resolved_revision: str,
    prefix: str,
    snapshot_path: Path,
    files: list[Path],
    manifest: dict,
) -> None:
    manifest_path = snapshot_path.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(f"Dry run enabled; not publishing s3://{args.bucket}/{prefix}/")
        return

    run(["aws", "s3", "rm", f"s3://{args.bucket}/{prefix}/", "--recursive"])
    for path in files:
        rel = path.relative_to(snapshot_path)
        rel_key = "/".join(rel.parts)
        s3_key = f"{prefix}/{rel_key}"
        run(
            [
                "aws",
                "s3",
                "cp",
                str(path),
                f"s3://{args.bucket}/{s3_key}",
                "--content-type",
                content_type(path),
                *s3_cp_args(args.kms_key_id),
            ]
        )

    run(
        [
            "aws",
            "s3",
            "cp",
            str(manifest_path),
            f"s3://{args.bucket}/{prefix}/manifest.json",
            "--content-type",
            "application/json",
            "--cache-control",
            "no-store",
            *s3_cp_args(args.kms_key_id),
        ]
    )

    if args.cloudfront_distribution_id:
        run(
            [
                "aws",
                "cloudfront",
                "create-invalidation",
                "--distribution-id",
                args.cloudfront_distribution_id,
                "--paths",
                f"/{prefix}/*",
            ]
        )


def publish_probe(args: argparse.Namespace) -> None:
    prefix = dry_run_probe_prefix(args.cache_root)
    probe_key = f"{prefix}/probe.json"
    probe_path = Path(args.work_dir) / "dry-run-publish-probe.json"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "probe": "llima-safetensors",
        "cache_root": args.cache_root,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": {
            "repository": os.environ.get("GITHUB_REPOSITORY", ""),
            "run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
            "sha": os.environ.get("GITHUB_SHA", ""),
        },
    }
    probe_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Dry-run publish probe: writing s3://{args.bucket}/{probe_key}")
    run(
        [
            "aws",
            "s3",
            "cp",
            str(probe_path),
            f"s3://{args.bucket}/{probe_key}",
            "--content-type",
            "application/json",
            "--cache-control",
            "no-store",
            *s3_cp_args(args.kms_key_id),
        ]
    )
    run(["aws", "s3", "cp", f"s3://{args.bucket}/{probe_key}", "-"], capture=True)
    run(["aws", "s3", "rm", f"s3://{args.bucket}/{probe_key}"])
    print(f"Dry-run publish probe complete: s3://{args.bucket}/{probe_key}")


def process_model(api: HfApi, args: argparse.Namespace, spec: ModelSpec) -> str:
    cache_root = {
        "config": args.config_cache_root,
        "gguf": args.gguf_cache_root,
        "safetensors": args.cache_root,
    }[spec.payload_format]
    prefix = cache_prefix(cache_root, spec.repo_id)
    print(f"Resolving {spec.repo_id}@{spec.revision}")
    token = (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
        or None
    )
    info = api.model_info(repo_id=spec.repo_id, revision=spec.revision, token=token)
    resolved_revision = info.sha
    if not resolved_revision:
        fail(f"Unable to resolve Hugging Face revision for {spec.repo_id}@{spec.revision}")
    if spec.payload_format == "gguf":
        if spec.file_patterns:
            allow_patterns = validate_gguf_files(
                info, spec.repo_id, spec.file_patterns
            )
        else:
            allow_patterns = [select_gguf_file(info, spec.repo_id)]
    elif spec.payload_format == "config":
        allow_patterns = CONFIG_ALLOW_PATTERNS
    else:
        allow_patterns = SAFETENSORS_ALLOW_PATTERNS
    desired_selection_sha256 = selection_fingerprint(
        spec.payload_format, allow_patterns
    )
    if args.resolve_only:
        detail = (
            f" ({', '.join(allow_patterns)})"
            if spec.payload_format == "gguf"
            else ""
        )
        print(f"Resolved {spec.repo_id}@{spec.revision}: {resolved_revision}{detail}")
        return "resolved"

    cached_manifest = None if args.dry_run else fetch_cached_manifest(args.bucket, prefix)
    cached_revision = ""
    cached_selection_sha256 = ""
    if cached_manifest:
        cached_revision = str(cached_manifest.get("resolved_revision", ""))
        cached_selection_sha256 = str(
            cached_manifest.get("selection_sha256", "")
        )

    if (
        cached_revision == resolved_revision
        and cached_selection_sha256 == desired_selection_sha256
        and not args.force
    ):
        print(f"Cache current for {spec.repo_id}: {resolved_revision}")
        return "current"

    if args.force:
        print(f"Force sync requested for {spec.repo_id}: hf={resolved_revision}")
    elif cached_revision == resolved_revision:
        print(
            f"Cache selection stale for {spec.repo_id}: "
            f"cached={cached_selection_sha256 or '<missing>'}, "
            f"desired={desired_selection_sha256}"
        )
    elif cached_revision:
        print(f"Cache stale for {spec.repo_id}: cached={cached_revision}, hf={resolved_revision}")
    else:
        print(f"Cache missing for {spec.repo_id}: hf={resolved_revision}")

    model_work_dir = Path(args.work_dir) / spec.repo_id.replace("/", "__")
    if model_work_dir.exists():
        shutil.rmtree(model_work_dir)
    model_work_dir.parent.mkdir(parents=True, exist_ok=True)

    snapshot_path = Path(
        snapshot_download(
            repo_id=spec.repo_id,
            revision=resolved_revision,
            local_dir=model_work_dir,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            token=token,
        )
    )
    files = find_downloaded_files(snapshot_path)
    if not files:
        fail(f"No cache files matched for {spec.repo_id}@{resolved_revision}")
    if spec.payload_format == "safetensors" and not has_safetensors(files):
        fail(f"No safetensors files matched for {spec.repo_id}@{resolved_revision}")
    if spec.payload_format == "config" and not any(
        path.name == "config.json" for path in files
    ):
        fail(f"No config.json matched for {spec.repo_id}@{resolved_revision}")
    if spec.payload_format == "gguf" and (
        len(files) != len(allow_patterns)
        or any(path.suffix.lower() != ".gguf" for path in files)
    ):
        fail(
            f"Expected {len(allow_patterns)} GGUF file(s) for "
            f"{spec.repo_id}@{resolved_revision}, "
            f"found: {', '.join(str(path) for path in files)}"
        )

    manifest = build_manifest(
        spec=spec,
        resolved_revision=resolved_revision,
        prefix=prefix,
        files=files,
        snapshot_path=snapshot_path,
        base_url=args.base_url,
        payload_format=spec.payload_format,
        selection_sha256=desired_selection_sha256,
    )
    publish_model(
        args=args,
        spec=spec,
        resolved_revision=resolved_revision,
        prefix=prefix,
        snapshot_path=snapshot_path,
        files=files,
        manifest=manifest,
    )
    shutil.rmtree(model_work_dir, ignore_errors=True)
    return "updated"


def main() -> int:
    args = parse_args()
    specs = resolve_specs(args)
    if args.dry_run:
        print("Dry run enabled; model payload publishing is skipped.")
    if args.resolve_only:
        print("Resolve-only enabled; downloads and publishing are skipped.")
    if args.force:
        print("Force sync enabled; models will be downloaded even when cache is current.")
    if args.publish_probe and not args.dry_run:
        fail("--publish-probe can only be used with --dry-run")
    if args.publish_probe:
        publish_probe(args)

    api = HfApi()
    summary = {"current": 0, "updated": 0, "resolved": 0, "failed": 0}
    failures: list[dict[str, str]] = []
    for spec in specs:
        try:
            status = process_model(api, args, spec)
        except SystemExit as exc:
            if not args.resolve_only:
                raise
            status = "failed"
            message = str(exc)
            print(f"::error title=Failed to resolve {spec.repo_id}::{message}")
            failures.append(
                {
                    "repo_id": spec.repo_id,
                    "revision": spec.revision,
                    "error": message,
                }
            )
        except Exception as exc:
            if not args.resolve_only:
                raise
            status = "failed"
            message = f"{type(exc).__name__}: {exc}"
            print(f"::error title=Failed to resolve {spec.repo_id}::{message}")
            failures.append(
                {
                    "repo_id": spec.repo_id,
                    "revision": spec.revision,
                    "error": message,
                }
            )
        summary[status] = summary.get(status, 0) + 1

    result = {"processed": len(specs), **summary}
    if failures:
        result["failures"] = failures
    print(json.dumps(result, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
