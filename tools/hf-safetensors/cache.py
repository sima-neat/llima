#!/usr/bin/env python3
"""Cache latest Hugging Face safetensors inputs in Vulcan artifacts."""

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


ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "model.safetensors.index.json",
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


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache latest Hugging Face safetensors payloads in Vulcan S3."
    )
    parser.add_argument("--manifest", default="tools/hf-safetensors/manifest.txt")
    parser.add_argument("--repo-id", default="", help="Optional single model override.")
    parser.add_argument("--revision", default="main", help="Revision for --repo-id.")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--cache-root", default="llima-safetensors")
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


def parse_manifest(path: Path) -> list[ModelSpec]:
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
        specs.append(ModelSpec(repo_id=repo_id, revision=revision))

    if not specs:
        fail(f"Manifest did not contain any model entries: {path}")
    return specs


def resolve_specs(args: argparse.Namespace) -> list[ModelSpec]:
    if args.repo_id.strip():
        return [ModelSpec(validate_repo_id(args.repo_id), args.revision.strip() or "main")]
    return parse_manifest(Path(args.manifest))


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


def cloudfront_url(base_url: str, s3_key: str) -> str:
    return f"{base_url.rstrip('/')}/{s3_key}"


def dry_run_probe_prefix(cache_root: str) -> str:
    run_id = os.environ.get("GITHUB_RUN_ID") or "local"
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT") or "0"
    return f"{cache_root.strip('/')}/_dry-run/{run_id}-{run_attempt}"


def cache_prefix(cache_root: str, repo_id: str) -> str:
    return f"{cache_root.strip('/')}/{repo_id}/latest"


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
    prefix = cache_prefix(args.cache_root, spec.repo_id)
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
    if args.resolve_only:
        print(f"Resolved {spec.repo_id}@{spec.revision}: {resolved_revision}")
        return "resolved"

    cached_manifest = None if args.dry_run else fetch_cached_manifest(args.bucket, prefix)
    cached_revision = ""
    if cached_manifest:
        cached_revision = str(cached_manifest.get("resolved_revision", ""))

    if cached_revision == resolved_revision and not args.force:
        print(f"Cache current for {spec.repo_id}: {resolved_revision}")
        return "current"

    if cached_revision:
        print(f"Cache stale for {spec.repo_id}: cached={cached_revision}, hf={resolved_revision}")
    elif args.force:
        print(f"Force sync requested for {spec.repo_id}: hf={resolved_revision}")
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
            allow_patterns=ALLOW_PATTERNS,
            token=token,
        )
    )
    files = find_downloaded_files(snapshot_path)
    if not files:
        fail(f"No cache files matched for {spec.repo_id}@{resolved_revision}")
    if not has_safetensors(files):
        fail(f"No safetensors files matched for {spec.repo_id}@{resolved_revision}")

    manifest = build_manifest(
        spec=spec,
        resolved_revision=resolved_revision,
        prefix=prefix,
        files=files,
        snapshot_path=snapshot_path,
        base_url=args.base_url,
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
