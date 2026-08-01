#!/usr/bin/env python3
"""Classify whether a branch changes LLiMa compiler-related files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


COMPILER_PATHS = frozenset(
    {
        ".github/workflows/model-compiler-tests.yml",
        "build_compiler_wheel.sh",
        "deps/manifest.json",
        "pyproject.toml",
        "pyproject_metadata.py",
        "pytest.ini",
        "scripts/ci/install_sima_cli_main.sh",
        "sima_lmm/VERSION",
        "sima_lmm/__init__.py",
        "sima_lmm/host/__init__.py",
        "sima_lmm/host/compile_lmm.py",
        "sima_lmm/host/compile_lora_adapter.py",
        "sima_lmm/host/configuration_helper.py",
        "sima_lmm/utils.py",
        "tests/ci/test_resolve_compiler_test_scope.py",
        "tools/ci/audit_pytest_report.py",
        "tools/ci/prepare_model_inputs.py",
        "tools/ci/resolve_compiler_test_scope.py",
        "tools/compute_package_version.sh",
        "tools/ensure_wheel_build_env.sh",
        "tools/install_compiler.sh",
    }
)

COMPILER_PATH_PREFIXES = (
    "sima_lmm/config/",
    "sima_lmm/gguf/",
    "sima_lmm/hf/",
    "sima_lmm/model/",
    "sima_lmm/preproc/",
    "sima_lmm/tokenizer/",
    "tests/compilation/",
    "tools/hf-safetensors/",
)


@dataclass(frozen=True)
class ScopeDecision:
    run_compiler: bool
    reason: str
    changed_paths: tuple[str, ...] = ()
    matching_paths: tuple[str, ...] = ()


def normalize_changed_path(raw_path: str) -> str:
    path = raw_path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def is_compiler_path(raw_path: str) -> bool:
    path = normalize_changed_path(raw_path)
    return path in COMPILER_PATHS or path.startswith(COMPILER_PATH_PREFIXES)


def classify_changed_paths(
    changed_paths: list[str],
    *,
    head_ref: str,
    base_ref: str,
) -> ScopeDecision:
    normalized_paths = tuple(
        sorted(
            {
                path
                for raw_path in changed_paths
                if (path := normalize_changed_path(raw_path))
            }
        )
    )
    matching_paths = tuple(path for path in normalized_paths if is_compiler_path(path))

    if matching_paths:
        reason = (
            f"Compiler tests required: branch {head_ref} changes "
            f"{len(matching_paths)} compiler-impacting path(s) relative to {base_ref}."
        )
        return ScopeDecision(
            run_compiler=True,
            reason=reason,
            changed_paths=normalized_paths,
            matching_paths=matching_paths,
        )

    reason = (
        f"Compiler tests skipped: branch {head_ref} contains no explicitly "
        f"classified compiler-impacting changes relative to {base_ref}."
    )
    return ScopeDecision(
        run_compiler=False,
        reason=reason,
        changed_paths=normalized_paths,
    )


def force_run(reason: str) -> ScopeDecision:
    return ScopeDecision(run_compiler=True, reason=reason.strip())


def write_github_output(path: Path, decision: ScopeDecision) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(f"run_compiler={str(decision.run_compiler).lower()}\n")
        output.write(f"reason={decision.reason}\n")
        output.write(
            "matching_paths_json="
            f"{json.dumps(decision.matching_paths, separators=(',', ':'))}\n"
        )


def escape_markdown(value: str) -> str:
    return value.replace("\\", "\\\\").replace("`", "\\`").replace("|", "\\|")


def write_github_summary(path: Path, decision: ScopeDecision) -> None:
    with path.open("a", encoding="utf-8") as summary:
        summary.write("### Compiler test scope\n\n")
        summary.write(f"{decision.reason}\n\n")
        summary.write(
            f"- Decision: `{'run' if decision.run_compiler else 'skip'}`\n"
        )
        summary.write(f"- Changed paths inspected: {len(decision.changed_paths)}\n")
        summary.write(f"- Compiler-impacting paths: {len(decision.matching_paths)}\n")
        if decision.matching_paths:
            summary.write("\nMatched paths:\n\n")
            for matching_path in decision.matching_paths:
                summary.write(f"- `{escape_markdown(matching_path)}`\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve whether the expensive LLiMa compiler tests must run."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--changed-files",
        type=Path,
        help="Newline-delimited branch file list to classify.",
    )
    source.add_argument(
        "--force-run-reason",
        help="Run compiler tests without classifying a changed-file list.",
    )
    parser.add_argument("--head-ref")
    parser.add_argument("--base-ref", default="develop")
    parser.add_argument("--github-output", type=Path, required=True)
    parser.add_argument("--github-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.changed_files:
        if args.head_ref is None:
            raise SystemExit("--head-ref is required with --changed-files")
        changed_paths = args.changed_files.read_text(encoding="utf-8").splitlines()
        decision = classify_changed_paths(
            changed_paths,
            head_ref=args.head_ref,
            base_ref=args.base_ref,
        )
    else:
        decision = force_run(args.force_run_reason)

    write_github_output(args.github_output, decision)
    write_github_summary(args.github_summary, decision)
    print(decision.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
