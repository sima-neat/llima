#!/usr/bin/env python3
"""Classify which expensive LLiMa test suites a branch must run."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


SHARED_PATHS = frozenset(
    {
        ".github/workflows/vulcan-ci.yml",
        "deps/manifest.json",
        "pyproject.toml",
        "pyproject_metadata.py",
        "sima_lmm/VERSION",
        "sima_lmm/__init__.py",
        "tests/ci/test_resolve_test_scope.py",
        "tools/ci/resolve_test_scope.py",
        "tools/compute_package_version.sh",
    }
)

COMPILER_PATHS = frozenset(
    {
        ".github/workflows/cache-hf-safetensors.yml",
        ".github/workflows/model-compiler-tests.yml",
        "build_compiler_wheel.sh",
        "pytest.ini",
        "scripts/ci/install_sima_cli_main.sh",
        "scripts/gen_models--openai--whisper.py",
        "scripts/generate_reference_configs.py",
        "sima_lmm/host/__init__.py",
        "sima_lmm/host/compile_lmm.py",
        "sima_lmm/host/compile_lora_adapter.py",
        "sima_lmm/host/configuration_helper.py",
        "sima_lmm/utils.py",
        "tests/ci/test_prepare_model_inputs.py",
        "tools/ci/audit_pytest_report.py",
        "tools/ci/prepare_model_inputs.py",
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

DEVKIT_PATHS = frozenset(
    {
        ".gitmodules",
        "CMakeLists.txt",
        "build.sh",
        "toolchain-sima.cmake",
        "tools/install_llima.sh",
    }
)

DEVKIT_PATH_PREFIXES = (
    "cmake/",
    "sima_lmm/assets/",
    "sima_lmm/devkit/",
    "tests/runtime/",
    "third_party/",
)

TEST_NEUTRAL_PATHS = frozenset(
    {
        ".github/CODEOWNERS",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/workflows/pr-sanitization.yml",
        ".github/workflows/release.yml",
        ".gitignore",
        "AGENTS.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "README.md",
        "build_mole_package.sh",
        "sima_lmm/host/benchmark.py",
        "sima_lmm/host/deploy_lmm.py",
        "sima_lmm/host/deploy_lora.py",
        "tests/README.md",
        "tests/__init__.py",
        "tests/test_cmake_package_compatibility.py",
        "tests/test_install_llima_safety.py",
        "tools/install_mole.sh",
    }
)

TEST_NEUTRAL_PATH_PREFIXES = (
    ".github/ISSUE_TEMPLATE/",
    "docs/",
    "sima_lmm/mole/",
    "skills/",
)


@dataclass(frozen=True)
class ScopeDecision:
    run_compiler: bool
    run_devkit: bool
    compiler_reason: str
    devkit_reason: str
    changed_paths: tuple[str, ...] = ()
    compiler_paths: tuple[str, ...] = ()
    devkit_paths: tuple[str, ...] = ()
    neutral_paths: tuple[str, ...] = ()
    unknown_paths: tuple[str, ...] = ()


def normalize_changed_path(raw_path: str) -> str:
    path = raw_path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def _matches(
    path: str,
    exact_paths: frozenset[str],
    prefixes: tuple[str, ...] = (),
) -> bool:
    return path in exact_paths or path.startswith(prefixes)


def is_compiler_path(raw_path: str) -> bool:
    path = normalize_changed_path(raw_path)
    return _matches(
        path,
        SHARED_PATHS | COMPILER_PATHS,
        COMPILER_PATH_PREFIXES,
    )


def is_devkit_path(raw_path: str) -> bool:
    path = normalize_changed_path(raw_path)
    return _matches(path, SHARED_PATHS | DEVKIT_PATHS, DEVKIT_PATH_PREFIXES)


def is_test_neutral_path(raw_path: str) -> bool:
    path = normalize_changed_path(raw_path)
    return _matches(path, TEST_NEUTRAL_PATHS, TEST_NEUTRAL_PATH_PREFIXES)


def _suite_reason(
    suite: str,
    *,
    run: bool,
    matching_paths: tuple[str, ...],
    unknown_paths: tuple[str, ...],
    head_ref: str,
    base_ref: str,
) -> str:
    if unknown_paths:
        return (
            f"{suite} tests required: branch {head_ref} contains "
            f"{len(unknown_paths)} unclassified path(s) relative to "
            f"{base_ref}; "
            "running fail-safe coverage."
        )
    if run:
        return (
            f"{suite} tests required: branch {head_ref} changes "
            f"{len(matching_paths)} {suite.lower()}-impacting path(s) "
            f"relative to {base_ref}."
        )
    return (
        f"{suite} tests skipped: branch {head_ref} contains no explicitly "
        f"classified {suite.lower()}-impacting changes relative to {base_ref}."
    )


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
    compiler_paths = tuple(
        path for path in normalized_paths if is_compiler_path(path)
    )
    devkit_paths = tuple(
        path for path in normalized_paths if is_devkit_path(path)
    )
    neutral_paths = tuple(
        path for path in normalized_paths if is_test_neutral_path(path)
    )
    classified_paths = (
        set(compiler_paths) | set(devkit_paths) | set(neutral_paths)
    )
    unknown_paths = tuple(
        path for path in normalized_paths if path not in classified_paths
    )

    run_compiler = bool(compiler_paths or unknown_paths)
    run_devkit = bool(devkit_paths or unknown_paths)
    compiler_reason = _suite_reason(
        "Compiler",
        run=run_compiler,
        matching_paths=compiler_paths,
        unknown_paths=unknown_paths,
        head_ref=head_ref,
        base_ref=base_ref,
    )
    devkit_reason = _suite_reason(
        "DevKit runtime",
        run=run_devkit,
        matching_paths=devkit_paths,
        unknown_paths=unknown_paths,
        head_ref=head_ref,
        base_ref=base_ref,
    )
    return ScopeDecision(
        run_compiler=run_compiler,
        run_devkit=run_devkit,
        compiler_reason=compiler_reason,
        devkit_reason=devkit_reason,
        changed_paths=normalized_paths,
        compiler_paths=compiler_paths,
        devkit_paths=devkit_paths,
        neutral_paths=neutral_paths,
        unknown_paths=unknown_paths,
    )


def force_run(reason: str) -> ScopeDecision:
    reason = reason.strip()
    return ScopeDecision(
        run_compiler=True,
        run_devkit=True,
        compiler_reason=reason,
        devkit_reason=reason,
    )


def write_github_output(path: Path, decision: ScopeDecision) -> None:
    values: tuple[tuple[str, object], ...] = (
        ("run_compiler", str(decision.run_compiler).lower()),
        ("run_devkit", str(decision.run_devkit).lower()),
        ("compiler_reason", decision.compiler_reason),
        ("devkit_reason", decision.devkit_reason),
        ("compiler_paths_json", decision.compiler_paths),
        ("devkit_paths_json", decision.devkit_paths),
        ("unknown_paths_json", decision.unknown_paths),
    )
    with path.open("a", encoding="utf-8") as output:
        for name, value in values:
            if isinstance(value, tuple):
                value = json.dumps(value, separators=(",", ":"))
            output.write(f"{name}={value}\n")


def escape_markdown(value: str) -> str:
    return value.replace("\\", "\\\\").replace("`", "\\`").replace("|", "\\|")


def write_github_summary(path: Path, decision: ScopeDecision) -> None:
    with path.open("a", encoding="utf-8") as summary:
        summary.write("### Test scope\n\n")
        summary.write("| Suite | Decision | Matched paths |\n")
        summary.write("| --- | --- | ---: |\n")
        summary.write(
            f"| Compiler | `{'run' if decision.run_compiler else 'skip'}` | "
            f"{len(decision.compiler_paths)} |\n"
        )
        summary.write(
            "| DevKit runtime | "
            f"`{'run' if decision.run_devkit else 'skip'}` | "
            f"{len(decision.devkit_paths)} |\n\n"
        )
        summary.write(f"- {decision.compiler_reason}\n")
        summary.write(f"- {decision.devkit_reason}\n")
        summary.write(
            f"- Changed paths inspected: {len(decision.changed_paths)}\n"
        )
        summary.write(f"- Test-neutral paths: {len(decision.neutral_paths)}\n")
        summary.write(f"- Unclassified paths: {len(decision.unknown_paths)}\n")

        path_groups = (
            ("Compiler-impacting paths", decision.compiler_paths),
            ("DevKit-impacting paths", decision.devkit_paths),
            ("Unclassified fail-safe paths", decision.unknown_paths),
        )
        for heading, paths in path_groups:
            if not paths:
                continue
            summary.write(f"\n{heading}:\n\n")
            for matching_path in paths:
                summary.write(f"- `{escape_markdown(matching_path)}`\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve which expensive LLiMa test suites must run."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--changed-files",
        type=Path,
        help="Newline-delimited branch file list to classify.",
    )
    source.add_argument(
        "--force-run-reason",
        help=(
            "Run compiler and DevKit tests without classifying changed files."
        ),
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
        changed_paths = args.changed_files.read_text(
            encoding="utf-8"
        ).splitlines()
        decision = classify_changed_paths(
            changed_paths,
            head_ref=args.head_ref,
            base_ref=args.base_ref,
        )
    else:
        decision = force_run(args.force_run_reason)

    write_github_output(args.github_output, decision)
    write_github_summary(args.github_summary, decision)
    print(decision.compiler_reason)
    print(decision.devkit_reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
