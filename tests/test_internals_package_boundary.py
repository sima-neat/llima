"""LLiMa consumes one selected Internals release without reinterpreting it."""

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.premerge


def cmake() -> str:
    return (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")


def build_script() -> str:
    return (ROOT / "build.sh").read_text(encoding="utf-8")


def test_internals_is_located_without_a_derived_version() -> None:
    """The selected artifact decides the release; LLiMa does not ask for one."""
    assert "find_package(NeatInternals CONFIG REQUIRED)" in cmake()
    assert "find_package(NeatInternals ${SIMA_LMM_PLATFORM_VERSION}" not in cmake()


def test_no_manually_constructed_internals_version_ranges() -> None:
    text = cmake()
    for removed in (
        "SIMA_LMM_INTERNALS_PACKAGE_MIN_VERSION",
        "SIMA_LMM_INTERNALS_PACKAGE_MAX_VERSION",
        "SIMA_LMM_MEMORY_LIB_VERSION",
        "SIMA_LMM_MEMORY_LIB_DEPENDENCY",
    ):
        assert removed not in text, removed

    # Internals packages are named without a hand-built range.  Dependencies
    # that Debian tooling derives from real ELF links are untouched.
    assert "neat-runtime, " in text
    assert "neat-internals-dev, " in text
    assert "neat-runtime (>=" not in text
    assert "neat-runtime (<<" not in text
    assert "neat-internals-dev (>=" not in text
    assert "neat-internals-dev (<<" not in text


def test_memory_is_not_looked_up_in_the_internals_artifact() -> None:
    """Memory comes from the platform, not from what Internals delivered."""
    text = build_script()
    assert "resolve_neat_internals_memory_version" not in text
    assert "SIMA_LMM_MEMORY_LIB_VERSION" not in text
    assert "simaai-memory-lib" not in text


def test_every_delivered_internals_package_is_installed_and_forwarded() -> None:
    text = build_script()
    # No package whitelist: whatever the artifact contains is installed.
    assert "deb_pattern_groups" not in text
    assert "internals_patterns" not in text
    assert (
        'mapfile -t debs < <(find "${NEAT_INTERNALS_DEB_DIR}" -maxdepth 1 '
        "-type f -name '*.deb' | sort)" in text
    )
    # And the publishable artifact forwards every cached Internals package.
    assert (
        'mapfile -t internals_debs < <(find "${NEAT_INTERNALS_DEB_DIR}" '
        "-maxdepth 1 -type f -name '*.deb' | sort)" in text
    )


def test_llima_has_no_recovery_or_libcamera_branch() -> None:
    """Recovery is an Internals build option; LLiMa never learns about it."""
    text = build_script()
    assert "recovery" not in text.lower()
    assert "libcamera" not in text.lower()


def test_manifest_still_selects_only_environment_and_release() -> None:
    manifest = json.loads((ROOT / "deps/manifest.json").read_text(encoding="utf-8"))
    assert "platform-package-version" not in manifest
