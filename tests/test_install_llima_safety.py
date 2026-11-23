import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.premerge
BASH_MAJOR = int(
    re.search(
        r"version (\d+)",
        subprocess.run(
            ["bash", "--version"], check=True, capture_output=True, text=True
        ).stdout,
    ).group(1)
)
requires_associative_arrays = pytest.mark.skipif(
    BASH_MAJOR < 4, reason="installer requires Bash 4 associative arrays"
)


def build_deb(
    directory: Path,
    package: str,
    version: str,
    *,
    provides: str = "",
    replaces: str = "",
    conflicts: str = "",
) -> Path:
    package_root = directory / f"{package}-root"
    control_dir = package_root / "DEBIAN"
    control_dir.mkdir(parents=True)
    fields = [
        f"Package: {package}",
        f"Version: {version}",
        "Architecture: arm64",
        "Maintainer: test <test@example.com>",
        f"Description: test package {package}",
    ]
    for name, value in (
        ("Provides", provides),
        ("Replaces", replaces),
        ("Conflicts", conflicts),
    ):
        if value:
            fields.append(f"{name}: {value}")
    (control_dir / "control").write_text("\n".join(fields) + "\n")
    deb = directory / f"{package}_{version}_arm64.deb"
    subprocess.run(
        ["dpkg-deb", "--build", str(package_root), str(deb)],
        check=True,
        capture_output=True,
        text=True,
    )
    return deb


def write_command(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)


def run_installer(
    tmp_path: Path,
    *removed_packages: str,
    extra_package: str = "",
    additional_removal: str = "",
) -> subprocess.CompletedProcess:
    bundle = tmp_path / "bundle"
    commands = tmp_path / "commands"
    bundle.mkdir()
    commands.mkdir()

    debs = [
        build_deb(bundle, package, "0.4.0")
        for package in ("sima-lmm-core", "sima-lmm-cli", "sima-lmm-dev")
    ]
    if extra_package:
        debs.append(build_deb(bundle, extra_package, "0.4.0"))
    (bundle / "llima-install-manifest.txt").write_text(
        "".join(f"{deb.name}\n" for deb in debs)
    )

    write_command(
        commands / "apt-get",
        """
if [[ " $* " == *" --simulate "* ]]; then
  cat "$TEST_APT_PLAN"
else
  printf '%s\\n' "$@" > "$TEST_APT_ARGS"
  for option in "$@"; do
    if [[ "$option" == DPkg::Pre-Install-Pkgs::* ]]; then
      bash -c "${option#*=}" < "$TEST_APT_ACTIONS"
    fi
  done
fi
""",
    )
    write_command(
        commands / "dpkg-query",
        """
package="${!#}"
case "${package}" in
  sima-lmm-core|sima-lmm-cli|sima-lmm-dev) echo "0.4.0" ;;
  *) exit 1 ;;
esac
""",
    )
    write_command(
        commands / "sudo",
        """
if [[ "${1:-}" == "-n" ]]; then
  exit 0
fi
exec "$@"
""",
    )
    write_command(commands / "llima", "exit 0\n")
    installer = bundle / "install_llima.sh"
    installer.write_bytes((ROOT / "tools/install_llima.sh").read_bytes())
    installer.chmod(0o755)
    plan = tmp_path / "apt-plan"
    plan.write_text("".join(f"Remv {package} [3.0.0]\n" for package in removed_packages))
    actions = tmp_path / "apt-actions"
    actual_removals = list(removed_packages)
    if additional_removal:
        actual_removals.append(additional_removal)
    actions.write_text(
        "VERSION 2\n\n"
        + "".join(f"{package.split(':')[0]} 3.0.0 > - **REMOVE**\n" for package in actual_removals)
    )
    env = os.environ.copy()
    env["TEST_APT_ACTIONS"] = str(actions)
    env["TEST_APT_PLAN"] = str(plan)
    env["TEST_APT_ARGS"] = str(tmp_path / "apt-args")
    env["PATH"] = f"{commands}:{env['PATH']}"
    env["LLIMA_INSTALLER_SKIP_PLATFORM_CHECK"] = "ON"
    return subprocess.run(
        [str(installer)],
        cwd=bundle,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@requires_associative_arrays
def test_installer_accepts_the_three_llima_packages(tmp_path: Path) -> None:
    result = run_installer(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--no-remove" in (tmp_path / "apt-args").read_text().splitlines()
    assert "Validated 3 Debian package(s)" in result.stdout
    assert "LLiMa 0.4.0 installed successfully" in result.stdout


@requires_associative_arrays
@pytest.mark.parametrize("packages", [("sima-neat",), ("sima-neat:arm64", "sima-neat-dev")])
def test_installer_allows_core_removal(tmp_path: Path, packages: tuple[str, ...]) -> None:
    result = run_installer(tmp_path, *packages)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "LLiMa 0.4.0 installed successfully" in result.stdout
    removed = " ".join(f"{package.split(':')[0]} 3.0.0" for package in packages)
    assert f"Removed Core packages: {removed}." in result.stdout
    assert "Install a matching Core" in result.stdout
    assert "--no-remove" not in (tmp_path / "apt-args").read_text().splitlines()


@requires_associative_arrays
@pytest.mark.parametrize("packages", [("simaai-common",), ("sima-neat", "simaai-common")])
def test_installer_rejects_removing_unrelated_platform_package(
    tmp_path: Path, packages: tuple[str, ...]
) -> None:
    result = run_installer(tmp_path, *packages)

    assert result.returncode != 0
    assert not (tmp_path / "apt-args").exists()
    assert "Refusing to install because APT would remove simaai-common" in result.stderr


@requires_associative_arrays
def test_installer_rejects_unplanned_removal_before_dpkg(tmp_path: Path) -> None:
    result = run_installer(tmp_path, "sima-neat", additional_removal="simaai-common")

    assert result.returncode != 0
    assert "Refusing to remove simaai-common" in result.stderr
    assert "LLiMa 0.4.0 installed successfully" not in result.stdout


@requires_associative_arrays
def test_installer_rejects_non_llima_packages(tmp_path: Path) -> None:
    result = run_installer(tmp_path, extra_package="neat-runtime")

    assert result.returncode != 0
    assert "Install bundle contains unexpected package neat-runtime" in result.stderr
