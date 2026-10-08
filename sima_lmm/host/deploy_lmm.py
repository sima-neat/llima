import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from sima_lmm.host.remote_destination import (
    RemoteSession,
    local_size_kb,
    parse_remote_destination,
    report_free_space,
)


def _abort(message):
    """
    Terminate with a user-facing error message.
    """
    print(message, file=sys.stderr)
    sys.exit(-1)


def _speculative_role(sima_files_dir: Path) -> bool | None:
    config_file = sima_files_dir / "devkit" / "vlm_config.json"
    if not config_file.is_file():
        return None
    with config_file.open() as config_stream:
        config = json.load(config_stream)
    spec_config = config.get("lm_cfg", {}).get("speculative_decoding_cfg")
    if spec_config is None:
        return None
    return bool(spec_config.get("is_draft", False))


def _resolve_deploy_sources(src_dir: Path) -> list[tuple[str | None, Path]]:
    """Resolve one normal compiler output or a speculative target/draft parent."""
    if not src_dir.is_dir():
        raise RuntimeError(f"Source directory does not exist or is not a directory: {src_dir}")

    direct_sima_dir = src_dir if src_dir.name == "sima_files" else src_dir / "sima_files"
    if (direct_sima_dir / "devkit").is_dir():
        return [(None, direct_sima_dir)]

    target: tuple[str, Path] | None = None
    draft: tuple[str, Path] | None = None
    for child in sorted(src_dir.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        sima_dir = child / "sima_files"
        role = _speculative_role(sima_dir)
        if role is None:
            continue
        if role:
            if draft is not None:
                raise RuntimeError(
                    f"Multiple speculative draft models found under {src_dir}: "
                    f"{draft[0]} and {child.name}"
                )
            draft = (child.name, sima_dir)
        else:
            if target is not None:
                raise RuntimeError(
                    f"Multiple speculative target models found under {src_dir}: "
                    f"{target[0]} and {child.name}"
                )
            target = (child.name, sima_dir)

    if target is None and draft is None:
        raise RuntimeError(
            f"No compiled model found under {src_dir}. Expected sima_files/ or "
            "speculative target/draft children containing sima_files/."
        )
    if target is None or draft is None:
        missing = "target" if target is None else "draft"
        raise RuntimeError(f"Speculative model under {src_dir} is missing its {missing} model.")
    return [target, draft]


def _validate_sima_files(src_sima_dir: Path) -> None:
    if not (src_sima_dir / "devkit").is_dir():
        raise RuntimeError(f"devkit directory cannot be found in {src_sima_dir}")
    if not (src_sima_dir / "mpk").is_dir():
        raise RuntimeError(f"mpk directory cannot be found in {src_sima_dir}")


def _deploy_sima_files(src_sima_dir: Path, dst_dir: Path | str, rsh: str | None = None) -> None:
    """Deploy one direct sima_files directory to a runtime model directory."""
    src_devkit_dir = src_sima_dir / "devkit"
    src_mpk_dir = src_sima_dir / "mpk"

    # Extract the elf_files.
    src_elf_dir = src_sima_dir / "elf_files"
    src_elf_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".sh") as f:
        cmd = """
            for f in ../mpk/*.tar.gz
            do
                mpk_file=$f
                elf_file=${f:7:-11}_stage1_mla.elf

                if [ -f $elf_file ]
                then
                    mpk_file_time=$(stat -c %Z $mpk_file)
                    elf_file_time=$(stat -c %Z $elf_file)
                    if [ $mpk_file_time -gt $elf_file_time ]
                    then
                        tar xzvf $mpk_file $elf_file
                    fi
                else
                    tar xzvf $mpk_file $elf_file
                fi
            done
        """
        f.write(cmd)
        f.flush()
        subprocess.check_call(f"bash {f.name}", cwd=src_elf_dir, shell=True)

    # Collect the src dirs to copy.
    src_dirs = [src_devkit_dir, src_elf_dir]
    src_npy_dir = src_devkit_dir.parent / "npy_files"
    if src_npy_dir.is_dir():
        src_dirs.append(src_npy_dir)

    # Use rsync to copy the data to the destination.
    rsh_args = ["-e", rsh] if rsh else []
    cmd = ["rsync", "-aP", "--mkpath", *rsh_args, *src_dirs, dst_dir]
    subprocess.check_call(cmd)


def _deployment_size_kb(sources: list[tuple[str | None, Path]]) -> int:
    """Upper-bound estimate of the deployed size, taken before ELF extraction."""
    paths = []
    for _, src_sima_dir in sources:
        paths += [src_sima_dir / "devkit", src_sima_dir / "mpk", src_sima_dir / "npy_files"]
    return local_size_kb(paths)


def deploy(
    src_dir: Path, dst_dir: Path | str, rsh: str | None = None, preflight: bool = True
) -> None:
    sources = _resolve_deploy_sources(src_dir)
    for _, src_sima_dir in sources:
        _validate_sima_files(src_sima_dir)

    remote = parse_remote_destination(str(dst_dir))

    def model_destination(model_name: str | None) -> Path | str:
        if remote is None:
            return Path(dst_dir) if model_name is None else Path(dst_dir) / model_name
        return str(remote) if model_name is None else remote.join(model_name)

    if remote is None or not preflight:
        for model_name, src_sima_dir in sources:
            _deploy_sima_files(src_sima_dir, model_destination(model_name), rsh)
        return

    # Check the remote destination once, before the slow ELF extraction, and
    # reuse the same SSH connection for every transfer.
    with RemoteSession(remote.host, rsh) as session:
        print(f"Checking {remote} ...", flush=True)
        free_kb = session.preflight(remote.path)
        report_free_space(str(remote), free_kb, _deployment_size_kb(sources))
        for model_name, src_sima_dir in sources:
            _deploy_sima_files(src_sima_dir, model_destination(model_name), session.rsync_rsh)


def main():
    parser = argparse.ArgumentParser(description="LMM deploy utility")
    parser.add_argument(
        "src_dir", type=Path,
        help="Path to the source directory with compiled mpk tar.gz files and devkit files"
    )
    parser.add_argument(
        "dst_dir",
        help="Destination directory, either a local path or a remote [user@]host:/path"
    )
    parser.add_argument(
        "--rsh", default=os.environ.get("RSYNC_RSH"),
        help="Remote shell command for a remote destination, used for both the pre-flight "
             "check and rsync (default: $RSYNC_RSH, otherwise ssh with a shared connection)"
    )
    parser.add_argument(
        "--no-preflight", action="store_true",
        help="Skip the remote destination check that runs before ELF extraction"
    )
    args = parser.parse_args()

    try:
        deploy(args.src_dir, args.dst_dir, rsh=args.rsh, preflight=not args.no_preflight)
    except RuntimeError as error:
        _abort(str(error))


if __name__ == "__main__":
    main()
