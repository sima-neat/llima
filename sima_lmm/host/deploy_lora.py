import argparse
import os
import subprocess
import sys
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


def _check_local_elf_dir(src_dir: Path, dst_dir: str) -> None:
    # Check to ensure the destination folder contains "elf_files".
    test_dir = f"{dst_dir}/elf_files/"
    dry_run_cmd = ["rsync", "--dry-run", test_dir, f"{src_dir}/tmp"]
    try:
        subprocess.run(dry_run_cmd, check=True)
    except subprocess.CalledProcessError as _:
        raise RuntimeError(
            f"The destination folder {dst_dir} does not contain ELF folder.\n"
            f"Expected {test_dir}."
        )


def _copy_npy_files(src_dir: Path, target_folder: str, rsh: str | None) -> None:
    # Copy npy files to "npy_files" sub-folder in the destination folder.
    src_files = f"{src_dir}/"
    rsh_args = ["-e", rsh] if rsh else []
    cmd = ["rsync", "-aP", "--mkpath", *rsh_args, src_files, target_folder]
    subprocess.check_call(cmd)


def deploy(src_dir: Path, dst_dir: str, rsh: str | None = None, preflight: bool = True) -> None:
    dst_dir = str(dst_dir).rstrip("/") or "/"
    remote = parse_remote_destination(dst_dir)
    if remote is None or not preflight:
        _check_local_elf_dir(src_dir, dst_dir)
        _copy_npy_files(src_dir, f"{dst_dir}/npy_files", rsh)
        return

    # Check the remote model directory and reuse the SSH connection for the copy.
    with RemoteSession(remote.host, rsh) as session:
        print(f"Checking {remote} ...", flush=True)
        free_kb = session.preflight(
            remote.child_path("npy_files"), required_dir=remote.child_path("elf_files")
        )
        report_free_space(str(remote), free_kb, local_size_kb([src_dir]))
        _copy_npy_files(src_dir, remote.join("npy_files"), session.rsync_rsh)


def main():
    parser = argparse.ArgumentParser(description="LoRA deploy utility")
    parser.add_argument(
        "src_dir", type=Path,
        help="Path to the source directory with LoRA numpy files"
    )
    parser.add_argument(
        "dst_dir",
        help="Deployed model directory, either a local path or a remote [user@]host:/path"
    )
    parser.add_argument(
        "--rsh", default=os.environ.get("RSYNC_RSH"),
        help="Remote shell command for a remote destination, used for both the pre-flight "
             "check and rsync (default: $RSYNC_RSH, otherwise ssh with a shared connection)"
    )
    parser.add_argument(
        "--no-preflight", action="store_true",
        help="Skip the remote destination check that runs before copying"
    )
    args = parser.parse_args()

    try:
        deploy(args.src_dir, args.dst_dir, rsh=args.rsh, preflight=not args.no_preflight)
    except RuntimeError as error:
        _abort(str(error))


if __name__ == "__main__":
    main()
