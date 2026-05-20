import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def _abort(message):
    """
    Terminate with a user-facing error message.
    """
    print(message, file=sys.stderr)
    sys.exit(-1)


def main():
    parser = argparse.ArgumentParser(description="LoRA deploy utility")
    parser.add_argument(
        "src_dir", type=Path,
        help="Path to the source directory with LoRA numpy files"
    )
    parser.add_argument(
        "dst_dir", type=Path, help="Path to the destination directory to be copied to"
    )
    args = parser.parse_args()

    # Check to ensure the destination folder contains "elf_files".
    test_dir = f"{args.dst_dir}/elf_files/"
    dry_run_cmd = ["rsync", "--dry-run", test_dir, f"{args.src_dir}/tmp"]
    try:
        subprocess.run(dry_run_cmd, check=True)
    except subprocess.CalledProcessError as _:
        _abort(
            f"The destination folder {args.dst_dir} does not contain ELF folder.\n"
            f"Expected {test_dir}."
        )

    # Copy npy files to "npy_files" sub-folder in the destination folder.
    src_files = f"{args.src_dir}/"
    target_folder = args.dst_dir / "npy_files"
    cmd = ["rsync", "-aP", "--mkpath", src_files, target_folder]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
