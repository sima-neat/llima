#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2025 SiMa.ai, All Rights Reserved.       ||
#**************************************************************************
# NOTICE:  All information contained herein is, and remains the property of
# SiMa.ai. The intellectual and technical concepts contained herein are
# proprietary to SiMa and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
#**************************************************************************

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
