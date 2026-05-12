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
    parser = argparse.ArgumentParser(description="LMM deploy utility")
    parser.add_argument(
        "src_dir", type=Path,
        help="Path to the source directory with compiled mpk tar.gz files and devkit files"
    )
    parser.add_argument(
        "dst_dir", type=Path, help="Path to the destination directory to be copied to"
    )
    args = parser.parse_args()

    # Check if the src_dir contains the required directories.
    src_devkit_dir = next(args.src_dir.rglob("devkit"), None)
    if src_devkit_dir is None or not src_devkit_dir.is_dir():
        _abort(f"devkit directory cannot be found in {args.src_dir}")

    src_mpk_dir = src_devkit_dir.parent / "mpk"
    if not src_mpk_dir.is_dir():
        _abort(f"mpk directory cannot be found in {src_mpk_dir.parent}")

    # Extract the elf_files.
    src_elf_dir = src_devkit_dir.parent / "elf_files"
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
    cmd = ["rsync", "-aP", "--mkpath", *src_dirs, args.dst_dir]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
