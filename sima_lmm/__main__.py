# **************************************************************************
# ||                        SiMa.ai CONFIDENTIAL                          ||
# ||   Unpublished Copyright (c) 2022-2026 SiMa.ai, All Rights Reserved.  ||
# **************************************************************************
#  NOTICE:  All information contained herein is, and remains the property of
#  SiMa.ai. The intellectual and technical concepts contained herein are
#  proprietary to SiMa and may be covered by U.S. and Foreign Patents,
#  patents in process, and are protected by trade secret or copyright law.
#
#  Dissemination of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained from
#  SiMa.ai.  Access to the source code contained herein is hereby forbidden
#  to anyone except current SiMa.ai employees, managers or contractors who
#  have executed Confidentiality and Non-disclosure agreements explicitly
#  covering such access.
#
#  The copyright notice above does not evidence any actual or intended
#  publication or disclosure  of  this source code, which includes information
#  that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
#  ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
#  DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
#  CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
#  LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
#  CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
#  REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
#  SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# **************************************************************************

import argparse
import sys

from sima_lmm import library_dir

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--library_dir",
        action="store_true",
        required=True,
        help="Print the path to the library directory."
    )
    args = parser.parse_args()
    if args.library_dir:
        print(library_dir())


if __name__ == "__main__":
    main()
