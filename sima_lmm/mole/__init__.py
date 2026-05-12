#########################################################
# Copyright (C) 2026 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################

"""
MoLE

Exposes mole entry points, also handles initial package-level setup,
including logging configuration and warning suppression.
"""

import logging
import os
import warnings

from tqdm import TqdmExperimentalWarning

from sima_lmm.mole import performance_bench
from sima_lmm.mole.extras import compare_many, compare_two
from sima_lmm.mole.modalixboard import ModalixBoard
from sima_lmm.mole.mole import prepare_mole, run
from sima_lmm.mole.utils import setup_logging

__all__ = [
    "ModalixBoard",
    "compare_many",
    "compare_two",
    "prepare_mole",
    "run",
    "performance_bench",
]

setup_logging(logging._nameToLevel[os.getenv("MOLE_LOG_LEVEL", "DEBUG")])
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
