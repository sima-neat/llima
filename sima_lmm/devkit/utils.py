#########################################################
# Copyright (C) 2024 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################

import logging
import os

from sima_lmm.devkit.cpp_ext import CLI, WEB, ZMQServer, connect_cpp, disconnect_cpp
from sima_utils.logging.sima_logger import sima_log_info


def connect(log_level: int | str):
    if isinstance(log_level, str):
        log_level = logging.getLevelNamesMapping()[log_level]
    assert isinstance(log_level, int)

    connect_cpp([], "run.cpp.log", log_level)
    sima_log_info("MLA dispatcher connected")


def disconnect():
    disconnect_cpp()
    sima_log_info("MLA dispatcher disconnected")
