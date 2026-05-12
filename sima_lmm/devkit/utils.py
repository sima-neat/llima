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

import importlib.resources
import logging
import os

from sima_lmm.devkit.cpp_ext import CLI, WEB, ZMQServer, connect_cpp, disconnect_cpp
from sima_utils.logging.sima_logger import sima_log_info


def connect(log_level: int | str):
    # Locate sample image file and audio file to warm up the libraries during startup.
    assets_dir = importlib.resources.files("sima_lmm.assets")
    sample_image_file_name = assets_dir / "sjc.jpg"
    sample_audio_file_name = assets_dir / "why_is_the_sky_blue.wav"
    assert sample_image_file_name.is_file()
    assert sample_audio_file_name.is_file()

    if isinstance(log_level, str):
        log_level = logging.getLevelNamesMapping()[log_level]
    assert isinstance(log_level, int)

    connect_cpp(
        [],
        "run.cpp.log",
        log_level,
        sample_image_file_name,
        sample_audio_file_name
    )
    sima_log_info("MLA dispatcher connected")


def disconnect():
    disconnect_cpp()
    sima_log_info("MLA dispatcher disconnected")
