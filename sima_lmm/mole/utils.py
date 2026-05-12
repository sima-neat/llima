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
Utilities for MoLE.
"""

import datetime
import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Literal

import rich
import rich.box
from lm_eval.utils import HIGHER_IS_BETTER_SYMBOLS, handle_non_serializable
from loguru import logger
from PIL import Image
from pytablewriter import LatexTableWriter, MarkdownTableWriter
from rich.table import Table


class Colors(Enum):
    """
    A collection of color hexes for use throughout the project.
    Source: https://github.com/joshdick/onedark.vim/blob/main/colors/onedark.vim#L17

    " +---------------------------------------------+
    " |  Color Name  |         RGB        |   Hex   |
    " |--------------+--------------------+---------|
    " | Black        | rgb(40, 44, 52)    | #282c34 |
    " |--------------+--------------------+---------|
    " | White        | rgb(171, 178, 191) | #abb2bf |
    " |--------------+--------------------+---------|
    " | Light Red    | rgb(224, 108, 117) | #e06c75 |
    " |--------------+--------------------+---------|
    " | Dark Red     | rgb(190, 80, 70)   | #be5046 |
    " |--------------+--------------------+---------|
    " | Green        | rgb(152, 195, 121) | #98c379 |
    " |--------------+--------------------+---------|
    " | Light Yellow | rgb(229, 192, 123) | #e5c07b |
    " |--------------+--------------------+---------|
    " | Dark Yellow  | rgb(209, 154, 102) | #d19a66 |
    " |--------------+--------------------+---------|
    " | Blue         | rgb(97, 175, 239)  | #61afef |
    " |--------------+--------------------+---------|
    " | Magenta      | rgb(198, 120, 221) | #c678dd |
    " |--------------+--------------------+---------|
    " | Cyan         | rgb(86, 182, 194)  | #56b6c2 |
    " |--------------+--------------------+---------|
    " | Gutter Grey  | rgb(76, 82, 99)    | #4b5263 |
    " |--------------+--------------------+---------|
    " | Comment Grey | rgb(92, 99, 112)   | #5c6370 |
    " +---------------------------------------------+

    """

    SEAFOAM_MING = "#C7EAE4"
    DIGITAL_LAVENDER = "#DCD0FF"
    DUSTY_ROSE = "#F3D9DC"

    TURQOUISE = "#40E0D0"
    VIVID_VIOLET = "#BF55EC"
    CORAL = "#FF7F50"

    BLACK = "#282c34"
    WHITE = "#abb2bf"
    LIGHT_RED = "#e06c75"
    DARK_RED = "#be5046"
    GREEN = "#98c379"
    LIGHT_YELLOW = "#e5c07b"
    DARK_YELLOW = "#d19a66"
    BLUE = "#61afef"
    MAGENTA = "#c678dd"
    CYAN = "#56b6c2"
    GUTTER_GREY = "#4b5263"
    COMMENT_GREY = "#5c6370"


def setup_logging(verbosity: int = logging.DEBUG):
    """
    setup_logging Configure the Loguru logger.

    Args:
        verbosity: The default logging level, specified as a standard logging constant
            (e.g., `logging.INFO`, `logging.DEBUG`). This can be overridden by the LOGLEVEL
            environment variable. Defaults to logging.INFO.
    """

    logger.remove()

    raw_log_level = os.environ.get("LOGLEVEL", "INFO")
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = level_map.get(raw_log_level.upper(), verbosity)
    logger.add(
        sys.stdout,
        level=log_level,
        # format="<green>{time:YYYY-MM-DD:HH:mm:ss}</green> "
        # "<level>{level: <8}</level> "
        # "[<cyan>{name}</cyan>:<cyan>{line}</cyan>] - "
        # "<level>{message}</level>",
        format=(
            "<green>{time:YYYY-MM-DD:HH:mm:ss}</green>"
            " | "
            "<level>{level:>8}</level> "
            " | "
            f"<fg {Colors.CYAN.value}>{{file.name:>15.15}}</fg {Colors.CYAN.value}>"
            ":"
            f"<fg {Colors.MAGENTA.value}>{{function:<20.20}}</fg {Colors.MAGENTA.value}>"
            ":"
            f"<fg {Colors.LIGHT_RED.value}>{{line:>4}}</fg {Colors.LIGHT_RED.value}>"
            " | "
            "<white>{message}</white>"
        ),
        colorize=True,
    )
    logger.add(f"logs/log_{datetime.datetime.now(tz=datetime.timezone.utc).isoformat()}UTC.log")
    if log_level > logging.DEBUG:
        noisy_loggers = ["urllib3", "filelock", "fsspec", "requests"]
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def make_table(
    result_dict,
    column: str = "results",
    sort_results: bool = False,
    output_format: Literal["rich", "markdown", "latex"] = "rich",
    table_title: str = "Results",
) -> Table | str:
    """
    Generates a formatted table of evaluation results.

    The function is based on `lm_eval.utils.make_table`, and has been updated to support multiple
    output formsts including a colorized `rich` table for the console, a `Markdown` table for
    reports, and a `LaTeX` table for folks who value quality typesetting.

    Args:
        result_dict: The dictionary containing evaluation results.
        column: The key in `result_dict` to source data from, typically 'results' or 'groups'.
            Defaults to "results".
        sort_results: Whether to sort the tasks/groups alphabetically. Defaults to False.
        output_format: The desired output format. Defaults to "rich".
        table_title: The title to be displayed above the table. Defaults to "Results".

    Returns:
        A `rich.table.Table` object if `output_format` is "rich", otherwise a string containing
        the formatted table.

    Raises:
        ValueError: If an unknown `column` or `output_format` is provided.
    """

    match column:
        case "results":
            column_name = "Tasks"
        case "groups":
            column_name = "Groups"
        case _:
            raise ValueError(f"Unknown {column}")

    all_headers = [
        column_name,
        "Version",
        "Filter",
        "n-shot",
        "Metric",
        "",
        "Value",
        "",
        "Stderr",
    ]

    md_writer = MarkdownTableWriter()
    md_writer.headers = all_headers

    latex_writer = LatexTableWriter()
    latex_writer.headers = all_headers

    rich_table = Table(
        title=table_title,
        show_header=True,
        box=rich.box.SQUARE,
        header_style=f"bold {Colors.MAGENTA.value}",
    )
    for header in all_headers:
        if header.lower() == "value":
            rich_table.add_column(
                header,
                style=Colors.CYAN.value,
                no_wrap=True,
                justify="right",
            )
        elif header.lower() in ("tasks", "groups"):
            rich_table.add_column(
                header,
                style=Colors.GREEN.value,
                no_wrap=True,
            )
        else:
            rich_table.add_column(header, justify="center")

    values = []

    keys = result_dict[column].keys()
    if sort_results:
        # sort entries alphabetically by task or group name.
        # NOTE: we default here to false, because order matters for multi-level table printing a la
        # mmlu.
        # sorting here would mess that up
        keys = sorted(keys)
    for k in keys:
        dic = result_dict[column][k]
        version = result_dict["versions"].get(k, "    N/A")
        n = str(result_dict.get("n-shot", " ").get(k, " "))
        higher_is_better = result_dict.get("higher_is_better", {}).get(k, {})

        if "alias" in dic:
            k = dic.pop("alias")

        metric_items = dic.items()
        metric_items = sorted(metric_items)

        for (mf), v in metric_items:
            m, _, f = mf.partition(",")
            if m.endswith("_stderr"):
                continue

            hib = HIGHER_IS_BETTER_SYMBOLS.get(higher_is_better.get(m), "")

            v = "%.4f" % v if isinstance(v, float) else v

            if m + "_stderr" + "," + f in dic:
                se = dic[m + "_stderr" + "," + f]
                se = "   N/A" if se == "N/A" else "%.4f" % se
                values.append([k, version, f, n, m, hib, v, "±", se])
            else:
                values.append([k, version, f, n, m, hib, v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values
    for row in values:
        rich_table.add_row(
            *(str(handle_non_serializable(r)) for r in row),
            style=Colors.WHITE.value,
        )
    match output_format:
        case "rich":
            return rich_table
        case "latex":
            return latex_writer.dumps()
        case "markdown":
            return md_writer.dumps()
        case _:
            raise ValueError(f"Invalid output_format: {output_format}")
