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

from typing import Any, Literal

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import rich
import rich.box
from lm_eval.utils import HIGHER_IS_BETTER_SYMBOLS, handle_non_serializable
from loguru import logger
from pytablewriter import LatexTableWriter, MarkdownTableWriter
from rich.table import Table

from sima_lmm.mole.utils import Colors

DELTA_CHAR = "Δ"


def hex_by_pct(percentage: float) -> str:
    percentage = max(-100, min(100, percentage))
    if percentage < 0:
        r, g, b = 255, round(255 * (percentage + 100) / 100), 0
    else:
        r, g, b = round(255 * (100 - percentage) / 100), 255, 0
    return f"#{r:02x}{g:02x}{b:02x}".upper()


def hex_eq_split(num_colors: int, cmap_name: str = "cool") -> list[str]:
    """
    Cool/BuPu preferred
    """
    cmap = plt.get_cmap(cmap_name)
    points = np.random.permutation(np.linspace(0, 1, num_colors))
    hex_colors = [matplotlib.colors.to_hex(cmap(p)) for p in points]

    return hex_colors


def compare_two(
    ref_results: dict[str, Any],
    gen_results: dict[str, Any],
    ref_col_header: str = "Reference Value",
    gen_col_header: str = "Generated Value",
    output_format: Literal["rich", "markdown", "latex"] = "rich",
    table_title: str = "Results",
) -> Table | str:
    """
    Creates a comparison table from two sets of evaluation results.

    This function compares metrics from a reference and a generated results dictionary,
    calculating the absolute and percentage difference. It can output the table in
    multiple formats.

    Args:
        ref_results: A dictionary containing the reference (e.g., baseline) evaluation results.
        gen_results: A dictionary containing the generated evaluation results to compare.
        ref_col_header: The header for the reference value column. Defaults to "Reference Value".
        gen_col_header: The header for the generated value column. Defaults to "Generated Value".
        output_format: The desired output format. Can be "rich", "markdown", or "latex". Defaults to
            "rich".
        table_title: The table's title. Defaults to "Results"

    Raises:
        ValueError: Raised if an unsupported `output_format` is provided.
        AssertionError: Raised if there are no common metrics between the reference and generated
            results.

    Returns:
        A `rich.table.Table` object if output_format is "rich", otherwise a string
        representation of the table in the specified format (Markdown or LaTeX).
    """
    column_name = "Task"
    all_headers = [
        column_name, "Metric", "", ref_col_header, gen_col_header, "Difference", "% Difference"
    ]

    column = "results"
    comp_list = []
    ref_results_keys = ref_results[column].keys()
    gen_results_keys = gen_results[column].keys()

    common_keys = list(set(ref_results_keys).intersection(gen_results_keys))
    assert len(common_keys) > 0, (
        f"Reference metrics: {ref_results_keys} have no overlap with generated metrics: "
        f"{gen_results_keys}"
    )

    for k in common_keys:
        ref_dic = ref_results[column][k]
        gen_dic = gen_results[column][k]

        ref_version = ref_results["versions"].get(k, "    N/A")
        gen_version = gen_results["versions"].get(k, "    N/A")

        if ref_version != gen_version:
            logger.warning(
                f"Metric {k} version mismatch! Reference {ref_version} Generated {gen_version}"
            )
            continue

        ref_nshot = str(ref_results.get("n-shot", " ").get(k, " "))
        gen_nshot = str(gen_results.get("n-shot", " ").get(k, " "))

        if ref_nshot != gen_nshot:
            logger.warning(
                f"Metric {k} n-shot mismatch! Reference: {ref_nshot} Generated {gen_nshot}"
            )
            continue

        higher_is_better = gen_results.get("higher_is_better", {}).get(k, {})

        gen_dic.pop("alias", "")
        ref_dic.pop("alias", "")

        combined_items = zip(sorted(ref_dic.items()), sorted(gen_dic.items()), strict=True)

        for (ref_mf, ref_v), (gen_mf, gen_v) in combined_items:
            (ref_m, _, ref_f) = ref_mf.partition(",")
            (gen_m, _, gen_f) = gen_mf.partition(",")

            delta = gen_v - ref_v
            delta_pct = (delta / gen_v) * 100
            if ref_m.endswith("_stderr"):
                continue
            ref_v = f"{ref_v:.4f}" if isinstance(ref_v, float) else ref_v
            gen_v = f"{gen_v:.4f}" if isinstance(gen_v, float) else gen_v

            if output_format == "rich":
                col_hex = hex_by_pct(delta_pct)
                delta = f"[{col_hex}]{delta:.4f}[/]"
                delta_pct = f"[{col_hex}]{delta_pct:.2f}[/]"
            else:
                delta = f"{delta:.4f}"
                delta_pct = f"{delta_pct:.2f}"

            hib = HIGHER_IS_BETTER_SYMBOLS.get(higher_is_better.get(ref_m), "")
            comp_list.append([k, gen_m, hib, ref_v, gen_v, delta, delta_pct])
            k = ""

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

    md_writer.value_matrix = comp_list
    latex_writer.value_matrix = comp_list
    for row in comp_list:
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


def compare_many(
    ref_results: tuple[str, dict[str, Any]],
    gen_results: list[tuple[str, dict[str, Any]]],
    output_format: Literal["rich", "markdown", "latex"] = "rich",
    table_title: str = "Comparison of FP32 vs Board",
) -> Table | str:
    assert len(gen_results) > 0
    ref_col_title, ref_result = ref_results
    column_name = "Task"

    cols = hex_eq_split(len(gen_results))

    if output_format == "rich":
        all_headers = [column_name, "Metric", "", f"{ref_col_title}"] + [
            f"[{cols[i]}]{item}[/]"
            for i, (gen_col_title, _) in enumerate(gen_results)
            for _, item in enumerate([gen_col_title, DELTA_CHAR, f"% {DELTA_CHAR}"])
        ]
    else:
        all_headers = [column_name, "Metric", "", ref_col_title] + [
            item
            for gen_col_title, _ in gen_results
            for item in [
                gen_col_title, f"{DELTA_CHAR} {gen_col_title}", f"% {DELTA_CHAR} {gen_col_title}"
            ]
        ]
    column = "results"
    comp_list = []
    ref_result_keys = ref_result[column].keys()
    gen_results_keys = [res[column].keys() for _, res in gen_results]

    common_keys = [set(ref_result_keys).intersection(g_keys) for g_keys in gen_results_keys]
    assert all(len(x) > 0 for x in common_keys), (
        f"Reference metrics: {ref_result_keys} have no overlap with generated metrics: "
        f"{gen_results_keys}"
    )

    for k in {e for s in common_keys for e in s}:
        ref_dic = ref_result[column][k]
        gen_dics = [gen_result[column][k] for _, gen_result in gen_results]

        ref_version = ref_result["versions"].get(k, "    N/A")
        gen_versions = [gen_result["versions"].get(k, "    N/A") for _, gen_result in gen_results]

        if any(ref_version != gen_version for gen_version in gen_versions):
            logger.warning(
                f"Metric {k} version mismatch! Reference {ref_version} Generated {gen_versions}"
            )
            continue

        ref_nshot = str(ref_result.get("n-shot", " ").get(k, " "))
        gen_nshots = [
            str(gen_result.get("n-shot", " ").get(k, " ")) for _, gen_result in gen_results
        ]

        if any(ref_nshot != gen_nshot for gen_nshot in gen_nshots):
            logger.warning(
                f"Metric {k} n-shot mismatch! Reference: {ref_nshot} Generated {gen_nshots}"
            )
            continue

        higher_is_better = ref_result.get("higher_is_better", {}).get(k, {})

        [gen_dic.pop("alias", "") for gen_dic in gen_dics]
        ref_dic.pop("alias", "")

        combined_items = zip(
            sorted(ref_dic.items()), *[sorted(gen_dic.items()) for gen_dic in gen_dics], strict=True
        )

        for (ref_mf, ref_v), *gen_pairs in combined_items:
            (ref_m, _, ref_f) = ref_mf.partition(",")
            if ref_m.endswith("_stderr"):
                continue

            hib = HIGHER_IS_BETTER_SYMBOLS.get(higher_is_better.get(ref_m), " ")
            row = []

            for gen_mf, gen_v in gen_pairs:
                (gen_m, _, gen_f) = gen_mf.partition(",")

                delta = gen_v - ref_v
                delta_pct = (delta / gen_v) * 100
                gen_v = f"{gen_v:.4f}" if isinstance(gen_v, float) else gen_v

                if output_format == "rich":
                    col_hex = hex_by_pct(delta_pct)
                    delta = f"[{col_hex}]{delta:.4f}[/]"
                    delta_pct = f"[{col_hex}]{delta_pct:.2f}[/]"
                else:
                    delta = f"{delta:.4f}"
                    delta_pct = f"{delta_pct:.2f}"

                row.extend([gen_v, delta, delta_pct])

            ref_v = f"{ref_v:.4f}" if isinstance(ref_v, float) else ref_v
            row = [k, ref_m, hib, ref_v] + row
            comp_list.append(row)
            k = ""

    md_writer = MarkdownTableWriter()
    md_writer.headers = all_headers

    latex_writer = LatexTableWriter()
    latex_writer.headers = all_headers

    rich_table = Table(
        title=table_title,
        show_header=True,
        box=rich.box.SQUARE,
        header_style="bold",
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

    md_writer.value_matrix = comp_list
    latex_writer.value_matrix = comp_list
    for row in comp_list:
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
