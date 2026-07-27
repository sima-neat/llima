#!/usr/bin/env python3
"""Validate a required compiler-test group's JUnit execution count."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = ET.parse(args.report).getroot()
    test_count = sum(1 for _ in report.iter("testcase"))
    skipped_count = sum(1 for _ in report.iter("skipped"))

    if test_count != args.expected:
        raise SystemExit(
            f"{args.label} collection mismatch: expected {args.expected}, "
            f"executed {test_count}."
        )
    if skipped_count:
        raise SystemExit(
            f"{args.label} unexpectedly skipped {skipped_count} tests."
        )

    print(f"{args.label} audit: {test_count} tests, 0 skipped")


if __name__ == "__main__":
    main()
