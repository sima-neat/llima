#!/usr/bin/env python3
"""Recompute LLiMa DVT response TPS from raw per-interval evidence.

The benchmark's ``LLIMA_DIRECT_VLM_INTERVAL`` records, not its rounded PASS
line, are authoritative.  This script fails closed on missing, duplicated, or
misclassified intervals and prints one harmonic response rate per trial plus
the untrimmed mean.  It intentionally performs no trimming and no trace-tax
correction.
"""

from __future__ import annotations

import argparse
import collections
import math
import re
from pathlib import Path


INTERVAL = re.compile(
    r"^LLIMA_DIRECT_VLM_INTERVAL "
    r"trial=(?P<trial>\d+) "
    r"interval=(?P<interval>\d+) "
    r"phase=(?P<phase>warmup|measured) "
    r"output_token_id=(?P<token>\d+) "
    r"period_us=(?P<period>[0-9]+(?:\.[0-9]+)?)$"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--measured", type=int, required=True)
    args = parser.parse_args()

    records: dict[int, dict[int, tuple[str, int, float]]] = collections.defaultdict(dict)
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = INTERVAL.match(line.strip())
        if not match:
            continue
        trial = int(match["trial"])
        interval = int(match["interval"])
        if interval in records[trial]:
            raise SystemExit(f"duplicate trial={trial} interval={interval}")
        period = float(match["period"])
        if not math.isfinite(period) or period <= 0:
            raise SystemExit(f"invalid period trial={trial} interval={interval}")
        records[trial][interval] = (
            match["phase"], int(match["token"]), period
        )

    expected_indices = list(range(args.warmup + args.measured))
    rates: list[float] = []
    for trial in range(1, args.trials + 1):
        found = records.get(trial, {})
        if sorted(found) != expected_indices:
            missing = sorted(set(expected_indices) - set(found))
            extra = sorted(set(found) - set(expected_indices))
            raise SystemExit(
                f"trial {trial}: incomplete evidence missing={missing} extra={extra}"
            )
        for interval in expected_indices:
            expected_phase = "warmup" if interval < args.warmup else "measured"
            if found[interval][0] != expected_phase:
                raise SystemExit(
                    f"trial {trial} interval {interval}: expected {expected_phase}"
                )
        periods = [
            found[index][2]
            for index in range(args.warmup, args.warmup + args.measured)
        ]
        rate = 1.0e6 * len(periods) / sum(periods)
        rates.append(rate)
        print(
            f"LLIMA_DIRECT_VLM_RECOMPUTED_TRIAL trial={trial} "
            f"measured_intervals={len(periods)} response_tps={rate:.9f}"
        )

    mean = sum(rates) / len(rates)
    print(
        f"LLIMA_DIRECT_VLM_RECOMPUTED_PASS trials={len(rates)} "
        f"warmup_intervals={args.warmup} measured_intervals={args.measured} "
        f"response_tps_mean={mean:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
