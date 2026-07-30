#!/usr/bin/env python3
"""Combine validated attacked-region metric tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence


OUTPUT_FIELDS = (
    "region",
    "method",
    "n",
    "fid",
    "sfid",
    "fva_rate",
    "fs",
    "mnac",
    "cd",
    "cout",
    "directional_fr",
)
METHODS = ("A0", "A11")
METRICS = OUTPUT_FIELDS[3:]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metric table: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def combine_region_metrics(
    regions: Sequence[tuple[str, str | Path]],
    *,
    output_dir: str | Path,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Validate two-method region tables and write one comparison report."""

    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if not regions or len({name for name, _ in regions}) != len(regions):
        raise ValueError("region names must be non-empty and unique")
    combined = []
    for region, raw_path in regions:
        rows = _read_csv(Path(raw_path))
        selected = [
            row
            for row in rows
            if row.get("task") == "smile" and row.get("method") in METHODS
        ]
        by_method = {row["method"]: row for row in selected}
        if set(by_method) != set(METHODS) or len(selected) != len(METHODS):
            raise ValueError(
                f"{region} must contain exactly A0 and A11 smile rows"
            )
        for method in METHODS:
            row = by_method[method]
            count = int(float(row["n"]))
            if count != expected_count:
                raise ValueError(
                    f"{region}/{method} must contain {expected_count} rows; "
                    f"found {count}"
                )
            normalized: dict[str, Any] = {
                "region": region,
                "method": method,
                "n": count,
            }
            for field in METRICS:
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError(
                        f"{region}/{method}/{field} must be finite"
                    )
                normalized[field] = value
            combined.append(normalized)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(combined, output_dir / "combined_metrics.csv")
    (output_dir / "combined_metrics.json").write_text(
        json.dumps(combined, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Attacked A0 vs A11 Region Comparison",
        "",
        "| Region | Method | N | FID down | sFID down | FVA up | FS up | MNAC down | CD down | COUT up | FR (%) up |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined:
        lines.append(
            f"| {row['region']} | {row['method']} | {row['n']} | "
            f"{row['fid']:.4f} | {row['sfid']:.4f} | "
            f"{100 * row['fva_rate']:.1f} | {row['fs']:.4f} | "
            f"{row['mnac']:.4f} | {row['cd']:.4f} | "
            f"{row['cout']:.4f} | {100 * row['directional_fr']:.1f} |"
        )
    (output_dir / "combined_metrics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return combined


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region",
        action="append",
        nargs=2,
        metavar=("NAME", "FULL_METRICS_CSV"),
        required=True,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_count", type=int, default=300)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    combine_region_metrics(
        args.region,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
