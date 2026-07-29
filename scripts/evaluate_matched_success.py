#!/usr/bin/env python3
"""Evaluate fixed versus adaptive CCI on a calibration-frozen success grid."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cci_diff.matched_success import (
    acceptance_flags,
    calibration_frontier,
    freeze_common_operating_points,
    matched_estimates,
    paired_cluster_bootstrap,
)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    calibration_rows = _read_rows(args.calibration_csv)
    test_rows = _read_rows(args.test_csv)
    variants = (args.fixed_variant, args.adaptive_variant)
    frontiers = {
        variant: calibration_frontier(
            [
                row
                for row in calibration_rows
                if row["variant"] == variant
            ],
            identity_floor=args.identity_floor,
            locality_ceiling=args.locality_ceiling,
        )
        for variant in variants
    }
    frozen = freeze_common_operating_points(
        frontiers,
        step=args.success_step,
    )
    estimates = matched_estimates(
        test_rows,
        frozen,
        fixed_variant=args.fixed_variant,
        adaptive_variant=args.adaptive_variant,
    )
    intervals = paired_cluster_bootstrap(
        test_rows,
        frozen,
        fixed_variant=args.fixed_variant,
        adaptive_variant=args.adaptive_variant,
        seed=args.bootstrap_seed,
        samples=args.bootstrap_samples,
    )
    flags = acceptance_flags(
        estimates,
        intervals,
        fixed_variant=args.fixed_variant,
        adaptive_variant=args.adaptive_variant,
    )
    payload = {
        "fixed_variant": args.fixed_variant,
        "adaptive_variant": args.adaptive_variant,
        "calibration_frontiers": {
            variant: [asdict(point) for point in points]
            for variant, points in frontiers.items()
        },
        "common_success_grid": list(frozen.grid),
        "frozen_weights": frozen.weights,
        "test_estimates": estimates,
        "paired_cluster_bootstrap": intervals,
        "acceptance": flags,
        "supported": flags["supported"],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "identity_cluster",
        "source_id",
        "seed",
        "variant",
        "effort",
        "target_success",
        "independent_non_target_drift",
        "identity_cosine",
        "outside_semantic_l1",
    }
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(
                "matched-success CSV is missing: " + ", ".join(missing)
            )
        row["seed"] = int(row["seed"])
        row["target_success"] = _parse_probability(row["target_success"])
        for field in (
            "independent_non_target_drift",
            "identity_cosine",
            "outside_semantic_l1",
        ):
            row[field] = float(row[field])
    return rows


def _parse_probability(value: Any) -> float:
    normalized = str(value).strip().casefold()
    if normalized in {"true", "yes"}:
        return 1.0
    if normalized in {"false", "no"}:
        return 0.0
    number = float(value)
    if not 0 <= number <= 1:
        raise ValueError("target_success must be a probability or boolean")
    return number


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--fixed_variant", default="A10")
    parser.add_argument("--adaptive_variant", default="A11")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--identity_floor", type=float, default=0.90)
    parser.add_argument("--locality_ceiling", type=float, default=0.02)
    parser.add_argument("--success_step", type=float, default=0.05)
    return parser


def main() -> int:
    evaluate(build_arg_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
