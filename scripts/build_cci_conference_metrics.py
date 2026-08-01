#!/usr/bin/env python3
"""Validate end-to-end CCI evidence and emit stable paper metric macros."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


METHODS = ("A0", "A11")
PREFIXES = {"A0": "EndToEndBLD", "A11": "EndToEndAdaptive"}
REQUIRED_AGGREGATES = (
    "fid",
    "sfid",
    "fva_rate",
    "fs",
    "mnac",
    "cd",
    "cout",
    "directional_fr",
)
RECONCILED_PAIR_FIELDS = {
    "directional_fr": "directional_flip",
    "fva_rate": "fva_cosine",
    "fs": "fs_cosine",
    "mnac": "mnac",
    "cout": "cout",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing metric table: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty metric table: {path}")
    return rows


def _truth(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _finite_float(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_aggregates(
    rows: list[dict[str, str]],
    expected_count: int,
) -> dict[str, dict[str, float]]:
    selected = [
        row
        for row in rows
        if row.get("task") == "smile" and row.get("method") in METHODS
    ]
    if len(selected) != len(METHODS) or {
        row["method"] for row in selected
    } != set(METHODS):
        raise ValueError("aggregate table must contain exactly A0 and A11")

    methods: dict[str, dict[str, float]] = {}
    for row in selected:
        method = row["method"]
        count = int(float(row["n"]))
        if count != expected_count:
            raise ValueError(
                f"{method} expected {expected_count} aggregate rows; found {count}"
            )
        values = {
            field: _finite_float(row[field], f"{method}/{field}")
            for field in REQUIRED_AGGREGATES
        }
        methods[method] = values
    return methods


def _pair_metric(rows: list[dict[str, str]], metric: str) -> float:
    source_field = RECONCILED_PAIR_FIELDS[metric]
    if metric == "directional_fr":
        return statistics.fmean(_truth(row[source_field]) for row in rows)
    if metric == "fva_rate":
        return statistics.fmean(
            _finite_float(row[source_field], source_field) > 0.5
            for row in rows
        )
    return statistics.fmean(
        _finite_float(row[source_field], source_field) for row in rows
    )


def _validate_and_reconcile_pairs(
    rows: list[dict[str, str]],
    methods: dict[str, dict[str, float]],
    expected_count: int,
) -> int:
    selected = [
        row
        for row in rows
        if row.get("feature") == "smile" and row.get("variant") in METHODS
    ]
    seen: set[tuple[str, int, str]] = set()
    by_method: dict[str, list[dict[str, str]]] = {
        method: [] for method in METHODS
    }
    for row in selected:
        method = row["variant"]
        sample_id = int(float(row["sample_id"]))
        key = ("smile", sample_id, method)
        if key in seen:
            raise ValueError(f"duplicate pair key: {key}")
        seen.add(key)
        by_method[method].append(row)

    for method, method_rows in by_method.items():
        if len(method_rows) != expected_count:
            raise ValueError(
                f"{method} expected {expected_count} pair rows; "
                f"found {len(method_rows)}"
            )
    ids_by_method = {
        method: {int(float(row["sample_id"])) for row in method_rows}
        for method, method_rows in by_method.items()
    }
    if ids_by_method["A0"] != ids_by_method["A11"]:
        raise ValueError("paired ID sets differ between A0 and A11")

    for method, method_rows in by_method.items():
        for metric in RECONCILED_PAIR_FIELDS:
            recomputed = _pair_metric(method_rows, metric)
            aggregate = methods[method][metric]
            if not math.isclose(
                recomputed, aggregate, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(
                    f"pair-level {metric} disagrees for {method}: "
                    f"{recomputed} != {aggregate}"
                )
    return len(ids_by_method["A0"])


def _compute_deltas(
    methods: dict[str, dict[str, float]],
) -> dict[str, float]:
    baseline = methods["A0"]
    adaptive = methods["A11"]
    baseline_cd = baseline["cd"]
    if baseline_cd == 0:
        raise ValueError("A0/cd must be non-zero for relative reduction")
    return {
        "fr_percentage_points": 100.0
        * (adaptive["directional_fr"] - baseline["directional_fr"]),
        "cout_gain": adaptive["cout"] - baseline["cout"],
        "cd_reduction_percent": 100.0
        * (baseline_cd - adaptive["cd"])
        / baseline_cd,
    }


def build_metrics(
    full_metrics_path: Path,
    pair_metrics_path: Path,
    expected_count: int,
) -> dict[str, object]:
    """Validate the paired experiment and return publication-ready metrics."""

    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    aggregate_rows = _read_csv(Path(full_metrics_path))
    pair_rows = _read_csv(Path(pair_metrics_path))
    methods = _validate_aggregates(aggregate_rows, expected_count)
    paired_count = _validate_and_reconcile_pairs(
        pair_rows, methods, expected_count
    )
    return {
        "cohort": {"paired_count": paired_count},
        "methods": methods,
        "deltas": _compute_deltas(methods),
    }


def _tex_command(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _tex_command_lines(payload: dict[str, object]) -> list[str]:
    lines = [
        "% Generated by scripts/build_cci_conference_metrics.py; do not edit."
    ]
    methods = payload["methods"]
    for method in METHODS:
        values = methods[method]
        prefix = PREFIXES[method]
        formatted = {
            "FID": f'{values["fid"]:.4f}',
            "SymmetricFID": f'{values["sfid"]:.4f}',
            "FVAPct": f'{100.0 * values["fva_rate"]:.1f}',
            "FS": f'{values["fs"]:.4f}',
            "MNAC": f'{values["mnac"]:.4f}',
            "CD": f'{values["cd"]:.4f}',
            "COUT": f'{values["cout"]:.4f}',
            "FRPct": f'{100.0 * values["directional_fr"]:.1f}',
        }
        lines.extend(
            _tex_command(prefix + suffix, value)
            for suffix, value in formatted.items()
        )
    deltas = payload["deltas"]
    lines.extend(
        (
            _tex_command(
                "EndToEndFRGainPctPoints",
                f'{deltas["fr_percentage_points"]:.1f}',
            ),
            _tex_command(
                "EndToEndCOUTGain", f'{deltas["cout_gain"]:.4f}'
            ),
            _tex_command(
                "EndToEndCDReductionPct",
                f'{deltas["cd_reduction_percent"]:.1f}',
            ),
        )
    )
    return lines


def write_outputs(
    payload: dict[str, object],
    json_path: Path,
    tex_path: Path,
) -> None:
    """Write stable JSON evidence and LaTeX commands."""

    json_path = Path(json_path)
    tex_path = Path(tex_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tex_path.write_text(
        "\n".join(_tex_command_lines(payload)) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_metrics", type=Path, required=True)
    parser.add_argument("--pair_metrics", type=Path, required=True)
    parser.add_argument("--expected_count", type=int, required=True)
    parser.add_argument("--json_out", type=Path, required=True)
    parser.add_argument("--tex_out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_metrics(
        args.full_metrics, args.pair_metrics, args.expected_count
    )
    write_outputs(payload, args.json_out, args.tex_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
