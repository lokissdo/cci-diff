#!/usr/bin/env python3
"""Summarize matched CCI component ablations against full A3."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


VARIANT_LABELS = {
    "A0": "raw BLD",
    "A1": "legacy classifier hook",
    "A2": "fixed constraint weights",
    "A3": "full adaptive CCI",
    "A4": "without projection",
    "A5": "without target guidance",
    "A6": "without gradient normalization",
    "A7": "without target-first budget",
    "A8": "without denoising schedule",
    "A9": "without final correction",
}

METRICS = (
    "target_success",
    "directional_flip",
    "desired_probability",
    "fva_cosine",
    "fs_cosine",
    "mnac",
    "changed_fraction_5",
    "outside_semantic_fraction_5",
    "outside_generation_fraction_5",
    "residual_tv",
    "runtime_seconds",
)


def _number(value: Any) -> float:
    text = str(value).strip().lower()
    if text == "true":
        return 1.0
    if text == "false":
        return 0.0
    return float(value)


def paired_metric_delta(
    baseline_rows: Sequence[Mapping[str, Any]],
    ablation_rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    seed: int = 42,
    bootstrap_iterations: int = 10_000,
) -> dict[str, Any]:
    """Return paired baseline-minus-ablation statistics for one metric."""

    baseline = {str(row["sample_id"]): row for row in baseline_rows}
    ablation = {str(row["sample_id"]): row for row in ablation_rows}
    if set(baseline) != set(ablation):
        raise ValueError("baseline and ablation sample IDs must match")
    pairs = []
    for sample_id in sorted(baseline, key=lambda value: int(float(value))):
        left = baseline[sample_id].get(field)
        right = ablation[sample_id].get(field)
        if left in (None, "") or right in (None, ""):
            continue
        pairs.append((_number(left), _number(right)))
    if not pairs:
        return {
            "count": 0,
            "baseline_mean": None,
            "ablation_mean": None,
            "delta": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    values = np.asarray(pairs, dtype=float)
    differences = values[:, 0] - values[:, 1]
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(
        differences,
        size=(bootstrap_iterations, len(differences)),
        replace=True,
    ).mean(axis=1)
    return {
        "count": len(pairs),
        "baseline_mean": float(values[:, 0].mean()),
        "ablation_mean": float(values[:, 1].mean()),
        "delta": float(differences.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [
        _number(row[field])
        for row in rows
        if row.get(field) not in (None, "")
    ]
    return float(np.mean(values)) if values else None


def _format(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def summarize(root: Path, *, baseline: str = "A3", seed: int = 42) -> dict[str, Any]:
    root = Path(root)
    rows = _read_rows(root / "ace_pair_metrics.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["variant"], row["feature"]), []).append(row)
    variants = sorted({variant for variant, _ in grouped})
    tasks = sorted({feature for _, feature in grouped})
    if baseline not in variants:
        raise ValueError(f"baseline variant not found: {baseline}")

    aggregate_rows = []
    for variant in variants:
        for task in tasks:
            task_rows = grouped.get((variant, task), [])
            aggregate_rows.append(
                {
                    "variant": variant,
                    "description": VARIANT_LABELS.get(variant, variant),
                    "task": task,
                    "count": len(task_rows),
                    **{field: _mean(task_rows, field) for field in METRICS},
                }
            )

    delta_rows = []
    for variant in variants:
        if variant == baseline:
            continue
        for task in tasks:
            baseline_rows = grouped[(baseline, task)]
            ablation_rows = grouped[(variant, task)]
            for offset, field in enumerate(METRICS):
                delta_rows.append(
                    {
                        "baseline": baseline,
                        "variant": variant,
                        "description": VARIANT_LABELS.get(variant, variant),
                        "task": task,
                        "metric": field,
                        **paired_metric_delta(
                            baseline_rows,
                            ablation_rows,
                            field,
                            seed=seed + offset,
                        ),
                    }
                )

    _write_csv(aggregate_rows, root / "component_ablation_metrics.csv")
    _write_csv(delta_rows, root / "component_ablation_deltas_vs_a3.csv")

    lines = [
        "# CCI Component Ablation",
        "",
        f"Matched pilot with `{len(grouped[(baseline, tasks[0])])}` samples per task. "
        "Deltas are `A3 - variant`; positive target deltas favor full A3, while "
        "negative MNAC/change/runtime deltas favor full A3.",
        "",
        "## Absolute Metrics",
        "",
        "| Variant | Task | ACE FR | Desired p | FVA cosine | FS | MNAC | Changed 5% | Outside semantic 5% | TV | Runtime s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['variant']} | {row['task']} | {_format(row['directional_flip'])} | "
            f"{_format(row['desired_probability'])} | {_format(row['fva_cosine'])} | "
            f"{_format(row['fs_cosine'])} | {_format(row['mnac'])} | "
            f"{_format(row['changed_fraction_5'])} | "
            f"{_format(row['outside_semantic_fraction_5'])} | "
            f"{_format(row['residual_tv'])} | {_format(row['runtime_seconds'], 2)} |"
        )

    delta_lookup = {
        (row["variant"], row["task"], row["metric"]): row for row in delta_rows
    }
    lines.extend(
        [
            "",
            "## Paired Deltas From Full A3",
            "",
            "| Removed/replaced component | Task | Delta FR | Delta desired p | Delta FVA | Delta MNAC | Delta changed 5% | Delta runtime s |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in variants:
        if variant == baseline:
            continue
        for task in tasks:
            value = lambda field: delta_lookup[(variant, task, field)]["delta"]
            lines.append(
                f"| {variant}: {VARIANT_LABELS.get(variant, variant)} | {task} | "
                f"{_format(value('directional_flip'))} | "
                f"{_format(value('desired_probability'))} | "
                f"{_format(value('fva_cosine'))} | {_format(value('mnac'))} | "
                f"{_format(value('changed_fraction_5'))} | "
                f"{_format(value('runtime_seconds'), 2)} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "A component is supported only when removing it causes a repeatable target degradation "
            "without a compensating preservation gain. A zero flip-rate delta with a small "
            "probability change is treated as inconclusive, not as proof that the component helps.",
            "FID/sFID are reported separately when available as exploratory distribution diagnostics.",
        ]
    )
    (root / "component_ablation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return {"aggregate": aggregate_rows, "deltas": delta_rows}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--baseline", default="A3")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summarize(Path(args.experiment_root), baseline=args.baseline, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
