#!/usr/bin/env python3
"""Join two completed clean-CCI runs and report paired inference-step deltas."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


METRICS = (
    "desired_probability",
    "fva_cosine",
    "fs_cosine",
    "mnac",
    "changed_fraction_5",
    "outside_semantic_fraction_5",
    "runtime_seconds",
)


def _read_rows(root: Path) -> dict[tuple[str, int], dict[str, str]]:
    with (root / "ace_pair_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {(row["feature"], int(float(row["sample_id"]))): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate feature/sample rows in {root}")
    return keyed


def _boolean(value: str) -> bool:
    return value.strip().lower() == "true"


def compare(
    step35_root: Path,
    step50_root: Path,
    output_dir: Path,
    *,
    method_label: str = "A3 CCI",
) -> dict:
    rows35 = _read_rows(step35_root)
    rows50 = _read_rows(step50_root)
    if rows35.keys() != rows50.keys():
        raise ValueError("The 35-step and 50-step cohorts are not identical")

    output_dir.mkdir(parents=True, exist_ok=True)
    paired_rows = []
    summary = {}
    for feature in sorted({key[0] for key in rows35}):
        keys = sorted(key for key in rows35 if key[0] == feature)
        transitions = Counter()
        deltas = {metric: [] for metric in METRICS}
        for key in keys:
            row35 = rows35[key]
            row50 = rows50[key]
            success35 = _boolean(row35["target_success"])
            success50 = _boolean(row50["target_success"])
            transition = (
                "both" if success35 and success50
                else "35_only" if success35
                else "50_only" if success50
                else "neither"
            )
            transitions[transition] += 1
            paired = {
                "feature": feature,
                "sample_id": key[1],
                "source_path": row35["source_path"],
                "input_35_path": row35["input_path"],
                "output_35_path": row35["output_path"],
                "comparison_35_path": row35["comparison_path"],
                "output_50_path": row50["output_path"],
                "comparison_50_path": row50["comparison_path"],
                "target_success_35": success35,
                "target_success_50": success50,
                "success_transition": transition,
            }
            for metric in METRICS:
                value35 = float(row35[metric])
                value50 = float(row50[metric])
                delta = value50 - value35
                paired[f"{metric}_35"] = value35
                paired[f"{metric}_50"] = value50
                paired[f"{metric}_delta_50_minus_35"] = delta
                deltas[metric].append(delta)
            paired_rows.append(paired)
        summary[feature] = {
            "count": len(keys),
            "fr_35": sum(_boolean(rows35[key]["target_success"]) for key in keys) / len(keys),
            "fr_50": sum(_boolean(rows50[key]["target_success"]) for key in keys) / len(keys),
            "transitions": dict(transitions),
            "mean_delta_50_minus_35": {
                metric: sum(values) / len(values) for metric, values in deltas.items()
            },
        }

    with (output_dir / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    payload = {
        "method_label": method_label,
        "step35_root": str(step35_root),
        "step50_root": str(step50_root),
        "comparison_is_paired": True,
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )

    lines = [
        f"# {method_label} Inference-Step Comparison",
        "",
        f"The same paired sources, seed, {method_label} configuration, prompt, model, and x4/y4/f3 mask were used in both runs. Only the denoising step count changed.",
        "",
        "| Task | N | ACE FR 35 | ACE FR 50 | Both pass | 35 only | 50 only | Neither | Mean desired-probability delta (50-35) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for feature, values in summary.items():
        transitions = values["transitions"]
        lines.append(
            f"| {feature} | {values['count']} | {100 * values['fr_35']:.1f}% | "
            f"{100 * values['fr_50']:.1f}% | {transitions.get('both', 0)} | "
            f"{transitions.get('35_only', 0)} | {transitions.get('50_only', 0)} | "
            f"{transitions.get('neither', 0)} | "
            f"{values['mean_delta_50_minus_35']['desired_probability']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Observed ACE FR by task: "
            + "; ".join(
                f"{feature}: 35={100 * values['fr_35']:.1f}%, "
                f"50={100 * values['fr_50']:.1f}%"
                for feature, values in summary.items()
            )
            + ".",
            "",
            "`paired_metrics.csv` contains every input path, both output paths, both input-output comparison paths, target outcomes, and metric deltas.",
        ]
    )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps35-root", type=Path, required=True)
    parser.add_argument("--steps50-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-label", default="A3 CCI")
    args = parser.parse_args()
    compare(
        args.steps35_root,
        args.steps50_root,
        args.output_dir,
        method_label=args.method_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
