#!/usr/bin/env python3
"""Export clean CCI JSONL traces as wide CSV and optional PNG plots."""

from __future__ import annotations

import argparse
from typing import Any

from cci_diff.cci_trace import load_cci_trace


def trace_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = sorted(
        {
            name
            for record in records
            for name in record["constraints"]
        }
    )
    rows = []
    for record in records:
        row = {
            "step": record["step"],
            "timestep": record["timestep"],
            "progress": record["progress"],
            "target_probability": record["target"]["target_probability"],
            "required_probability": record["target"]["required_probability"],
            "target_activation": record["target"]["activation"],
            "target_gradient_norm": record["target"]["gradient_norm"],
            "eta": record["update"]["eta"],
            "update_norm": record["update"]["norm"],
            "target_constraint_cosine": record["update"][
                "target_constraint_cosine"
            ],
        }
        for name in names:
            values = record["constraints"].get(name, {})
            row[f"{name}.lambda"] = values.get("lambda_after")
            row[f"{name}.residual"] = values.get("residual")
            row[f"{name}.gradient_norm"] = values.get("gradient_norm")
        rows.append(row)
    return rows


def write_trace_csv(records: list[dict[str, Any]], path: str) -> None:
    import csv
    from pathlib import Path

    rows = trace_rows(records)
    if not rows:
        raise ValueError("Cannot export an empty CCI trace")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_trace(records: list[dict[str, Any]], path: str) -> None:
    from pathlib import Path

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Trace PNG output requires: pip install -e '.[plot]'"
        ) from exc
    rows = trace_rows(records)
    if not rows:
        raise ValueError("Cannot plot an empty CCI trace")
    steps = [row["step"] for row in rows]
    constraint_names = sorted(
        key.removesuffix(".lambda")
        for key in rows[0]
        if key.endswith(".lambda")
    )
    figure, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(
        steps,
        [row["target_probability"] for row in rows],
        label="desired probability",
    )
    axes[0].plot(
        steps,
        [row["required_probability"] for row in rows],
        "--",
        label="required",
    )
    axes[0].set_ylabel("probability")
    axes[0].legend()
    for name in constraint_names:
        axes[1].plot(
            steps,
            [row[f"{name}.lambda"] for row in rows],
            label=name,
        )
        axes[2].plot(
            steps,
            [row[f"{name}.residual"] for row in rows],
            label=name,
        )
    axes[1].set_ylabel("dual lambda")
    axes[1].legend()
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_ylabel("normalized residual")
    axes[2].legend()
    axes[3].plot(
        steps,
        [row["target_gradient_norm"] for row in rows],
        label="target grad",
    )
    for name in constraint_names:
        axes[3].plot(
            steps,
            [row[f"{name}.gradient_norm"] for row in rows],
            label=f"{name} grad",
        )
    axes[3].plot(
        steps,
        [row["update_norm"] for row in rows],
        label="update norm",
    )
    axes[3].plot(steps, [row["eta"] for row in rows], label="eta")
    axes[3].plot(
        steps,
        [row["target_constraint_cosine"] for row in rows],
        label="cosine",
    )
    axes[3].set_xlabel("selected denoising step")
    axes[3].legend()
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--png", default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    records = load_cci_trace(args.trace)
    write_trace_csv(records, args.csv)
    if args.png:
        plot_trace(records, args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
