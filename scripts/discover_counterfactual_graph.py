#!/usr/bin/env python3
"""Analyze paired interventions into a counterfactual influence graph."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cci_diff.concept_graph import sha256_file  # noqa: E402
from cci_diff.counterfactual_graph import (  # noqa: E402
    InfluenceGraphResult,
    InterventionObservation,
    aggregate_region_sets,
    build_influence_graph,
)


def read_observations(path: str | Path) -> list[InterventionObservation]:
    """Load intervention rows written by the paired runner."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Intervention results not found: {source}")
    observations = []
    with source.open(newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                regions = tuple(json.loads(row["regions"]))
                observations.append(
                    InterventionObservation(
                        target=row["target"],
                        desired_value=int(row["desired_value"]),
                        sample_id=int(row["sample_id"]),
                        seed=int(row["seed"]),
                        regions=regions,
                        source_probability=float(row["source_probability"]),
                        output_probability=float(row["output_probability"]),
                        mask_fraction=_optional_csv_float(
                            row.get("mask_fraction")
                        ),
                        identity_cosine=_optional_csv_float(
                            row.get("identity_cosine")
                        ),
                        non_target_drift=_optional_csv_float(
                            row.get("non_target_drift")
                        ),
                        outside_l1=_optional_csv_float(row.get("outside_l1")),
                        changed_fraction=_optional_csv_float(
                            row.get("changed_fraction")
                        ),
                        output_path=row.get("output_path") or None,
                        audit_path=row.get("audit_path") or None,
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid intervention row {row_number}: {error}"
                ) from error
    if not observations:
        raise ValueError("Intervention results contain no observations")
    return observations


def discover_graph(
    results_path: str | Path,
    output_dir: str | Path,
    *,
    required_flip_rate: float = 0.95,
    minimum_samples: int = 20,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    random_seed: int = 0,
    template_graph_path: str | Path | None = None,
) -> InfluenceGraphResult:
    """Estimate effects, select regions, and write discovery artifacts."""

    observations = read_observations(results_path)
    target = observations[0].target
    desired_value = observations[0].desired_value
    evidence = aggregate_region_sets(
        observations,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        random_seed=random_seed,
    )
    provenance = {
        "intervention_results": str(results_path),
        "intervention_results_sha256": sha256_file(results_path),
        "bootstrap_samples": bootstrap_samples,
        "confidence": confidence,
        "random_seed": random_seed,
        "effect_unit": "desired_class_probability_change",
        "causal_scope": "classifier-specific masked diffusion intervention",
    }
    if template_graph_path is not None:
        provenance["template_graph"] = str(template_graph_path)
        provenance["template_graph_sha256"] = sha256_file(
            template_graph_path
        )
    result = build_influence_graph(
        target=target,
        desired_value=desired_value,
        evidence_by_regions=evidence,
        required_flip_rate=required_flip_rate,
        minimum_samples=minimum_samples,
        provenance=provenance,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "influence_graph.json").write_text(
        json.dumps(result.to_dict(), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_dict_rows(
        destination / "region_set_metrics.csv",
        [item.to_dict() for item in result.evidence],
    )
    _write_dict_rows(
        destination / "interactions.csv",
        [item.to_dict() for item in result.interactions],
    )
    if template_graph_path is not None:
        _write_execution_graph(
            template_graph_path,
            result,
            destination / "selected_execution_graph.json",
        )
    (destination / "discovery_report.md").write_text(
        _render_report(result),
        encoding="utf-8",
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--required_flip_rate", type=float, default=0.95)
    parser.add_argument("--minimum_samples", type=int, default=20)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--template_graph", default=None)
    return parser


def _write_execution_graph(
    template_path: str | Path,
    result: InfluenceGraphResult,
    output_path: Path,
) -> None:
    payload = json.loads(Path(template_path).read_text(encoding="utf-8"))
    payload["region"]["audit_role"] = "target_region"
    payload["region"]["components"] = list(result.selected_regions)
    payload["discovery"] = {
        "graph_type": "classifier_counterfactual_influence",
        "selection_status": result.selection_status,
        "required_flip_rate": result.required_flip_rate,
        "selected_regions": list(result.selected_regions),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _render_report(result: InfluenceGraphResult) -> str:
    selected = ", ".join(result.selected_regions)
    lines = [
        "# Counterfactual Influence Discovery",
        "",
        f"- Target: `{result.target}` -> `{result.desired_value}`",
        f"- Selection: `{selected}`",
        f"- Status: `{result.selection_status}`",
        "- Selection rule: `Pareto target-efficiency selection`",
        "- Legacy required flip rate "
        f"(not used for selection): `{result.required_flip_rate:.3f}`",
        f"- Verified singleton edges: `{len(result.verified_edges)}`",
        "",
        "The edges describe classifier-specific effects under the measured "
        "masked diffusion intervention. They are not biological causal claims.",
        "",
        "## Region Sets",
        "",
        "| Regions | Pareto | Efficiency | FR | Mean effect | 95% CI | "
        "Mask fraction | Dominated by |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in result.evidence:
        mask = (
            "-"
            if item.mean_mask_fraction is None
            else f"{item.mean_mask_fraction:.4f}"
        )
        efficiency = (
            "-"
            if item.target_efficiency is None
            else f"{item.target_efficiency:.4f}"
        )
        dominated_by = ", ".join(
            "+".join(regions) for regions in item.dominated_by
        ) or "-"
        lines.append(
            f"| {', '.join(item.regions)} | {item.pareto_optimal} | "
            f"{efficiency} | {item.flip_rate:.3f} | "
            f"{item.mean_effect:.4f} | "
            f"[{item.effect_ci_low:.4f}, {item.effect_ci_high:.4f}] | "
            f"{mask} | {dominated_by} |"
        )
    return "\n".join(lines) + "\n"


def _write_dict_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    items = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not items:
            return
        fieldnames = list(items[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in items:
            writer.writerow(
                {
                    key: json.dumps(value)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _optional_csv_float(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def main() -> int:
    args = build_arg_parser().parse_args()
    discover_graph(
        args.results,
        args.output_dir,
        required_flip_rate=args.required_flip_rate,
        minimum_samples=args.minimum_samples,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        random_seed=args.random_seed,
        template_graph_path=args.template_graph,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
