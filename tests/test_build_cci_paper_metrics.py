import csv
import json
from pathlib import Path

import pytest

from scripts.build_cci_paper_metrics import build_metrics, write_tex


FIELDS = (
    "feature",
    "sample_id",
    "variant",
    "target_pass",
    "feasible",
    "identity_cosine",
    "strict_outside_mae",
    "non_target_drift",
    "runtime_seconds",
)


def write_cohort(path: Path, ids: tuple[int, ...], offset: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for sample_id in ids:
            for variant, flip, feasible, identity, outside, drift, runtime in (
                ("A0", False, False, 0.80, 4.0, 0.08, 20.0),
                ("A10", True, True, 0.90, 4.1, 0.07, 80.0),
                ("A11", True, True, 0.91, 4.2, 0.06, 90.0),
            ):
                writer.writerow(
                    {
                        "feature": "smile",
                        "sample_id": sample_id,
                        "variant": variant,
                        "target_pass": flip,
                        "feasible": feasible,
                        "identity_cosine": identity + offset,
                        "strict_outside_mae": outside,
                        "non_target_drift": drift,
                        "runtime_seconds": runtime,
                    }
                )


def write_ablation(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "sample_id": 26811,
                "before": {
                    "desired_probability": 0.10,
                    "identity_cosine": 0.83,
                    "mean_non_target_drift": 0.03,
                    "wall_seconds": 53.0,
                },
                "after": {
                    "desired_probability": 0.79,
                    "identity_cosine": 0.87,
                    "mean_non_target_drift": 0.06,
                    "wall_seconds": 85.0,
                },
                "pixel": {
                    "mean_absolute_difference": 0.0008,
                    "maximum_absolute_difference": 0.29,
                    "changed_fraction": 0.04,
                },
                "restoration": {"accepted_steps": 8},
            }
        ),
        encoding="utf-8",
    )


def test_build_metrics_aggregates_disjoint_paired_cohorts(tmp_path: Path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    ablation = tmp_path / "ablation.json"
    write_cohort(first, (1, 2), 0.00)
    write_cohort(second, (3, 4), 0.02)
    write_ablation(ablation)

    result = build_metrics([first, second], 2, ablation)

    assert result["cohort"]["unique_sources"] == 4
    assert result["methods"]["A11"]["count"] == 4
    assert result["methods"]["A11"]["flip_rate"] == pytest.approx(1.0)
    assert result["methods"]["A11"]["mean_identity_cosine"] == pytest.approx(
        0.92
    )
    assert result["restoration"]["accepted_steps"] == 8

    tex_path = tmp_path / "metrics.tex"
    write_tex(result, tex_path)
    assert "\\newcommand{\\AdaptiveCCISampleCount}{4}" in tex_path.read_text()
    assert "\\newcommand{\\AdaptiveCCIFlipRatePct}{100.0}" in tex_path.read_text()


def test_build_metrics_rejects_cross_cohort_duplicate_sources(tmp_path: Path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    ablation = tmp_path / "ablation.json"
    write_cohort(first, (1, 2), 0.00)
    write_cohort(second, (2, 3), 0.00)
    write_ablation(ablation)

    with pytest.raises(ValueError, match="cohorts are not disjoint"):
        build_metrics([first, second], 2, ablation)
