from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


REQUESTED_COLUMNS = (
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


def _write_metrics(path: Path, *, count: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in ("A0", "A11"):
        rows.append(
            {
                "method": method,
                "steps": 35,
                "task": "smile",
                "n": count,
                "fid": 1.0,
                "sfid": 2.0,
                "fva_rate": 0.9,
                "fs": 0.8,
                "mnac": 1.2,
                "cd": 2.3,
                "cout": 0.4,
                "directional_fr": 0.7,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_combiner_writes_four_validated_region_method_rows(tmp_path):
    from scripts.combine_attacked_region_metrics import combine_region_metrics

    mouth = tmp_path / "mouth" / "full_metrics.csv"
    lips = tmp_path / "lips" / "full_metrics.csv"
    _write_metrics(mouth)
    _write_metrics(lips)

    rows = combine_region_metrics(
        [("mouth", mouth), ("mouth_upper_lower_lip", lips)],
        output_dir=tmp_path / "combined",
        expected_count=300,
    )

    assert len(rows) == 4
    assert tuple(rows[0]) == REQUESTED_COLUMNS
    assert {(row["region"], row["method"]) for row in rows} == {
        ("mouth", "A0"),
        ("mouth", "A11"),
        ("mouth_upper_lower_lip", "A0"),
        ("mouth_upper_lower_lip", "A11"),
    }
    assert (tmp_path / "combined" / "combined_metrics.csv").is_file()
    report = (
        tmp_path / "combined" / "combined_metrics.md"
    ).read_text(encoding="utf-8")
    assert "FID" in report
    assert "sFID" in report
    assert "COUT" in report
    assert "FR (%)" in report


def test_combiner_rejects_incomplete_count(tmp_path):
    from scripts.combine_attacked_region_metrics import combine_region_metrics

    table = tmp_path / "full_metrics.csv"
    _write_metrics(table, count=299)

    with pytest.raises(ValueError, match="300"):
        combine_region_metrics(
            [("mouth", table)],
            output_dir=tmp_path / "combined",
            expected_count=300,
        )


def test_scheduler_contains_both_sequential_attacked_region_jobs():
    script = Path("scripts/run_attacked_region_300.sh").read_text(
        encoding="utf-8"
    )

    required = (
        "set -euo pipefail",
        "caffeinate -dimsu",
        "--limit 300",
        "--seed 42",
        "--random_sample_seed 42",
        "--controller_modes disabled trust_region",
        "--region_components mouth",
        "--region_components mouth upper_lip lower_lip",
        "--sample_ids_manifest",
        "--cci_post_attack smooth_boundary",
        "--attribute_classifier_path",
        "--tasks smile",
        "combine_attacked_region_metrics.py",
    )
    for value in required:
        assert value in script

    assert script.index("--region_components mouth") < script.index(
        "--region_components mouth upper_lip lower_lip"
    )


def test_scheduler_excludes_only_unreliable_face_image():
    script = Path("scripts/run_attacked_region_300.sh").read_text(
        encoding="utf-8"
    )
    exclusions = json.loads(
        Path("examples/attacked_region_excluded_ids.json").read_text(
            encoding="utf-8"
        )
    )

    assert '--exclude_ids_json "$EXCLUDED_IDS"' in script
    assert exclusions == {"smile": [10260]}
