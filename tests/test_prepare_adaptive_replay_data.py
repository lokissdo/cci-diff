import csv
import json
from pathlib import Path

import pytest

from scripts.prepare_adaptive_replay_data import prepare_adaptive_replay_data


MOUTH = ("mouth",)
PERIORAL = ("lower_lip", "mouth", "upper_lip")


def write_results(path, sample_ids):
    rows = []
    for sample_id in sample_ids:
        for variant in ("A0", "A11"):
            rows.append(
                {
                    "feature": "smile",
                    "sample_id": sample_id,
                    "variant": variant,
                    "source_probability": 0.9,
                    "desired_probability": 0.8 if sample_id % 3 else 0.4,
                    "identity_cosine": 0.95,
                    "outside_semantic_l1": 0.01,
                    "semantic_mask_fraction": 0.02,
                    "non_target_drift": 0.01,
                    "changed_fraction_1": 0.03,
                    "output_path": f"generated/{sample_id}/{variant}.png",
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_preparation_writes_four_disjoint_roles_without_evaluation_outputs(
    tmp_path,
):
    sample_ids = tuple(range(12))

    result = prepare_adaptive_replay_data(
        {
            MOUTH: write_results(tmp_path / "mouth.csv", sample_ids),
            PERIORAL: write_results(tmp_path / "perioral.csv", sample_ids),
        },
        tmp_path / "prepared",
        sample_ids=sample_ids,
        discovery_count=2,
        fit_count=4,
        calibration_count=4,
        evaluation_count=2,
        random_seed=42,
        variant="A11",
    )

    cohorts = result["cohorts"]
    assert {name: len(values) for name, values in cohorts.items()} == {
        "discovery": 2,
        "fit": 4,
        "calibration": 4,
        "evaluation": 2,
    }
    names = tuple(cohorts)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            assert set(cohorts[names[left]]).isdisjoint(cohorts[names[right]])

    with (tmp_path / "prepared/discovery_interventions.csv").open() as handle:
        discovery = list(csv.DictReader(handle))
    assert len(discovery) == 4
    assert {tuple(json.loads(row["regions"])) for row in discovery} == {
        MOUTH,
        PERIORAL,
    }

    with (tmp_path / "prepared/development_outcomes.csv").open() as handle:
        development = list(csv.DictReader(handle))
    assert len(development) == 16
    assert {row["cohort"] for row in development} == {"fit", "calibration"}
    assert all("output_path" not in row for row in development)
    assert all(
        float(row["identity_distance"]) == pytest.approx(0.05)
        for row in development
    )
    assert (
        set(int(row["sample_id"]) for row in development)
        & set(cohorts["evaluation"])
        == set()
    )
    assert (tmp_path / "prepared/split_manifest.json").is_file()
    assert (tmp_path / "prepared/evaluation_ids.json").is_file()


def test_preparation_is_deterministic_for_same_seed(tmp_path):
    sample_ids = tuple(range(12))
    roots = {
        MOUTH: write_results(tmp_path / "mouth.csv", sample_ids),
        PERIORAL: write_results(tmp_path / "perioral.csv", sample_ids),
    }

    first = prepare_adaptive_replay_data(
        roots,
        tmp_path / "first",
        sample_ids=sample_ids,
        discovery_count=2,
        fit_count=4,
        calibration_count=4,
        evaluation_count=2,
        random_seed=7,
    )
    second = prepare_adaptive_replay_data(
        roots,
        tmp_path / "second",
        sample_ids=sample_ids,
        discovery_count=2,
        fit_count=4,
        calibration_count=4,
        evaluation_count=2,
        random_seed=7,
    )

    assert first["cohorts"] == second["cohorts"]
    assert (tmp_path / "first/development_outcomes.csv").read_bytes() == (
        tmp_path / "second/development_outcomes.csv"
    ).read_bytes()
