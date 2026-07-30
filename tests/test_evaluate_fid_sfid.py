import csv

import numpy as np
import pytest

from scripts.evaluate_fid_sfid import (
    extract_or_load_activations,
    fid_from_activations,
    fid_sfid_from_activations,
    load_experiment_rows,
    runtime_summary_value,
    split_indices,
    summarize_pair_rows,
    validate_aligned_cohorts,
    write_reports,
)


def test_split_indices_is_deterministic_and_complete():
    first_a, first_b = split_indices(100, 42)
    second_a, second_b = split_indices(100, 42)

    np.testing.assert_array_equal(first_a, second_a)
    np.testing.assert_array_equal(first_b, second_b)
    assert len(first_a) == len(first_b) == 50
    assert set(first_a).isdisjoint(first_b)
    assert set(np.concatenate((first_a, first_b))) == set(range(100))


def test_fid_and_sfid_are_zero_for_identical_distributions():
    rng = np.random.default_rng(7)
    activations = rng.normal(size=(20, 4))

    assert fid_from_activations(activations, activations) == pytest.approx(
        0.0, abs=1e-8
    )
    result = fid_sfid_from_activations(activations, activations, seed=42)

    assert result["sfid"] == pytest.approx(
        (result["sfid_1"] + result["sfid_2"]) / 2
    )


@pytest.mark.parametrize("count", [3, 5])
def test_split_indices_rejects_small_or_odd_counts(count):
    with pytest.raises(ValueError):
        split_indices(count, 42)


def test_fid_rejects_misaligned_or_nonfinite_activations():
    with pytest.raises(ValueError, match="shape"):
        fid_from_activations(np.zeros((4, 2)), np.zeros((5, 2)))
    invalid = np.zeros((4, 2))
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fid_from_activations(invalid, invalid)


def _write_experiment(
    root,
    *,
    count=100,
    duplicate=False,
    id_offset=0,
    variants=(),
    tasks=("smile", "hair"),
):
    root.mkdir()
    rows = []
    for feature in tasks:
        for index in range(count):
            sample_id = id_offset + index
            source = root / f"{feature}_{sample_id}_source.png"
            output = root / f"{feature}_{sample_id}_output.png"
            source.write_bytes(b"source")
            output.write_bytes(b"output")
            rows.append(
                {
                    "feature": feature,
                    "sample_id": sample_id,
                    "source_path": source,
                    "output_path": output,
                }
            )
    if duplicate:
        rows[-1]["sample_id"] = rows[-2]["sample_id"]
    if variants:
        rows = [dict(row, variant=variant) for variant in variants for row in rows]
    with (root / "ace_pair_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_load_experiment_rows_requires_complete_unique_pairs(tmp_path):
    valid = tmp_path / "valid"
    _write_experiment(valid)
    loaded = load_experiment_rows(valid)
    assert len(loaded["smile"]) == len(loaded["hair"]) == 100

    incomplete = tmp_path / "incomplete"
    _write_experiment(incomplete, count=99)
    with pytest.raises(ValueError, match="100"):
        load_experiment_rows(incomplete)

    duplicate = tmp_path / "duplicate"
    _write_experiment(duplicate, duplicate=True)
    with pytest.raises(ValueError, match="duplicate"):
        load_experiment_rows(duplicate)


def test_load_experiment_rows_filters_variant_and_accepts_pilot_count(tmp_path):
    root = tmp_path / "ablation"
    _write_experiment(root, count=10, variants=("A3", "A4"))

    grouped = load_experiment_rows(root, expected_count=10, variant="A4")

    assert len(grouped["smile"]) == len(grouped["hair"]) == 10
    assert {row["variant"] for row in grouped["smile"]} == {"A4"}


def test_load_experiment_rows_accepts_smile_only_300_cohort(tmp_path):
    root = tmp_path / "smile_only"
    _write_experiment(root, count=300, tasks=("smile",))

    grouped = load_experiment_rows(
        root,
        expected_count=300,
        tasks=("smile",),
    )

    assert set(grouped) == {"smile"}
    assert len(grouped["smile"]) == 300


def test_parser_accepts_smile_only_task():
    from scripts.evaluate_fid_sfid import build_arg_parser

    args = build_arg_parser().parse_args(
        [
            "--experiment",
            "A0",
            "35",
            "outputs/test",
            "--experiment",
            "A11",
            "35",
            "outputs/test",
            "--output-dir",
            "outputs/metrics",
            "--tasks",
            "smile",
        ]
    )

    assert args.tasks == ["smile"]


def test_validate_aligned_cohorts_rejects_different_ids(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_experiment(first)
    _write_experiment(second, id_offset=1)
    with pytest.raises(ValueError, match="cohort"):
        validate_aligned_cohorts(
            {"first": load_experiment_rows(first), "second": load_experiment_rows(second)}
        )


def test_activation_cache_reuses_matching_fingerprint(tmp_path):
    images = [tmp_path / "a.png", tmp_path / "b.png"]
    for image in images:
        image.write_bytes(image.name.encode())
    cache = tmp_path / "features.npz"
    calls = []

    def extractor(paths):
        calls.append(list(paths))
        return np.arange(8, dtype=float).reshape(2, 4)

    first = extract_or_load_activations(images, cache, extractor)
    second = extract_or_load_activations(images, cache, extractor)
    np.testing.assert_array_equal(first, second)
    assert len(calls) == 1

    images[0].write_bytes(b"changed-size")
    extract_or_load_activations(images, cache, extractor)
    assert len(calls) == 2


def test_summarize_pair_rows_separates_accuracy_and_directional_fr():
    ace_rows = [
        {
            "target_success": "True",
            "directional_flip": "False",
            "desired_probability": "0.7",
            "fva_cosine": "0.9",
            "fs_cosine": "0.8",
            "mnac": "2",
            "cout": "0.1",
            "changed_fraction_5": "0.3",
            "outside_semantic_fraction_5": "0.2",
            "outside_generation_fraction_5": "0.1",
            "inside_semantic_l1": "0.05",
            "outside_semantic_l1": "0.01",
        },
        {
            "target_success": "True",
            "directional_flip": "True",
            "desired_probability": "0.9",
            "fva_cosine": "0.95",
            "fs_cosine": "0.85",
            "mnac": "1",
            "cout": "0.3",
            "changed_fraction_5": "0.4",
            "outside_semantic_fraction_5": "0.25",
            "outside_generation_fraction_5": "0.15",
            "inside_semantic_l1": "0.06",
            "outside_semantic_l1": "0.02",
        },
    ]
    pilot_rows = [{"desired_probability": "0.6"}, {"desired_probability": "0.85"}]

    result = summarize_pair_rows(ace_rows, pilot_rows)

    assert result["target_accuracy"] == 1.0
    assert result["directional_fr"] == 0.5
    assert result["same_classifier_fr_05"] == 1.0
    assert result["strong_target_rate_08"] == 0.5
    assert result["desired_probability"] == pytest.approx(0.8)
    assert result["cout"] == pytest.approx(0.2)


def test_runtime_summary_value_preserves_missing_legacy_runtime():
    summary = {
        "features": {
            "smile": {
                "variants": {
                    "A1": {"median_runtime": None},
                    "A3": {"median_runtime": 12.5},
                }
            }
        }
    }

    assert runtime_summary_value(summary, "smile", "A1") is None
    assert runtime_summary_value(summary, "smile", "A3") == 12.5


def test_write_reports_contains_all_eight_rows_and_full_columns(tmp_path):
    fid_rows = []
    full_rows = []
    for method in ("BLD", "CCI"):
        for steps in (35, 50):
            for task in ("smile", "hair"):
                fid = {
                    "method": method,
                    "steps": steps,
                    "task": task,
                    "n": 100,
                    "fid": 1.0,
                    "sfid_1": 2.0,
                    "sfid_2": 4.0,
                    "sfid": 3.0,
                }
                fid_rows.append(fid)
                full_rows.append(
                    {
                        **fid,
                        "target_accuracy": 0.2,
                        "directional_fr": 0.1,
                        "same_classifier_fr_05": 0.5,
                        "strong_target_rate_08": 0.4,
                        "desired_probability": 0.3,
                        "fva_rate": 1.0,
                        "fva_cosine": 0.9,
                        "fs": 0.8,
                        "mnac": 1.0,
                        "cd": 2.0,
                        "cout": 0.25,
                        "changed_fraction_5": 0.2,
                        "outside_semantic_fraction_5": 0.1,
                        "outside_generation_fraction_5": 0.1,
                        "inside_semantic_l1": 0.05,
                        "outside_semantic_l1": 0.01,
                        "median_runtime_seconds": 10.0,
                    }
                )

    write_reports(fid_rows, full_rows, tmp_path, {"seed": 42})

    with (tmp_path / "full_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert "directional_fr" in rows[0]
    assert "target_accuracy" in rows[0]
    assert "cout" in rows[0]
    assert (tmp_path / "fid_sfid_metrics.json").is_file()
    report = (tmp_path / "full_metrics.md").read_text().lower()
    assert "exploratory" in report
    assert "cout" in report


def test_write_reports_formats_missing_runtime_as_dash(tmp_path):
    row = {
        "method": "A1",
        "steps": 35,
        "task": "smile",
        "n": 10,
        "fid": 1.0,
        "sfid_1": 2.0,
        "sfid_2": 4.0,
        "sfid": 3.0,
        "target_accuracy": 0.2,
        "directional_fr": 0.1,
        "same_classifier_fr_05": 0.5,
        "strong_target_rate_08": 0.4,
        "desired_probability": 0.3,
        "fva_rate": 1.0,
        "fva_cosine": 0.9,
        "fs": 0.8,
        "mnac": 1.0,
        "cd": 2.0,
        "cout": 0.25,
        "changed_fraction_5": 0.2,
        "outside_semantic_fraction_5": 0.1,
        "outside_generation_fraction_5": 0.1,
        "inside_semantic_l1": 0.05,
        "outside_semantic_l1": 0.01,
        "median_runtime_seconds": None,
    }

    write_reports([row], [row], tmp_path, {"seed": 42})

    assert "| - |" in (tmp_path / "full_metrics.md").read_text()
