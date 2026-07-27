import csv

from scripts.compare_inference_step_runs import compare


FIELDS = (
    "feature",
    "sample_id",
    "source_path",
    "input_path",
    "output_path",
    "comparison_path",
    "target_success",
    "desired_probability",
    "fva_cosine",
    "fs_cosine",
    "mnac",
    "changed_fraction_5",
    "outside_semantic_fraction_5",
    "runtime_seconds",
)


def _write_run(root, success):
    root.mkdir()
    row = {
        "feature": "smile",
        "sample_id": 0,
        "source_path": "source.jpg",
        "input_path": "input.jpg",
        "output_path": "output.jpg",
        "comparison_path": "comparison.jpg",
        "target_success": str(success),
        "desired_probability": 0.6 if success else 0.4,
        "fva_cosine": 0.9,
        "fs_cosine": 0.9,
        "mnac": 1,
        "changed_fraction_5": 0.1,
        "outside_semantic_fraction_5": 0.05,
        "runtime_seconds": 1,
    }
    with (root / "ace_pair_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)


def test_report_uses_method_label_and_observed_step_result(tmp_path):
    run35 = tmp_path / "run35"
    run50 = tmp_path / "run50"
    output = tmp_path / "comparison"
    _write_run(run35, False)
    _write_run(run50, True)

    compare(run35, run50, output, method_label="Raw BLD")

    report = (output / "comparison.md").read_text()
    assert report.startswith("# Raw BLD Inference-Step Comparison")
    assert "A3 controller" not in report
    assert "smile: 35=0.0%, 50=100.0%" in report
    assert "35 steps as the current default" not in report
