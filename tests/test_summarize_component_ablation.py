import csv

import pytest

from scripts.summarize_component_ablation import METRICS, paired_metric_delta, summarize


def test_paired_metric_delta_aligns_samples_before_comparison():
    baseline = [
        {"sample_id": "1", "score": "0.9"},
        {"sample_id": "0", "score": "0.7"},
    ]
    ablation = [
        {"sample_id": "0", "score": "0.4"},
        {"sample_id": "1", "score": "0.8"},
    ]

    result = paired_metric_delta(baseline, ablation, "score")

    assert result["count"] == 2
    assert result["baseline_mean"] == pytest.approx(0.8)
    assert result["ablation_mean"] == pytest.approx(0.6)
    assert result["delta"] == pytest.approx(0.2)


def test_paired_metric_delta_rejects_unmatched_samples():
    with pytest.raises(ValueError, match="sample IDs"):
        paired_metric_delta(
            [{"sample_id": "0", "score": "1"}],
            [{"sample_id": "1", "score": "1"}],
            "score",
        )


def test_report_does_not_assume_a_ten_image_fid_sample(tmp_path):
    rows = []
    for variant in ("A2", "A3"):
        row = {"variant": variant, "feature": "smile", "sample_id": "0"}
        row.update({metric: "0" for metric in METRICS})
        rows.append(row)
    with (tmp_path / "ace_pair_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    summarize(tmp_path)

    report = (tmp_path / "component_ablation_report.md").read_text(
        encoding="utf-8"
    )
    assert "from ten images" not in report
    assert "reported separately when available" in report
