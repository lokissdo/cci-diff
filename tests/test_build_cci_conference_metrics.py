import csv
from pathlib import Path

import pytest

from scripts.build_cci_conference_metrics import build_metrics, write_outputs


FULL_FIELDS = (
    "method",
    "task",
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
PAIR_FIELDS = (
    "feature",
    "sample_id",
    "variant",
    "directional_flip",
    "fva_cosine",
    "fs_cosine",
    "mnac",
    "cout",
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _full_rows() -> list[dict]:
    return [
        {
            "method": "A0",
            "task": "smile",
            "n": 2,
            "fid": 1.0,
            "sfid": 2.0,
            "fva_rate": 0.5,
            "fs": 0.75,
            "mnac": 1.5,
            "cd": 2.0,
            "cout": 0.1,
            "directional_fr": 0.5,
        },
        {
            "method": "A11",
            "task": "smile",
            "n": 2,
            "fid": 1.1,
            "sfid": 1.9,
            "fva_rate": 1.0,
            "fs": 0.85,
            "mnac": 1.0,
            "cd": 1.8,
            "cout": 0.5,
            "directional_fr": 1.0,
        },
    ]


def _pair_rows() -> list[dict]:
    return [
        {
            "feature": "smile",
            "sample_id": 1,
            "variant": "A0",
            "directional_flip": "False",
            "fva_cosine": 0.4,
            "fs_cosine": 0.7,
            "mnac": 1.0,
            "cout": 0.0,
        },
        {
            "feature": "smile",
            "sample_id": 2,
            "variant": "A0",
            "directional_flip": "True",
            "fva_cosine": 0.6,
            "fs_cosine": 0.8,
            "mnac": 2.0,
            "cout": 0.2,
        },
        {
            "feature": "smile",
            "sample_id": 1,
            "variant": "A11",
            "directional_flip": "True",
            "fva_cosine": 0.6,
            "fs_cosine": 0.8,
            "mnac": 0.5,
            "cout": 0.4,
        },
        {
            "feature": "smile",
            "sample_id": 2,
            "variant": "A11",
            "directional_flip": "True",
            "fva_cosine": 0.7,
            "fs_cosine": 0.9,
            "mnac": 1.5,
            "cout": 0.6,
        },
    ]


def _write_inputs(
    tmp_path: Path,
    full_rows: list[dict] | None = None,
    pair_rows: list[dict] | None = None,
) -> tuple[Path, Path]:
    full_path = tmp_path / "full.csv"
    pair_path = tmp_path / "pairs.csv"
    _write_csv(full_path, FULL_FIELDS, full_rows or _full_rows())
    _write_csv(pair_path, PAIR_FIELDS, pair_rows or _pair_rows())
    return full_path, pair_path


def test_build_metrics_validates_pairs_and_reconciles_aggregates(tmp_path: Path):
    full_path, pair_path = _write_inputs(tmp_path)

    result = build_metrics(full_path, pair_path, expected_count=2)

    assert result["cohort"]["paired_count"] == 2
    assert result["methods"]["A0"]["directional_fr"] == pytest.approx(0.5)
    assert result["methods"]["A11"]["directional_fr"] == pytest.approx(1.0)
    assert result["deltas"]["fr_percentage_points"] == pytest.approx(50.0)
    assert result["deltas"]["cout_gain"] == pytest.approx(0.4)
    assert result["deltas"]["cd_reduction_percent"] == pytest.approx(10.0)


def test_write_outputs_formats_completed_paper_values(tmp_path: Path):
    payload = {
        "cohort": {"paired_count": 300},
        "methods": {
            "A0": {
                "fid": 17.372000862572463,
                "sfid": 72.5453010756013,
                "fva_rate": 1.0,
                "fs": 0.9961487014613934,
                "mnac": 2.4966666666666666,
                "cd": 2.9437373503318023,
                "cout": -0.09817892722214261,
                "directional_fr": 0.6433333333333333,
            },
            "A11": {
                "fid": 17.434511701400766,
                "sfid": 72.39167809062245,
                "fva_rate": 1.0,
                "fs": 0.9957703433020877,
                "mnac": 2.6233333333333335,
                "cd": 2.8893835735054174,
                "cout": 0.1151902814358473,
                "directional_fr": 0.81,
            },
        },
        "deltas": {
            "fr_percentage_points": 16.666666666666668,
            "cout_gain": 0.2133692086579899,
            "cd_reduction_percent": 1.846416728515675,
        },
    }
    json_path = tmp_path / "metrics.json"
    tex_path = tmp_path / "metrics.tex"

    write_outputs(payload, json_path, tex_path)

    tex = tex_path.read_text(encoding="utf-8")
    assert "\\newcommand{\\EndToEndBLDFID}{17.3720}" in tex
    assert "\\newcommand{\\EndToEndAdaptiveFRPct}{81.0}" in tex
    assert "\\newcommand{\\EndToEndFRGainPctPoints}{16.7}" in tex
    assert "\\newcommand{\\EndToEndCOUTGain}{0.2134}" in tex
    assert "\\newcommand{\\EndToEndCDReductionPct}{1.8}" in tex
    assert "SampleCount" not in tex
    assert '"paired_count": 300' in json_path.read_text(encoding="utf-8")


def test_build_metrics_rejects_duplicate_pair_key(tmp_path: Path):
    rows = _pair_rows()
    rows.append(dict(rows[0]))
    full_path, pair_path = _write_inputs(tmp_path, pair_rows=rows)

    with pytest.raises(ValueError, match="duplicate pair key"):
        build_metrics(full_path, pair_path, expected_count=2)


def test_build_metrics_rejects_mismatched_method_ids(tmp_path: Path):
    rows = _pair_rows()
    rows[-1]["sample_id"] = 3
    full_path, pair_path = _write_inputs(tmp_path, pair_rows=rows)

    with pytest.raises(ValueError, match="paired ID sets differ"):
        build_metrics(full_path, pair_path, expected_count=2)


def test_build_metrics_rejects_unexpected_count_or_missing_method(tmp_path: Path):
    full_path, pair_path = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="expected 3"):
        build_metrics(full_path, pair_path, expected_count=3)

    full_path, pair_path = _write_inputs(tmp_path, full_rows=[_full_rows()[0]])
    with pytest.raises(ValueError, match="exactly A0 and A11"):
        build_metrics(full_path, pair_path, expected_count=2)


def test_build_metrics_rejects_nonfinite_or_disagreeing_aggregate(tmp_path: Path):
    full_rows = _full_rows()
    full_rows[0]["fid"] = "nan"
    full_path, pair_path = _write_inputs(tmp_path, full_rows=full_rows)
    with pytest.raises(ValueError, match="must be finite"):
        build_metrics(full_path, pair_path, expected_count=2)

    full_rows = _full_rows()
    full_rows[1]["fs"] = 0.1
    full_path, pair_path = _write_inputs(tmp_path, full_rows=full_rows)
    with pytest.raises(ValueError, match="pair-level fs disagrees"):
        build_metrics(full_path, pair_path, expected_count=2)
