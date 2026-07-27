from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.run_fid_rerank_smoke import (
    build_arg_parser,
    build_pilot_commands,
    load_seed_candidate_pool,
    select_and_materialize,
    write_reference_manifest,
)


def _args(tmp_path):
    return build_arg_parser().parse_args(
        [
            "--classifier_path",
            "models/classifier.pth",
            "--identity_model_path",
            "models/identity.ts",
            "--output_dir",
            str(tmp_path / "smoke"),
        ]
    )


def test_smoke_cli_defaults_match_approved_design(tmp_path):
    args = _args(tmp_path)

    assert args.seeds == [42, 43, 44, 45]
    assert args.limit == 10
    assert args.reference_count == 1000
    assert args.proxy_dims == 64
    assert args.minimum_passes == 10
    assert args.num_inference_steps == 35


def test_smoke_cli_can_run_as_a_direct_script():
    completed = subprocess.run(
        [sys.executable, "scripts/run_fid_rerank_smoke.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--selection_only" in completed.stdout


def test_build_pilot_commands_pin_a3_mask_and_attack_configuration(tmp_path):
    commands = build_pilot_commands(_args(tmp_path))

    assert len(commands) == 4
    assert {command[command.index("--seed") + 1] for command in commands} == {
        "42",
        "43",
        "44",
        "45",
    }
    for command in commands:
        assert command[command.index("--variants") + 1] == "A3"
        assert command[command.index("--mask_shapes") + 1] == "4,4,3"
        assert command[command.index("--cci_post_attack") + 1] == "smooth_boundary"
        assert (
            command[command.index("--cci_post_attack_epsilon_schedule") + 1]
            == "0.05,0.08,0.10,0.30,0.50"
        )
        assert (
            command[command.index("--cci_post_attack_boundary_margin") + 1]
            == "0.03"
        )


def _save_image(path: Path, value: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (value, value, value)).save(path)


def _write_seed_cohort(root: Path, seed: int, sample_ids=(0, 1)):
    seed_root = root / "seeds" / f"seed_{seed}"
    seed_root.mkdir(parents=True)
    rows = []
    for sample_id in sample_ids:
        source = seed_root / f"source_{sample_id}.png"
        selected = seed_root / f"selected_{sample_id}.png"
        candidate_dir = seed_root / f"candidate_{sample_id}"
        raw = candidate_dir / "raw.png"
        corrected = candidate_dir / "corrected.png"
        _save_image(source, sample_id)
        _save_image(raw, 100 + sample_id)
        _save_image(corrected, 110 + sample_id)
        _save_image(selected, 110 + sample_id)
        audit = {
            "cci": {
                "post_attack": {
                    "raw_output_path": str(raw),
                    "corrected_output_path": str(corrected),
                    "candidates": [
                        {
                            "before_probability": 0.6,
                            "after_probability": 0.4,
                            "linf": 0.02,
                            "selected_epsilon": 0.05,
                        }
                    ],
                }
            }
        }
        (candidate_dir / "audit.json").write_text(json.dumps(audit))
        rows.append(
            {
                "feature": "smile",
                "sample_id": sample_id,
                "variant": "A3",
                "source_path": source,
                "output_path": selected,
                "candidate_dir": candidate_dir,
                "desired_probability": 0.6,
                "identity_cosine": 0.9,
                "outside_semantic_l1": 0.01,
                "post_attack_selected_epsilon": 0.05,
                "post_attack_linf": 0.02,
            }
        )
    with (seed_root / "pilot_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_load_seed_candidate_pool_is_aligned_and_resolves_attack_artifacts(tmp_path):
    root = tmp_path / "smoke"
    for seed in (42, 43):
        _write_seed_cohort(root, seed)

    rows = load_seed_candidate_pool(root, (42, 43), expected_count=2)

    assert [(row["sample_id"], row["seed"]) for row in rows] == [
        (0, 42),
        (0, 43),
        (1, 42),
        (1, 43),
    ]
    assert len(rows) == 4
    assert all(Path(row["output_path"]).name == "selected_0.png" for row in rows[:2])
    assert Path(rows[0]["raw_output_path"]).name == "raw.png"
    assert Path(rows[0]["corrected_output_path"]).name == "corrected.png"
    assert rows[0]["raw_target_pass"] is False


def test_load_seed_candidate_pool_falls_back_to_audit_identity(tmp_path):
    root = tmp_path / "smoke"
    _write_seed_cohort(root, 42)
    csv_path = root / "seeds" / "seed_42" / "pilot_results.csv"
    rows = list(csv.DictReader(csv_path.open()))
    rows[0]["identity_cosine"] = ""
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit_path = Path(rows[0]["candidate_dir"]) / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["cci"]["metrics"] = {"identity_cosine": 0.91}
    audit_path.write_text(json.dumps(audit))

    candidates = load_seed_candidate_pool(root, (42,), expected_count=2)

    assert candidates[0]["identity_cosine"] == 0.91


def test_load_seed_candidate_pool_rejects_misaligned_cohorts(tmp_path):
    root = tmp_path / "smoke"
    _write_seed_cohort(root, 42, sample_ids=(0, 1))
    _write_seed_cohort(root, 43, sample_ids=(0, 2))

    with pytest.raises(ValueError, match="cohort"):
        load_seed_candidate_pool(root, (42, 43), expected_count=2)


def test_write_reference_manifest_records_exclusion_and_feature_metadata(tmp_path):
    references = ((2, tmp_path / "2.jpg"), (4, tmp_path / "4.jpg"))

    payload = write_reference_manifest(
        tmp_path / "reference_manifest.json",
        references,
        excluded_ids={0, 1},
        dimensions=2048,
        cache_path=tmp_path / "reference_features.npz",
    )

    assert payload["reference_ids"] == [2, 4]
    assert payload["excluded_ids"] == [0, 1]
    assert payload["dimensions"] == 2048
    assert json.loads((tmp_path / "reference_manifest.json").read_text()) == payload


def _materialization_candidates(tmp_path):
    rows = []
    features = []
    for sample_id in (0, 1):
        source = tmp_path / f"source_{sample_id}.png"
        _save_image(source, 20 + sample_id)
        for offset, seed in enumerate((42, 43, 44, 45)):
            output = tmp_path / f"output_{sample_id}_{seed}.png"
            _save_image(output, 40 + sample_id * 4 + offset)
            rows.append(
                {
                    "sample_id": sample_id,
                    "seed": seed,
                    "source_path": str(source),
                    "output_path": str(output),
                    "desired_probability": 0.8,
                    "identity_cosine": 0.9,
                    "outside_semantic_l1": 0.01,
                    "post_attack_selected_epsilon": (
                        "" if seed == 42 else 0.05
                    ),
                    "post_attack_linf": 0.0 if seed == 42 else 0.01,
                    "raw_target_pass": seed == 42,
                }
            )
            features.append([float(offset - 1), float(sample_id)])
    return rows, np.asarray(features)


def test_select_and_materialize_writes_all_four_selector_artifacts(tmp_path):
    rows, candidate_features = _materialization_candidates(tmp_path)
    reference_features = np.array(
        [[-1.0, 0.0], [1.0, 0.0], [-1.0, 1.0], [1.0, 1.0]]
    )
    source_features = np.array([[0.0, 0.0], [0.0, 1.0]])
    output_dir = tmp_path / "results"

    payload = select_and_materialize(
        output_dir=output_dir,
        candidates=rows,
        candidate_activations=candidate_features,
        reference_activations=reference_features,
        source_activations=source_features,
        proxy_dims=2,
        minimum_passes=2,
        selector_seed=20260725,
    )

    assert set(payload["selectors"]) == {"S0", "S1", "S2", "S3"}
    for selector in ("s0", "s1", "s2", "s3"):
        assert (output_dir / f"selection_{selector}.csv").is_file()
        assert len(list((output_dir / "selected" / selector).glob("*.png"))) == 2
    assert (output_dir / "selector_metrics.csv").is_file()
    assert (output_dir / "fid_reranking_report.md").is_file()
    assert len(list((output_dir / "comparisons").glob("*.jpg"))) == 2
    assert payload["selectors"]["S3"]["generation_fr"] == 1.0
    report = (output_dir / "fid_reranking_report.md").read_text().lower()
    assert "exploratory" in report
    assert "2-image" in report
