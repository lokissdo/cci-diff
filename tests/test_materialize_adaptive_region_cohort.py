import csv
import hashlib
import json
from pathlib import Path

import pytest

import scripts.materialize_adaptive_region_cohort as module
from scripts.materialize_adaptive_region_cohort import (
    materialize_adaptive_cohort,
)


MOUTH = ("mouth",)
PERIORAL = ("lower_lip", "mouth", "upper_lip")


def canonical_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def write_selection_manifest(path, decisions):
    unsigned = {
        "version": 1,
        "policy_type": "risk_controlled_source_only_v1",
        "target": "Smiling",
        "desired_value": 0,
        "influence_graph_sha256": "a" * 64,
        "selector_sha256": "b" * 64,
        "feature_signature": "c" * 64,
        "generation_policy_signature": "d" * 64,
        "decisions": [
            {
                "sample_id": sample_id,
                "source_path": f"images/{sample_id}.jpg",
                "source_probability": 0.9,
                "selected_regions": list(regions),
                "selection_uses_generated_output": False,
                "fallback_used": regions == PERIORAL,
            }
            for sample_id, regions in sorted(decisions.items())
        ],
    }
    digest = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    path.write_text(
        json.dumps({**unsigned, "manifest_sha256": digest}, indent=2),
        encoding="utf-8",
    )
    return path


def write_candidate_csv(path, regions, sample_ids=(1, 2), variants=("A0", "A11")):
    rows = []
    for sample_id in sample_ids:
        for variant in variants:
            run_dir = path.parent / "+".join(regions) / str(sample_id) / variant
            run_dir.mkdir(parents=True, exist_ok=True)
            output = run_dir / "sd2_bld_grid.png"
            candidate = run_dir / "candidates/d8/sd2_bld_grid.png"
            candidate.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"post-attack")
            candidate.write_bytes(b"pre-attack")
            rows.append(
                {
                    "feature": "smile",
                    "sample_id": sample_id,
                    "variant": variant,
                    "source_path": f"images/{sample_id}.jpg",
                    "output_path": str(output),
                    "candidate_output_path": str(candidate),
                    "post_attack_escalated": "True",
                    "target_pass": "True",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return path


def candidate_results(tmp_path, *, variants=("A0", "A11")):
    return {
        MOUTH: write_candidate_csv(
            tmp_path / "mouth.csv", MOUTH, variants=variants
        ),
        PERIORAL: write_candidate_csv(
            tmp_path / "perioral.csv", PERIORAL, variants=variants
        ),
    }


def test_materializer_chooses_selected_root_for_both_variants(tmp_path):
    manifest = write_selection_manifest(
        tmp_path / "selections.json", {1: MOUTH, 2: PERIORAL}
    )

    rows = materialize_adaptive_cohort(
        manifest,
        candidate_results(tmp_path),
        tmp_path / "adaptive",
        expected_variants=("A0", "A11"),
        expected_count=2,
    )

    assert [
        (row["sample_id"], row["variant"], json.loads(row["selected_regions"]))
        for row in rows
    ] == [
        ("1", "A0", ["mouth"]),
        ("1", "A11", ["mouth"]),
        ("2", "A0", ["lower_lip", "mouth", "upper_lip"]),
        ("2", "A11", ["lower_lip", "mouth", "upper_lip"]),
    ]
    assert all(Path(row["output_path"]).read_bytes() == b"post-attack" for row in rows)
    assert (tmp_path / "adaptive/adaptive_results.csv").is_file()
    assert (tmp_path / "adaptive/pilot_results.csv").is_file()


def test_materializer_rejects_incomplete_variant_pairs(tmp_path):
    manifest = write_selection_manifest(
        tmp_path / "selections.json", {1: MOUTH, 2: PERIORAL}
    )

    with pytest.raises(ValueError, match="A11"):
        materialize_adaptive_cohort(
            manifest,
            candidate_results(tmp_path, variants=("A0",)),
            tmp_path / "adaptive",
            expected_variants=("A0", "A11"),
            expected_count=2,
        )


def test_materializer_rejects_manifest_changed_after_hashing(tmp_path):
    manifest = write_selection_manifest(
        tmp_path / "selections.json", {1: MOUTH, 2: PERIORAL}
    )
    payload = json.loads(manifest.read_text())
    payload["decisions"][0]["selected_regions"] = list(PERIORAL)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        materialize_adaptive_cohort(
            manifest,
            {MOUTH: tmp_path / "missing.csv", PERIORAL: tmp_path / "missing2.csv"},
            tmp_path / "adaptive",
            expected_count=2,
        )


def test_materializer_rejects_generated_fields_in_selection_decision(tmp_path):
    manifest = write_selection_manifest(
        tmp_path / "selections.json", {1: MOUTH, 2: PERIORAL}
    )
    payload = json.loads(manifest.read_text())
    payload["decisions"][0]["output_path"] = "forbidden.png"
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256")
    payload["manifest_sha256"] = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="generated-output"):
        materialize_adaptive_cohort(
            manifest,
            candidate_results(tmp_path),
            tmp_path / "adaptive",
            expected_count=2,
        )


def test_materializer_has_no_diffusion_or_subprocess_dependency():
    assert not hasattr(module, "subprocess")


def test_real_schema_preserves_post_attack_selected_output(tmp_path):
    manifest = write_selection_manifest(
        tmp_path / "selections.json", {1: MOUTH, 2: PERIORAL}
    )

    rows = materialize_adaptive_cohort(
        manifest,
        candidate_results(tmp_path),
        tmp_path / "adaptive",
        expected_count=2,
    )

    assert all(row["output_path"].endswith("sd2_bld_grid.png") for row in rows)
    assert all(row["post_attack_escalated"] == "True" for row in rows)
    assert all(
        row["candidate_output_path"].endswith(
            "candidates/d8/sd2_bld_grid.png"
        )
        for row in rows
    )
