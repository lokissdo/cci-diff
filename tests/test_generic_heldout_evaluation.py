import hashlib
import json
from argparse import Namespace
from pathlib import Path

import scripts.run_individual_region_cci as heldout


class RecordingBackend:
    def __init__(self):
        self.calls = []
        self.manifest_existed = []

    def generate(self, *, decision, selection_manifest, args):
        self.manifest_existed.append(Path(selection_manifest).is_file())
        self.calls.append((decision["sample_id"], "A11"))
        return {
            "sample_id": decision["sample_id"],
            "variant": "A11",
            "regions": decision["selected_regions"],
        }


def canonical_digest(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def args_for_three_sources(tmp_path):
    for name in ("selector.json", "policy.json", "masks.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    return Namespace(
        sample_ids=[1, 2, 3],
        output_dir=str(tmp_path / "out"),
        selector_model=str(tmp_path / "selector.json"),
        generation_policy_manifest=str(tmp_path / "policy.json"),
        semantic_mask_manifest=str(tmp_path / "masks.json"),
        continue_on_error=False,
        selection_manifest=None,
    )


def test_all_decisions_are_frozen_before_one_a11_call_per_source(
    tmp_path, monkeypatch
):
    args = args_for_three_sources(tmp_path)

    def freeze(phase_args):
        decisions = [
            {
                "sample_id": sample_id,
                "selected_regions": ["mouth"],
                "selection_uses_generated_output": False,
            }
            for sample_id in phase_args.sample_ids
        ]
        unsigned = {
            "version": 1,
            "policy_type": "risk_controlled_source_only_v1",
            "target": "Smiling",
            "desired_value": 0,
            "influence_graph_sha256": "a" * 64,
            "selector_sha256": "b" * 64,
            "feature_signature": "c" * 64,
            "generation_policy_signature": "d" * 64,
            "decisions": decisions,
        }
        payload = {**unsigned, "manifest_sha256": canonical_digest(unsigned)}
        path = Path(phase_args.selection_manifest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return {"selection_manifest_path": str(path)}

    monkeypatch.setattr(heldout, "run_individual_cci", freeze)
    backend = RecordingBackend()

    result = heldout.run_heldout_a11(args, generation_backend=backend)

    assert result["sample_ids"] == [1, 2, 3]
    assert backend.calls == [(1, "A11"), (2, "A11"), (3, "A11")]
    assert all(backend.manifest_existed)
    assert result["variant"] == "A11"
    assert result["generation_count"] == 3


def test_heldout_rejects_generated_or_oracle_decision_fields(
    tmp_path, monkeypatch
):
    args = args_for_three_sources(tmp_path)

    def freeze(phase_args):
        decisions = [
            {
                "sample_id": sample_id,
                "selected_regions": ["mouth"],
                "selection_uses_generated_output": False,
                "oracle_score": 0.9,
            }
            for sample_id in phase_args.sample_ids
        ]
        unsigned = {
            "version": 1,
            "policy_type": "risk_controlled_source_only_v1",
            "target": "Smiling",
            "desired_value": 0,
            "influence_graph_sha256": "a" * 64,
            "selector_sha256": "b" * 64,
            "feature_signature": "c" * 64,
            "generation_policy_signature": "d" * 64,
            "decisions": decisions,
        }
        payload = {**unsigned, "manifest_sha256": canonical_digest(unsigned)}
        path = Path(phase_args.selection_manifest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return {"selection_manifest_path": str(path)}

    monkeypatch.setattr(heldout, "run_individual_cci", freeze)

    try:
        heldout.run_heldout_a11(args, generation_backend=RecordingBackend())
    except ValueError as exc:
        assert "evaluation-only" in str(exc)
    else:
        raise AssertionError("oracle field was accepted in a source decision")
