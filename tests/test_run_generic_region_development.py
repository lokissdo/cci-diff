import json
from argparse import Namespace
from pathlib import Path

from cci_diff.counterfactual_graph import InterventionObservation
from scripts.run_generic_region_development import run_development


class FakeBackend:
    def __init__(self, root):
        self.root = Path(root)
        self.calls = []

    def screen(self, *, sample_ids, regions):
        self.calls.append(("screen", sample_ids, regions))
        return [
            {
                "sample_id": sample_id,
                "region": region,
                "captured_mass": 0.9 - index * 0.02,
                "region_density": 0.8 - index * 0.02,
                "mask_fraction": 0.01 + index * 0.01,
                "proposal_score": 0.9 - index * 0.02,
            }
            for sample_id in sample_ids
            for index, region in enumerate(regions)
        ]

    def intervene(self, *, sample_ids, region_sets):
        self.calls.append(("intervene", sample_ids, region_sets))
        return tuple(
            InterventionObservation(
                target="Smiling",
                desired_value=0,
                sample_id=sample_id,
                seed=42,
                regions=regions,
                source_probability=0.9,
                output_probability=0.4 - 0.02 * len(regions),
                mask_fraction=0.02 * len(regions),
                identity_cosine=0.95,
                outside_l1=0.01,
            )
            for regions in region_sets
            for sample_id in sample_ids
        )

    def extract_features(self, *, sample_ids, graph_path):
        self.calls.append(("extract_features", sample_ids, graph_path))
        path = self.root / "source_features.csv"
        path.write_text("sample_id,regions\n", encoding="utf-8")
        return path

    def fit(self, *, graph_path, source_features, development_outcomes, split_manifest):
        self.calls.append(("fit", graph_path))
        path = self.root / "selector_model.json"
        path.write_text(json.dumps({"frozen": True}), encoding="utf-8")
        return path


def args_for(tmp_path, data_size=30):
    return Namespace(
        data_size=data_size,
        seed=42,
        eligible_ids=tuple(range(10000)),
        evaluation_ids=(20000, 20001),
        output_dir=str(tmp_path / f"run-{data_size}"),
        required_flip_rate=0.95,
        bootstrap_samples=20,
        confidence=0.95,
        policy_signature="a" * 64,
    )


def test_data_size_30_runs_one_parameterized_workflow(tmp_path):
    backend = FakeBackend(tmp_path)

    result = run_development(args_for(tmp_path, data_size=30), backend=backend)

    assert result["counts"] == {
        "discovery": 4,
        "fit": 10,
        "calibration": 16,
    }
    assert result["max_components"] == 3
    assert result["variant"] == "A11"
    assert result["evaluation_overlap"] == []
    assert result["special_mode"] is None
    assert result["phase"] == "complete"
    assert (tmp_path / "run-30/development_run.json").is_file()


def test_completed_run_resumes_without_backend_calls(tmp_path):
    args = args_for(tmp_path)
    first = FakeBackend(tmp_path)
    expected = run_development(args, backend=first)
    second = FakeBackend(tmp_path)

    resumed = run_development(args, backend=second)

    assert resumed == expected
    assert second.calls == []


def test_changed_policy_rejects_existing_run(tmp_path):
    args = args_for(tmp_path)
    run_development(args, backend=FakeBackend(tmp_path))
    args.policy_signature = "b" * 64

    try:
        run_development(args, backend=FakeBackend(tmp_path))
    except ValueError as exc:
        assert "configuration" in str(exc)
    else:
        raise AssertionError("changed policy reused an incompatible run")


def test_interrupted_run_reuses_completed_phase_artifacts(tmp_path):
    args = args_for(tmp_path)
    run_development(args, backend=FakeBackend(tmp_path))
    manifest_path = Path(args.output_dir) / "development_run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["phase"] = "source_features"
    del manifest["artifacts"]["selector_model"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    resumed_backend = FakeBackend(tmp_path)

    result = run_development(args, backend=resumed_backend)

    assert result["phase"] == "complete"
    assert [call[0] for call in resumed_backend.calls] == ["fit"]


def test_completed_run_rejects_changed_checkpoint_bytes(tmp_path):
    args = args_for(tmp_path)
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"first")
    args.model_path = str(checkpoint)
    run_development(args, backend=FakeBackend(tmp_path))
    checkpoint.write_bytes(b"second")

    try:
        run_development(args, backend=FakeBackend(tmp_path))
    except ValueError as exc:
        assert "configuration" in str(exc)
    else:
        raise AssertionError("changed checkpoint reused a completed run")
