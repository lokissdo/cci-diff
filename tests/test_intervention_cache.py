from dataclasses import replace

from cci_diff.counterfactual_graph import InterventionObservation
from cci_diff.intervention_cache import (
    cache_key_for,
    load_cached_intervention,
    store_cached_intervention,
)


def valid_inputs():
    return {
        "source_sha256": "a" * 64,
        "mask_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "classifier_sha256": "d" * 64,
        "graph_sha256": "e" * 64,
        "policy_sha256": "f" * 64,
        "seed": 42,
    }


def observation():
    return InterventionObservation(
        target="Smiling",
        desired_value=0,
        sample_id=7,
        seed=42,
        regions=("mouth",),
        source_probability=0.9,
        output_probability=0.4,
        mask_fraction=0.03,
    )


def test_cache_key_changes_for_every_scientific_input():
    base = cache_key_for(**valid_inputs())

    for field in valid_inputs():
        changed = dict(valid_inputs())
        changed[field] = 43 if field == "seed" else "0" * 64
        assert cache_key_for(**changed) != base


def test_atomic_cache_round_trip_rejects_partial_entry(tmp_path):
    artifact = tmp_path / "generated.png"
    artifact.write_bytes(b"image")
    key = cache_key_for(**valid_inputs())

    entry = store_cached_intervention(
        tmp_path / "cache",
        key,
        observation(),
        {"output": artifact},
    )

    loaded = load_cached_intervention(tmp_path / "cache", key)
    assert loaded == entry
    assert loaded.artifacts["output"].read_bytes() == b"image"
    (entry.path / "complete.json").unlink()
    assert load_cached_intervention(tmp_path / "cache", key) is None


def test_complete_entries_are_immutable(tmp_path):
    artifact = tmp_path / "generated.png"
    artifact.write_bytes(b"image")
    key = cache_key_for(**valid_inputs())
    store_cached_intervention(
        tmp_path / "cache", key, observation(), {"output": artifact}
    )

    changed = replace(observation(), output_probability=0.2)
    try:
        store_cached_intervention(
            tmp_path / "cache", key, changed, {"output": artifact}
        )
    except ValueError as exc:
        assert "immutable" in str(exc)
    else:
        raise AssertionError("conflicting complete cache entry was overwritten")
