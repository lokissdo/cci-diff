import pytest

from cci_diff.counterfactual_graph import RegionSetEvidence
from cci_diff.generic_region_discovery import (
    BeamSearchConfig,
    advance_beam,
    propose_region_sets,
    shortlist_atomic_components,
)


def screening_rows(regions):
    rows = []
    for index, region in enumerate(regions):
        for sample_id in range(4):
            rows.append(
                {
                    "sample_id": sample_id,
                    "region": region,
                    "captured_mass": 0.8 - index * 0.03,
                    "region_density": 0.9 - index * 0.04,
                    "mask_fraction": 0.01 + index * 0.01,
                    "proposal_score": 1.0 - index * 0.05,
                }
            )
    return rows


def evidence(regions, effect, flip, area, ci=0.01):
    return RegionSetEvidence(
        regions=regions,
        row_count=4,
        sample_count=4,
        flip_rate=flip,
        mean_effect=effect,
        median_effect=effect,
        effect_ci_low=ci,
        effect_ci_high=effect + 0.1,
        mean_mask_fraction=area,
    )


def test_shortlist_is_generic_and_uses_source_scores():
    regions = ["hair", "nose", "mouth", "neck", "eye", "ear", "hat"]

    shortlist = shortlist_atomic_components(
        screening_rows(regions), BeamSearchConfig()
    )

    assert shortlist == tuple(regions[:6])
    assert "hair" in shortlist


def test_each_level_is_deterministic_and_budgeted():
    config = BeamSearchConfig(
        atomic_shortlist_size=6,
        beam_width=4,
        level_evaluation_budget=6,
        max_components=3,
    )
    shortlist = ("a", "b", "c", "d", "e", "f")

    singletons = propose_region_sets(shortlist, (), 1, config)
    pairs = propose_region_sets(
        shortlist, (("a",), ("b",), ("c",), ("d",)), 2, config
    )
    triples = propose_region_sets(
        shortlist,
        (("a", "b"), ("a", "c"), ("b", "c"), ("c", "d")),
        3,
        config,
    )

    assert len(singletons) == len(pairs) == len(triples) == 6
    assert all(len(item) == 1 for item in singletons)
    assert all(len(item) == 2 for item in pairs)
    assert all(len(item) == 3 for item in triples)


def test_advance_beam_prefers_supported_pareto_evidence():
    config = BeamSearchConfig(minimum_samples=4, beam_width=2)
    rows = [
        evidence(("a",), 0.3, 0.75, 0.02),
        evidence(("b",), 0.4, 0.95, 0.08),
        evidence(("c",), 0.1, 0.50, 0.20),
    ]

    beam = advance_beam(rows, config)

    assert beam == (("b",), ("a",))


def test_advance_beam_uses_deterministic_fallback_without_positive_ci():
    config = BeamSearchConfig(minimum_samples=4, beam_width=2)
    rows = [
        evidence(("wide",), 0.4, 0.90, 0.20, ci=-0.1),
        evidence(("small",), 0.2, 0.95, 0.05, ci=-0.1),
    ]

    assert advance_beam(rows, config) == (("small",), ("wide",))


def test_invalid_beam_configuration_is_rejected():
    with pytest.raises(ValueError):
        BeamSearchConfig(max_components=4)
