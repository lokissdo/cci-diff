import pytest


def point(
    variant,
    effort,
    *,
    success,
    drift,
    identity,
    locality,
    cluster="cluster",
    source="source",
    seed=42,
):
    return {
        "identity_cluster": cluster,
        "source_id": source,
        "seed": seed,
        "variant": variant,
        "effort": effort,
        "target_success": success,
        "independent_non_target_drift": drift,
        "identity_cosine": identity,
        "outside_semantic_l1": locality,
    }


def test_calibration_frontier_drops_dominated_and_unsafe_points():
    from cci_diff.matched_success import calibration_frontier

    rows = [
        point(
            "A10",
            "low",
            success=0.40,
            drift=0.08,
            identity=0.95,
            locality=0.01,
        ),
        point(
            "A10",
            "mid",
            success=0.60,
            drift=0.09,
            identity=0.95,
            locality=0.01,
        ),
        point(
            "A10",
            "bad",
            success=0.60,
            drift=0.11,
            identity=0.95,
            locality=0.01,
        ),
        point(
            "A10",
            "unsafe",
            success=0.80,
            drift=0.07,
            identity=0.85,
            locality=0.01,
        ),
    ]

    frontier = calibration_frontier(
        rows,
        identity_floor=0.90,
        locality_ceiling=0.02,
    )

    assert [item.effort for item in frontier] == ["low", "mid"]


def test_common_grid_and_interpolation_are_frozen_from_calibration():
    from cci_diff.matched_success import (
        FrontierPoint,
        freeze_common_operating_points,
    )

    fixed = [
        FrontierPoint("low", 0.40, 0.08),
        FrontierPoint("high", 0.70, 0.11),
    ]
    adaptive = [
        FrontierPoint("low", 0.45, 0.06),
        FrontierPoint("high", 0.75, 0.08),
    ]

    frozen = freeze_common_operating_points(
        {"A10": fixed, "A11": adaptive},
        step=0.05,
    )

    assert frozen.grid[0] == pytest.approx(0.45)
    assert frozen.grid[-1] == pytest.approx(0.70)
    assert sum(frozen.weights["A11"][0.70].values()) == pytest.approx(1.0)


def test_normalized_auc_uses_trapezoids():
    from cci_diff.matched_success import normalized_auc

    assert normalized_auc([(0.4, 0.10), (0.6, 0.06)]) == pytest.approx(
        0.08
    )


def test_paired_cluster_bootstrap_is_deterministic_and_favors_adaptive():
    from cci_diff.matched_success import (
        FrontierPoint,
        freeze_common_operating_points,
        paired_cluster_bootstrap,
    )

    frozen = freeze_common_operating_points(
        {
            "A10": [
                FrontierPoint("low", 0.5, 0.10),
                FrontierPoint("high", 0.7, 0.12),
            ],
            "A11": [
                FrontierPoint("low", 0.5, 0.07),
                FrontierPoint("high", 0.7, 0.08),
            ],
        },
        step=0.2,
    )
    rows = []
    for index in range(4):
        for variant, low_drift, high_drift in (
            ("A10", 0.10, 0.12),
            ("A11", 0.07, 0.08),
        ):
            rows.extend(
                [
                    point(
                        variant,
                        "low",
                        success=0.5,
                        drift=low_drift,
                        identity=0.96,
                        locality=0.01,
                        cluster=f"c{index}",
                        source=f"s{index}",
                    ),
                    point(
                        variant,
                        "high",
                        success=0.7,
                        drift=high_drift,
                        identity=0.96,
                        locality=0.01,
                        cluster=f"c{index}",
                        source=f"s{index}",
                    ),
                ]
            )

    first = paired_cluster_bootstrap(
        rows,
        frozen,
        fixed_variant="A10",
        adaptive_variant="A11",
        seed=9,
        samples=200,
    )
    second = paired_cluster_bootstrap(
        rows,
        frozen,
        fixed_variant="A10",
        adaptive_variant="A11",
        seed=9,
        samples=200,
    )

    assert first == second
    assert first["drift"]["estimate"] < 0
    assert set(first["drift"]) == {"estimate", "low", "high"}
