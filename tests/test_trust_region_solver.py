import pytest


def test_projection_keeps_nominal_when_feasible():
    from cci_diff.trust_region_solver import project_to_linear_constraints

    result = project_to_linear_constraints(
        nominal=(-0.1, 0.0),
        gram=((1.0, 0.0), (0.0, 1.0)),
        bounds=(-0.05,),
        radius=0.2,
        tolerance=1e-8,
    )

    assert result is not None
    assert result.step == pytest.approx((-0.1, 0.0))


def test_projection_finds_minimum_norm_target_step():
    from cci_diff.trust_region_solver import project_to_linear_constraints

    result = project_to_linear_constraints(
        nominal=(0.0, 0.0),
        gram=((1.0, 0.0), (0.0, 1.0)),
        bounds=(-0.1,),
        radius=0.2,
        tolerance=1e-8,
    )

    assert result is not None
    assert result.step == pytest.approx((-0.1, 0.0))
    assert result.active_indices == (0,)


def test_projection_returns_none_when_target_exceeds_radius():
    from cci_diff.trust_region_solver import project_to_linear_constraints

    assert (
        project_to_linear_constraints(
            nominal=(0.0,),
            gram=((1.0,),),
            bounds=(-0.3,),
            radius=0.2,
            tolerance=1e-8,
        )
        is None
    )


def test_lexicographic_envelope_minimizes_worst_safety_residual():
    from cci_diff.trust_region_solver import solve_lexicographic_envelope

    tau, result = solve_lexicographic_envelope(
        gram=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        target_bound=-0.1,
        safety_residuals=(0.2, 0.1),
        radius=0.3,
        tolerance=1e-7,
    )

    assert result is not None
    assert tau < 0.2
    assert result.norm <= 0.3 + 1e-7


def test_collinear_singular_gradients_are_regularized_deterministically():
    from cci_diff.trust_region_solver import project_to_linear_constraints

    arguments = {
        "nominal": (0.0, 0.0),
        "gram": ((1.0, 1.0), (1.0, 1.0)),
        "bounds": (-0.1, -0.1),
        "radius": 0.2,
        "tolerance": 1e-7,
    }
    first = project_to_linear_constraints(**arguments)
    second = project_to_linear_constraints(**arguments)

    assert first is not None
    assert second is not None
    assert first.step == pytest.approx(second.step)
    assert first.active_indices == second.active_indices
    assert first.primal_violation <= 1e-7


def test_projection_reports_primal_residual():
    from cci_diff.trust_region_solver import project_to_linear_constraints

    result = project_to_linear_constraints(
        nominal=(0.0, 0.0),
        gram=((1.0, 0.0), (0.0, 1.0)),
        bounds=(-0.1,),
        radius=0.1,
        tolerance=1e-7,
    )

    assert result is not None
    assert result.primal_violation <= 1e-7
    assert result.dual_violation <= 1e-7
