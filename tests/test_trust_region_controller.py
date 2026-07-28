import pytest


def trust_spec():
    from cci_diff.concept_graph import TrustRegionSpec

    return TrustRegionSpec()


def safety_observations(latents):
    from cci_diff.constraints import ConstraintObservation

    return (
        ConstraintObservation("identity", latents[0].square(), 0.5),
        ConstraintObservation(
            "outside_locality",
            latents[1].square(),
            2.0,
        ),
    )


def test_controller_makes_target_progress_and_reduces_drift():
    import torch

    from cci_diff.constraint_controller import target_margin
    from cci_diff.trust_region_controller import (
        LexicographicTrustRegionController,
    )

    latents = torch.tensor([1.0, 1.0], requires_grad=True)
    target = target_margin(
        latents[0],
        desired_value=0,
        target_probability=0.8,
    )
    drift = latents[1].square()
    controller = LexicographicTrustRegionController(trust_spec())

    result = controller.compute_update(
        latents=latents,
        target=target,
        drift_loss=drift,
        safety_constraints=safety_observations(latents),
        effective_support=torch.ones_like(latents),
    )

    target_gradient = torch.autograd.grad(
        target.loss,
        latents,
        retain_graph=True,
    )[0]
    assert torch.dot(target_gradient, result.clean_delta) < 0
    assert result.clean_delta[1] < 0
    assert result.record["solver"]["requested_target_progress"] > 0
    assert result.record["solver"]["primal_violation"] <= 1e-5


def test_controller_masks_gradients_before_gram_geometry():
    import torch

    from cci_diff.constraint_controller import target_margin
    from cci_diff.trust_region_controller import (
        LexicographicTrustRegionController,
    )

    latents = torch.tensor([1.0, 1.0], requires_grad=True)
    target = target_margin(
        latents.sum(),
        desired_value=0,
        target_probability=0.8,
    )
    result = LexicographicTrustRegionController(trust_spec()).compute_update(
        latents=latents,
        target=target,
        drift_loss=latents[1].square(),
        safety_constraints=safety_observations(latents),
        effective_support=torch.tensor([1.0, 0.0]),
    )

    assert result.clean_delta[1].item() == 0.0
    assert (
        result.record["gradients"]["target"]["masked_norm"]
        < result.record["gradients"]["target"]["raw_norm"]
    )


def test_controller_uses_margin_guard_after_target_is_feasible():
    import torch

    from cci_diff.constraint_controller import target_margin
    from cci_diff.constraints import ConstraintObservation
    from cci_diff.trust_region_controller import (
        LexicographicTrustRegionController,
    )

    latents = torch.tensor([-2.0, 0.5], requires_grad=True)
    target = target_margin(
        latents[0],
        desired_value=0,
        target_probability=0.8,
    )
    safety = (
        ConstraintObservation("identity", latents[0].square() * 0.0, 0.5),
        ConstraintObservation(
            "outside_locality",
            latents[1].square(),
            2.0,
        ),
    )
    result = LexicographicTrustRegionController(trust_spec()).compute_update(
        latents=latents,
        target=target,
        drift_loss=latents[1].square(),
        safety_constraints=safety,
        effective_support=torch.ones_like(latents),
    )

    assert result.record["target"]["guard_mode"] == "maintain"


def test_fixed_trust_matched_uses_same_constraints_but_fixed_nominal():
    import torch

    from cci_diff.constraint_controller import target_margin
    from cci_diff.trust_region_controller import (
        LexicographicTrustRegionController,
    )

    latents = torch.tensor([1.0, 1.0], requires_grad=True)
    result = LexicographicTrustRegionController(trust_spec()).compute_update(
        latents=latents,
        target=target_margin(latents[0], 0, 0.8),
        drift_loss=latents[1].square(),
        safety_constraints=safety_observations(latents),
        effective_support=torch.ones_like(latents),
        mode="fixed_trust_matched",
    )

    assert result.record["solver"]["mode"] == "fixed_trust_matched"


def test_controller_skips_unreliable_zero_target_gradient():
    import torch

    from cci_diff.constraint_controller import target_margin
    from cci_diff.trust_region_controller import (
        LexicographicTrustRegionController,
    )

    latents = torch.tensor([1.0, 1.0], requires_grad=True)
    result = LexicographicTrustRegionController(trust_spec()).compute_update(
        latents=latents,
        target=target_margin(latents[0] * 0.0, 0, 0.8),
        drift_loss=latents[1].square(),
        safety_constraints=safety_observations(latents),
        effective_support=torch.ones_like(latents),
    )

    assert torch.equal(result.clean_delta, torch.zeros_like(latents))
    assert (
        result.record["update"]["skip_reason"]
        == "unreliable_target_gradient"
    )


def test_controller_rejects_wrong_safety_surface_and_rolls_back_nonfinite():
    import torch

    from cci_diff.constraint_controller import target_margin
    from cci_diff.constraints import ConstraintObservation
    from cci_diff.trust_region_controller import (
        LexicographicTrustRegionController,
    )

    latents = torch.tensor([1.0, 1.0], requires_grad=True)
    controller = LexicographicTrustRegionController(trust_spec())
    with pytest.raises(ValueError, match="identity.*outside_locality"):
        controller.compute_update(
            latents=latents,
            target=target_margin(latents[0], 0, 0.8),
            drift_loss=latents[1].square(),
            safety_constraints=(
                ConstraintObservation("identity", latents[0], 0.5),
            ),
            effective_support=torch.ones_like(latents),
        )

    nonfinite = controller.compute_update(
        latents=latents,
        target=target_margin(latents[0] * float("nan"), 0, 0.8),
        drift_loss=latents[1].square(),
        safety_constraints=safety_observations(latents),
        effective_support=torch.ones_like(latents),
    )
    assert torch.equal(nonfinite.clean_delta, torch.zeros_like(latents))
    assert nonfinite.record["update"]["skip_reason"] == "nonfinite"


def test_controller_adapts_radius_from_observed_progress():
    from cci_diff.trust_region_controller import (
        LexicographicTrustRegionController,
    )

    controller = LexicographicTrustRegionController(trust_spec())
    controller.observe_outcome(
        requested_progress=0.1,
        actual_progress=0.01,
        step_norm=controller.radius,
    )
    assert controller.radius == pytest.approx(0.075)

    controller.observe_outcome(
        requested_progress=0.1,
        actual_progress=0.1,
        step_norm=controller.radius,
    )
    assert controller.radius == pytest.approx(0.1125)
