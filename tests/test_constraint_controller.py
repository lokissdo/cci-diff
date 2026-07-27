import math
import unittest


class TestConstraintController(unittest.TestCase):
    def controller_spec(self):
        from cci_diff.concept_graph import concept_graph_from_dict
        from test_concept_graph import valid_graph_payload

        return concept_graph_from_dict(valid_graph_payload()).controller

    def test_margin_has_correct_direction_for_both_binary_targets(self):
        import torch

        from cci_diff.constraint_controller import target_margin

        positive = target_margin(torch.tensor(2.0), 1, 0.8)
        negative = target_margin(torch.tensor(2.0), 0, 0.8)

        self.assertEqual(positive.loss.item(), 0.0)
        self.assertEqual(positive.activation, 0.0)
        self.assertAlmostEqual(negative.signed_logit.item(), -2.0)
        self.assertAlmostEqual(negative.loss.item(), math.log(4.0) + 2.0)
        self.assertGreater(negative.activation, 0.0)

    def test_dual_multiplier_rises_on_violation_and_falls_on_satisfaction(self):
        from cci_diff.constraint_controller import update_dual_multiplier

        spec = self.controller_spec()
        raised, residual = update_dual_multiplier(
            0.1,
            value=0.03,
            tolerance=0.02,
            spec=spec,
        )
        lowered, _ = update_dual_multiplier(
            raised,
            value=0.005,
            tolerance=0.02,
            spec=spec,
        )

        self.assertAlmostEqual(residual, 0.5)
        self.assertGreater(raised, 0.1)
        self.assertLess(lowered, raised)

    def test_projection_removes_only_target_opposing_component(self):
        import torch

        from cci_diff.constraint_controller import project_target_conflict

        target = torch.tensor([1.0, 0.0])
        constraint = torch.tensor([-1.0, 2.0])
        projected, applied, cosine = project_target_conflict(
            target,
            constraint,
            gradient_floor=1e-5,
        )

        self.assertTrue(applied)
        self.assertLess(cosine, 0.0)
        self.assertTrue(torch.allclose(projected, torch.tensor([0.0, 2.0])))
        self.assertGreaterEqual(torch.dot(target, projected).item(), 0.0)

    def test_target_priority_budgets_constraints_by_margin_progress(self):
        import torch

        from cci_diff.constraint_controller import budget_constraint_for_target

        target = torch.tensor([3.0, 4.0])
        constraint = torch.tensor([0.0, 20.0])

        far, far_scale, far_budget = budget_constraint_for_target(
            target,
            constraint,
            target_activation=1.0,
            gradient_floor=1e-5,
        )
        midway, midway_scale, midway_budget = budget_constraint_for_target(
            target,
            constraint,
            target_activation=0.5,
            gradient_floor=1e-5,
        )
        satisfied, satisfied_scale, satisfied_budget = budget_constraint_for_target(
            target,
            constraint,
            target_activation=0.0,
            gradient_floor=1e-5,
        )

        self.assertTrue(torch.equal(far, torch.zeros_like(constraint)))
        self.assertEqual(far_scale, 0.0)
        self.assertEqual(far_budget, 0.0)
        self.assertAlmostEqual(torch.linalg.vector_norm(midway).item(), 2.5)
        self.assertAlmostEqual(midway_scale, 0.125)
        self.assertAlmostEqual(midway_budget, 2.5)
        self.assertTrue(torch.equal(satisfied, constraint))
        self.assertEqual(satisfied_scale, 1.0)
        self.assertIsNone(satisfied_budget)

    def test_trust_clip_bounds_full_masked_update(self):
        import torch

        from cci_diff.constraint_controller import clip_update_norm

        update, before, after = clip_update_norm(
            torch.tensor([3.0, 4.0]),
            trust_radius=0.2,
            gradient_floor=1e-5,
        )
        self.assertAlmostEqual(before, 5.0)
        self.assertAlmostEqual(after, 0.2, places=6)
        self.assertAlmostEqual(
            torch.linalg.vector_norm(update).item(),
            0.2,
            places=6,
        )

    def test_zero_gradient_is_not_inflated_or_made_non_finite(self):
        import torch

        from cci_diff.constraint_controller import normalize_with_ema

        normalized, ema, raw_norm = normalize_with_ema(
            torch.zeros(3),
            previous_ema=0.0,
            beta=0.9,
            floor=1e-5,
        )
        self.assertEqual(raw_norm, 0.0)
        self.assertEqual(ema, 0.0)
        self.assertTrue(torch.equal(normalized, torch.zeros(3)))

    def test_compute_update_uses_target_and_first_step_constraint_feedback(self):
        import torch

        from cci_diff.constraint_controller import (
            ConstraintFeedbackController,
            target_margin,
        )
        from cci_diff.constraints import ConstraintObservation

        latents = torch.tensor([1.0, 1.0], requires_grad=True)
        margin = target_margin(latents[0], desired_value=0, target_probability=0.8)
        observations = (
            ConstraintObservation("locality", latents[1].square(), tolerance=0.5),
        )
        controller = ConstraintFeedbackController(self.controller_spec())
        result = controller.compute_update(
            latents=latents,
            target=margin,
            constraints=observations,
            latent_mask=torch.tensor([1.0, 0.0]),
            eta=0.2,
            project_conflicts=True,
            mode="feedback",
        )

        locality = result.record["constraints"]["locality"]
        self.assertGreater(locality["coefficient"], 0.0)
        self.assertGreater(locality["lambda_after"], 0.0)
        self.assertEqual(result.delta[1].item(), 0.0)
        self.assertLessEqual(
            result.record["update"]["norm"],
            self.controller_spec().trust_radius + 1e-6,
        )

    def test_target_guidance_can_be_disabled_without_disabling_constraints(self):
        import torch

        from cci_diff.constraint_controller import ConstraintFeedbackController, target_margin
        from cci_diff.constraints import ConstraintObservation

        latents = torch.tensor([1.0, 1.0], requires_grad=True)
        controller = ConstraintFeedbackController(
            self.controller_spec(), use_target_guidance=False
        )
        result = controller.compute_update(
            latents=latents,
            target=target_margin(latents[0], 0, 0.8),
            constraints=(ConstraintObservation("locality", latents[1].square(), 0.5),),
            latent_mask=torch.ones_like(latents),
            eta=0.01,
        )

        self.assertEqual(result.delta[0].item(), 0.0)
        self.assertNotEqual(result.delta[1].item(), 0.0)
        self.assertFalse(result.record["target"]["guidance_enabled"])

    def test_gradient_normalization_can_be_disabled(self):
        import torch

        from cci_diff.constraint_controller import ConstraintFeedbackController, target_margin

        latents = torch.tensor([0.25], requires_grad=True)
        controller = ConstraintFeedbackController(
            self.controller_spec(), normalize_gradients=False
        )
        result = controller.compute_update(
            latents=latents,
            target=target_margin(4.0 * latents[0], 0, 0.8),
            constraints=(),
            latent_mask=torch.ones_like(latents),
            eta=0.01,
        )

        self.assertAlmostEqual(
            result.record["target"]["normalized_gradient_norm"],
            result.record["target"]["gradient_norm"],
        )
        self.assertFalse(result.record["update"]["gradient_normalization"])

    def test_target_budget_can_be_disabled_independently(self):
        import torch

        from cci_diff.constraint_controller import ConstraintFeedbackController, target_margin
        from cci_diff.constraints import ConstraintObservation

        latents = torch.tensor([1.0, 1.0], requires_grad=True)
        controller = ConstraintFeedbackController(
            self.controller_spec(), budget_constraints=False
        )
        result = controller.compute_update(
            latents=latents,
            target=target_margin(latents[0], 0, 0.8),
            constraints=(ConstraintObservation("locality", latents[1].square(), 0.5),),
            latent_mask=torch.ones_like(latents),
            eta=0.01,
        )

        self.assertEqual(result.record["update"]["constraint_scale"], 1.0)
        self.assertIsNone(result.record["update"]["constraint_budget"])
        self.assertFalse(result.record["update"]["target_budget"])

    def test_satisfied_target_still_projects_target_opposing_constraints(self):
        import torch

        from cci_diff.constraint_controller import (
            ConstraintFeedbackController,
            target_margin,
        )
        from cci_diff.constraints import ConstraintObservation

        latents = torch.tensor([2.0, 1.0], requires_grad=True)
        margin = target_margin(latents[0], desired_value=1, target_probability=0.8)
        controller = ConstraintFeedbackController(self.controller_spec())
        result = controller.compute_update(
            latents=latents,
            target=margin,
            constraints=(
                ConstraintObservation(
                    "identity",
                    latents[0].square() + latents[1].square(),
                    tolerance=0.5,
                ),
            ),
            latent_mask=torch.ones_like(latents),
            eta=0.01,
            project_conflicts=True,
            mode="feedback",
        )

        self.assertEqual(margin.activation, 0.0)
        self.assertTrue(result.record["update"]["projected"])
        self.assertAlmostEqual(result.delta[0].item(), 0.0, places=6)
        self.assertNotEqual(result.delta[1].item(), 0.0)

    def test_disabled_mode_measures_but_never_updates_direction_or_duals(self):
        import torch

        from cci_diff.constraint_controller import (
            ConstraintFeedbackController,
            target_margin,
        )
        from cci_diff.constraints import ConstraintObservation

        latents = torch.tensor([1.0], requires_grad=True)
        controller = ConstraintFeedbackController(self.controller_spec())
        result = controller.compute_update(
            latents=latents,
            target=target_margin(latents[0], 0, 0.8),
            constraints=(
                ConstraintObservation("locality", latents.square().mean(), 0.5),
            ),
            latent_mask=torch.ones_like(latents),
            eta=0.2,
            mode="disabled",
        )

        self.assertTrue(torch.equal(result.delta, torch.zeros_like(latents)))
        self.assertEqual(controller.multipliers, {})
        self.assertEqual(
            result.record["constraints"]["locality"]["coefficient"],
            0.0,
        )

    def test_fixed_equal_and_projection_disabled_modes_are_explicit(self):
        import torch

        from cci_diff.constraint_controller import (
            ConstraintFeedbackController,
            target_margin,
        )
        from cci_diff.constraints import ConstraintObservation

        latents = torch.tensor([1.0, -1.0], requires_grad=True)
        controller = ConstraintFeedbackController(self.controller_spec())
        result = controller.compute_update(
            latents=latents,
            target=target_margin(latents[0], 0, 0.8),
            constraints=(
                ConstraintObservation("violated", latents[1].square(), 0.5),
                ConstraintObservation("satisfied", latents[1].square() * 0.0, 0.5),
            ),
            latent_mask=torch.ones_like(latents),
            eta=0.01,
            project_conflicts=False,
            mode="fixed_equal",
        )

        self.assertEqual(
            result.record["constraints"]["violated"]["coefficient"],
            1.0,
        )
        self.assertEqual(
            result.record["constraints"]["satisfied"]["coefficient"],
            0.0,
        )
        self.assertFalse(result.record["update"]["projected"])
        self.assertIsNone(result.record["update"]["target_constraint_cosine"])

    def test_two_nonfinite_steps_raise_and_state_is_not_advanced(self):
        import torch

        from cci_diff.constraint_controller import (
            ConstraintFeedbackController,
            target_margin,
        )
        from cci_diff.constraints import ConstraintObservation

        controller = ConstraintFeedbackController(self.controller_spec())
        for attempt in range(2):
            latents = torch.tensor([1.0], requires_grad=True)
            kwargs = dict(
                latents=latents,
                target=target_margin(latents[0], 0, 0.8),
                constraints=(
                    ConstraintObservation("bad", latents[0] * float("nan"), 0.5),
                ),
                latent_mask=torch.ones_like(latents),
                eta=0.1,
            )
            if attempt == 0:
                result = controller.compute_update(**kwargs)
                self.assertEqual(result.record["update"]["skip_reason"], "nonfinite")
                self.assertEqual(controller.multipliers, {})
            else:
                with self.assertRaisesRegex(FloatingPointError, "Two consecutive"):
                    controller.compute_update(**kwargs)

    def test_unreliable_target_gradient_is_reported_after_two_steps(self):
        import torch

        from cci_diff.constraint_controller import (
            ConstraintFeedbackController,
            target_margin,
        )

        controller = ConstraintFeedbackController(self.controller_spec())
        records = []
        for _ in range(2):
            latents = torch.tensor([1.0], requires_grad=True)
            records.append(
                controller.compute_update(
                    latents=latents,
                    target=target_margin(latents[0] * 0.0, 0, 0.8),
                    constraints=(),
                    latent_mask=torch.ones_like(latents),
                    eta=0.1,
                ).record
            )
        self.assertFalse(records[0]["target"]["unreliable_target_gradient"])
        self.assertTrue(records[1]["target"]["unreliable_target_gradient"])


if __name__ == "__main__":
    unittest.main()
