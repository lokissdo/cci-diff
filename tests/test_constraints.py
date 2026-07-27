import unittest


class MeanClassifier:
    def forward_logits(self, images):
        import torch

        logits = torch.zeros(
            (images.shape[0], 40),
            device=images.device,
            dtype=images.dtype,
        )
        logits[:, 31] = images.mean(dim=(1, 2, 3))
        logits[:, 21] = images[:, :, :1, :].mean(dim=(1, 2, 3))
        return logits


class TestConstraintEvaluators(unittest.TestCase):
    def test_target_returns_differentiable_mean_logit(self):
        import torch

        from cci_diff.constraints import CelebAAttributeTarget

        image = torch.full((1, 3, 4, 4), 0.5, requires_grad=True)
        target = CelebAAttributeTarget(
            MeanClassifier(),
            attribute_index=31,
            input_size=4,
        )
        logit = target.logit(image)
        logit.backward()

        self.assertAlmostEqual(logit.item(), 0.2265695, places=5)
        self.assertIsNotNone(image.grad)

    def test_outside_l1_ignores_changes_inside_generation_mask(self):
        import torch

        from cci_diff.constraints import ConstraintContext, OutsideL1Constraint

        source = torch.zeros((1, 3, 2, 2))
        generation_mask = torch.zeros((1, 1, 2, 2))
        generation_mask[:, :, 0, 0] = 1.0
        context = ConstraintContext(source, generation_mask, generation_mask)
        evaluator = OutsideL1Constraint("outside_locality", tolerance=0.02)
        evaluator.bind(context)
        inside_change = source.clone()
        inside_change[:, :, 0, 0] = 1.0
        outside_change = source.clone()
        outside_change[:, :, 1, 1] = 1.0

        self.assertEqual(evaluator.measure(inside_change).item(), 0.0)
        self.assertGreater(evaluator.measure(outside_change).item(), 0.0)

    def test_attribute_constraint_measures_source_probability_drift(self):
        import torch

        from cci_diff.constraints import (
            CelebAAttributeConstraint,
            ConstraintContext,
        )

        source = torch.zeros((1, 3, 4, 4))
        mask = torch.ones((1, 1, 4, 4))
        evaluator = CelebAAttributeConstraint(
            "mouth_open",
            MeanClassifier(),
            21,
            input_size=4,
            tolerance=0.1,
        )
        evaluator.bind(ConstraintContext(source, mask, mask))

        self.assertEqual(evaluator.measure(source).item(), 0.0)
        self.assertGreater(evaluator.measure(torch.ones_like(source)).item(), 0.0)

    def test_masked_residual_tv_penalizes_non_smooth_edit_inside_mask(self):
        import torch

        from cci_diff.constraints import (
            ConstraintContext,
            MaskedResidualTVConstraint,
        )

        source = torch.zeros((1, 3, 3, 3))
        semantic = torch.ones((1, 1, 3, 3))
        evaluator = MaskedResidualTVConstraint("residual_tv", tolerance=0.015)
        evaluator.bind(ConstraintContext(source, semantic, semantic))
        checker = source.clone()
        checker[:, :, 1, 1] = 1.0

        self.assertEqual(evaluator.measure(source).item(), 0.0)
        self.assertGreater(evaluator.measure(checker).item(), 0.0)

    def test_constraints_require_binding_and_nonempty_support(self):
        import torch

        from cci_diff.constraints import (
            ConstraintContext,
            MaskedResidualTVConstraint,
            OutsideL1Constraint,
        )

        image = torch.zeros((1, 3, 2, 2))
        with self.assertRaisesRegex(RuntimeError, "not bound"):
            OutsideL1Constraint("outside_locality", 0.1).measure(image)
        with self.assertRaisesRegex(ValueError, "outside-mask pixel"):
            OutsideL1Constraint("outside_locality", 0.1).bind(
                ConstraintContext(image, torch.ones((1, 1, 2, 2)), image[:, :1])
            )
        with self.assertRaisesRegex(ValueError, "non-empty semantic mask"):
            MaskedResidualTVConstraint("residual_tv", 0.1).bind(
                ConstraintContext(
                    image,
                    torch.zeros((1, 1, 2, 2)),
                    torch.zeros((1, 1, 2, 2)),
                )
            )

    def test_frozen_evaluator_passes_gradient_only_to_image(self):
        import torch

        from cci_diff.constraints import CelebAAttributeTarget

        class FrozenClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(
                    torch.tensor(1.0),
                    requires_grad=False,
                )

            def forward_logits(self, images):
                logits = torch.zeros(
                    (images.shape[0], 40),
                    device=images.device,
                    dtype=images.dtype,
                )
                return logits + images.mean() * self.scale

        model = FrozenClassifier()
        image = torch.ones((1, 3, 2, 2), requires_grad=True)
        CelebAAttributeTarget(model, 31, 2).logit(image).backward()

        self.assertIsNotNone(image.grad)
        self.assertIsNone(model.scale.grad)


if __name__ == "__main__":
    unittest.main()
