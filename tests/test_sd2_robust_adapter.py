import unittest


class RecordingClassifier:
    def __init__(self):
        self.calls = 0

    def forward_logits(self, images):
        import torch

        self.calls += 1
        logits = torch.zeros(
            (images.shape[0], 40),
            device=images.device,
            dtype=images.dtype,
        )
        logits[:, 31] = images.mean(dim=(1, 2, 3))
        return logits


class TestSD2RobustAdapter(unittest.TestCase):
    def test_schedule_applies_expected_steps_with_linear_decay(self):
        from cci_diff.adapters.sd2_robust import robust_step_size

        applied = {
            step: robust_step_size(
                step,
                start=4,
                end=16,
                every=2,
                base=0.2,
            )
            for step in range(27)
        }

        self.assertEqual(
            [step for step, value in applied.items() if value is not None],
            [4, 6, 8, 10, 12, 14, 16],
        )
        self.assertAlmostEqual(applied[4], 0.2)
        self.assertGreater(applied[4], applied[16])

    def test_tv_penalizes_checkerboard_more_than_smooth_residual(self):
        import torch

        from cci_diff.adapters.sd2_robust import residual_tv_loss

        source = torch.zeros((1, 3, 4, 4))
        smooth = torch.zeros_like(source)
        checker = torch.tensor(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 4, 4).repeat(1, 3, 1, 1)
        mask = torch.ones((1, 1, 4, 4))

        self.assertGreater(
            residual_tv_loss(checker, source, mask),
            residual_tv_loss(smooth, source, mask),
        )

    def test_boundary_loss_is_zero_for_identical_images(self):
        import torch

        from cci_diff.adapters.sd2_robust import boundary_loss

        image = torch.rand((1, 3, 4, 4))
        boundary = torch.ones((1, 1, 4, 4))

        self.assertEqual(boundary_loss(image, image, boundary).item(), 0.0)

    def test_multi_scale_classifier_loss_uses_every_view_and_backpropagates(self):
        import torch

        from cci_diff.adapters.sd2_robust import multi_scale_classifier_loss

        classifier = RecordingClassifier()
        decoded = torch.rand((1, 3, 16, 16), requires_grad=True)

        loss = multi_scale_classifier_loss(
            classifier,
            decoded,
            label_index=31,
            desired_value=0,
            scales=(8, 12, 16),
            input_size=16,
            blur_sigma=1.0,
        )
        loss.backward()

        self.assertEqual(classifier.calls, 3)
        self.assertIsNotNone(decoded.grad)
        self.assertTrue(torch.isfinite(decoded.grad).all())

    def test_separate_gradient_guidance_handles_zero_term_without_nan(self):
        import torch

        from cci_diff.adapters.sd2_robust import apply_robust_latent_guidance

        latents = torch.tensor([[1.0, 1.0]])
        mask = torch.tensor([[1.0, 0.0]])

        def loss_fn(decoded):
            semantic = decoded.sum()
            zero = decoded.sum() * 0.0
            return {"smile": semantic, "boundary": zero, "tv": zero}

        guided, stats = apply_robust_latent_guidance(
            latents,
            decode_fn=lambda value: value,
            loss_fn=loss_fn,
            weights={"smile": 1.0, "boundary": 0.3, "tv": 0.05},
            step_size=0.2,
            generation_mask=mask,
        )

        self.assertTrue(torch.isfinite(guided).all())
        self.assertLess(guided[0, 0], latents[0, 0])
        self.assertEqual(guided[0, 1], latents[0, 1])
        self.assertEqual(stats["boundary_gradient_norm"], 0.0)


if __name__ == "__main__":
    unittest.main()
