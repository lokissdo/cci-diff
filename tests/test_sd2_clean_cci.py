import unittest
from types import SimpleNamespace


class TestSD2CleanPrediction(unittest.TestCase):
    def controller_spec(self):
        from cci_diff.concept_graph import ControllerSpec

        return ControllerSpec(
            0.2,
            0.5,
            4.0,
            0.2,
            0.15,
            0.9,
            1e-5,
            (0.15, 0.65),
            2,
        )

    def test_epsilon_prediction_matches_ddim_equation(self):
        import torch

        from cci_diff.adapters.sd2_clean_cci import predict_clean_latents

        sample = torch.tensor([2.0], dtype=torch.float32)
        epsilon = torch.tensor([1.0], dtype=torch.float32)
        result = predict_clean_latents(
            sample,
            epsilon,
            torch.tensor(0.64),
            "epsilon",
        )

        self.assertTrue(torch.allclose(result, torch.tensor([1.75])))

    def test_velocity_prediction_matches_diffusers_equation(self):
        import torch

        from cci_diff.adapters.sd2_clean_cci import predict_clean_latents

        sample = torch.tensor([2.0], dtype=torch.float32)
        velocity = torch.tensor([1.0], dtype=torch.float32)
        result = predict_clean_latents(
            sample,
            velocity,
            torch.tensor(0.64),
            "v_prediction",
        )

        self.assertTrue(torch.allclose(result, torch.tensor([1.0])))

    def test_sample_and_unknown_prediction_types_are_rejected(self):
        import torch

        from cci_diff.adapters.sd2_clean_cci import predict_clean_latents

        for prediction_type in ("sample", "mystery"):
            with self.subTest(prediction_type=prediction_type):
                with self.assertRaisesRegex(ValueError, prediction_type):
                    predict_clean_latents(
                        torch.ones(1),
                        torch.ones(1),
                        torch.tensor(0.5),
                        prediction_type,
                    )

    def test_guidance_eta_uses_normalized_bell_gate(self):
        from cci_diff.adapters.sd2_clean_cci import guidance_eta
        from cci_diff.concept_graph import concept_graph_from_dict
        from test_concept_graph import valid_graph_payload

        spec = concept_graph_from_dict(valid_graph_payload()).controller
        self.assertIsNone(guidance_eta(0, 0.10, spec))
        self.assertAlmostEqual(guidance_eta(2, 0.40, spec), 0.2)
        self.assertIsNone(guidance_eta(3, 0.40, spec))
        self.assertAlmostEqual(guidance_eta(4, 0.15, spec), 0.0)
        self.assertAlmostEqual(guidance_eta(4, 0.65, spec), 0.0)

    def test_guidance_eta_can_disable_interval_and_bell_schedule(self):
        from cci_diff.adapters.sd2_clean_cci import guidance_eta
        from cci_diff.concept_graph import concept_graph_from_dict
        from test_concept_graph import valid_graph_payload

        spec = concept_graph_from_dict(valid_graph_payload()).controller
        self.assertAlmostEqual(guidance_eta(0, 0.0, spec, scheduled=False), 0.2)
        self.assertIsNone(guidance_eta(1, 0.5, spec, scheduled=False))
        self.assertAlmostEqual(guidance_eta(2, 1.0, spec, scheduled=False), 0.2)

    def test_alpha_lookup_preserves_sample_dtype(self):
        import torch

        from cci_diff.adapters.sd2_clean_cci import alpha_prod_for_step

        scheduler = SimpleNamespace(
            alphas_cumprod=torch.tensor([0.9, 0.8], dtype=torch.float64)
        )
        sample = torch.ones(1, dtype=torch.float32)
        alpha = alpha_prod_for_step(scheduler, torch.tensor(1), sample)

        self.assertEqual(alpha.dtype, torch.float32)
        self.assertEqual(alpha.device, sample.device)
        self.assertAlmostEqual(alpha.item(), 0.8)

    def test_clean_prediction_backpropagates_only_to_sample(self):
        import torch

        from cci_diff.adapters.sd2_clean_cci import predict_clean_latents

        sample = torch.tensor([1.0], requires_grad=True)
        model_output = torch.tensor([0.25], requires_grad=True)
        clean = predict_clean_latents(
            sample,
            model_output,
            torch.tensor(0.81),
            "epsilon",
        )
        clean.sum().backward()

        self.assertIsNotNone(sample.grad)
        self.assertIsNone(model_output.grad)

    def test_decode_clean_latents_is_differentiable_and_maps_to_unit_range(self):
        import torch

        from cci_diff.adapters.sd2_clean_cci import decode_clean_latents

        class FakeVAE:
            def decode(self, latents):
                return SimpleNamespace(sample=latents)

        latents = torch.tensor([[[[-0.1, 0.1]]]], requires_grad=True)
        decoded = decode_clean_latents(FakeVAE(), latents, latent_scale=1.0)
        decoded.sum().backward()

        self.assertTrue(torch.all((decoded >= 0.0) & (decoded <= 1.0)))
        self.assertIsNotNone(latents.grad)

    def test_clean_hook_changes_noise_traces_target_and_detaches_unet_output(self):
        import tempfile
        from pathlib import Path

        import torch

        from cci_diff.adapters.sd2_clean_cci import CleanCCIGuidanceHook
        from cci_diff.cci_trace import JSONLTraceWriter, load_cci_trace
        from cci_diff.constraint_controller import ConstraintFeedbackController
        from cci_diff.constraints import ConstraintContext
        from cci_diff.sd2_bld_backend import SD2DenoisingStep

        class Scheduler:
            alphas_cumprod = torch.linspace(0.01, 0.99, 1000)
            config = SimpleNamespace(prediction_type="epsilon")

        class VAE:
            def decode(self, latent):
                return SimpleNamespace(sample=latent[:, :3] * 0.02)

        class Target:
            def logit(self, image):
                return image.mean()

        class Locality:
            name = "locality"
            tolerance = 0.02

            def bind(self, context: ConstraintContext):
                self.source = context.source_image

            def measure(self, image):
                return (image - self.source).abs().mean()

        latents = torch.ones((1, 4, 2, 2))
        noise = torch.zeros_like(latents, requires_grad=True)
        mask = torch.ones((1, 1, 2, 2))
        step = SD2DenoisingStep(
            4,
            torch.tensor(500),
            "neutral expression",
            latents,
            noise,
            torch.zeros_like(latents),
            mask,
            mask,
            11,
            0.4,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            trace = Path(tmpdir) / "trace.jsonl"
            hook = CleanCCIGuidanceHook(
                scheduler=Scheduler(),
                vae=VAE(),
                target_evaluator=Target(),
                constraint_evaluators=(Locality(),),
                controller=ConstraintFeedbackController(self.controller_spec()),
                desired_value=0,
                target_probability=0.8,
                trace_writer=JSONLTraceWriter(trace),
            )
            guided = hook(step)
            records = load_cci_trace(trace)
            final = hook.evaluate_image(torch.zeros((1, 3, 2, 2)))

        self.assertFalse(torch.equal(guided, noise))
        self.assertIsNone(noise.grad)
        self.assertEqual(records[0]["prediction_type"], "epsilon")
        self.assertIn("locality", records[0]["constraints"])
        self.assertIn("post_update_probability", records[0]["target"])
        self.assertGreaterEqual(
            records[0]["target"]["post_update_probability"],
            records[0]["target"]["target_probability"],
        )
        self.assertFalse(final["feasible"])
        self.assertIn("locality", final["failed_constraints"])

    def test_clean_hook_observes_detached_predicted_clean_frames(self):
        import tempfile
        from pathlib import Path

        import torch

        from cci_diff.adapters.sd2_clean_cci import CleanCCIGuidanceHook
        from cci_diff.cci_trace import JSONLTraceWriter
        from cci_diff.constraint_controller import ConstraintFeedbackController
        from cci_diff.sd2_bld_backend import SD2DenoisingStep

        class Scheduler:
            alphas_cumprod = torch.linspace(0.01, 0.99, 1000)
            config = SimpleNamespace(prediction_type="epsilon")

        class VAE:
            def decode(self, latent):
                return SimpleNamespace(sample=latent[:, :3] * 0.02)

        class Target:
            def logit(self, image):
                return image.mean()

        latents = torch.ones((1, 4, 2, 2))
        mask = torch.ones((1, 1, 2, 2))
        step = SD2DenoisingStep(
            4,
            torch.tensor(500),
            "neutral expression",
            latents,
            torch.zeros_like(latents),
            torch.zeros_like(latents),
            mask,
            mask,
            11,
            0.4,
        )
        frames = []
        with tempfile.TemporaryDirectory() as tmpdir:
            hook = CleanCCIGuidanceHook(
                scheduler=Scheduler(),
                vae=VAE(),
                target_evaluator=Target(),
                constraint_evaluators=(),
                controller=ConstraintFeedbackController(self.controller_spec()),
                desired_value=0,
                target_probability=0.8,
                trace_writer=JSONLTraceWriter(Path(tmpdir) / "trace.jsonl"),
                frame_observer=frames.append,
            )
            hook(step)

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["step"], 4)
        self.assertEqual(frames[0]["timestep"], 500)
        self.assertAlmostEqual(frames[0]["progress"], 0.4)
        self.assertFalse(frames[0]["before_image"].requires_grad)
        self.assertFalse(frames[0]["after_image"].requires_grad)

    def test_clean_hook_rejects_direct_sample_before_writing_trace(self):
        import tempfile
        from pathlib import Path

        import torch

        from cci_diff.adapters.sd2_clean_cci import CleanCCIGuidanceHook
        from cci_diff.cci_trace import JSONLTraceWriter
        from cci_diff.constraint_controller import ConstraintFeedbackController
        from cci_diff.sd2_bld_backend import SD2DenoisingStep

        class Scheduler:
            alphas_cumprod = torch.ones(10) * 0.5
            config = SimpleNamespace(prediction_type="sample")

        class VAE:
            def decode(self, latent):
                return SimpleNamespace(sample=latent[:, :3] * 0.02)

        class Target:
            def logit(self, image):
                return image.mean()

        latents = torch.ones((1, 4, 2, 2))
        mask = torch.ones((1, 1, 2, 2))
        step = SD2DenoisingStep(
            4,
            torch.tensor(5),
            "neutral",
            latents,
            torch.zeros_like(latents),
            torch.zeros_like(latents),
            mask,
            mask,
            11,
            0.4,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            trace = Path(tmpdir) / "trace.jsonl"
            hook = CleanCCIGuidanceHook(
                scheduler=Scheduler(),
                vae=VAE(),
                target_evaluator=Target(),
                constraint_evaluators=(),
                controller=ConstraintFeedbackController(self.controller_spec()),
                desired_value=0,
                target_probability=0.8,
                trace_writer=JSONLTraceWriter(trace),
            )
            with self.assertRaisesRegex(ValueError, "sample"):
                hook(step)
            self.assertEqual(trace.read_text(encoding="utf-8"), "")

    def test_clean_hook_skips_first_nonfinite_step_and_raises_on_second(self):
        import tempfile
        from pathlib import Path

        import torch

        from cci_diff.adapters.sd2_clean_cci import CleanCCIGuidanceHook
        from cci_diff.cci_trace import JSONLTraceWriter
        from cci_diff.constraint_controller import ConstraintFeedbackController
        from cci_diff.sd2_bld_backend import SD2DenoisingStep

        class Scheduler:
            alphas_cumprod = torch.ones(10) * 0.5
            config = SimpleNamespace(prediction_type="epsilon")

        class VAE:
            def decode(self, latent):
                return SimpleNamespace(sample=latent[:, :3] * 0.02)

        class NaNTarget:
            def logit(self, image):
                return image.mean() * float("nan")

        latents = torch.ones((1, 4, 2, 2))
        noise = torch.zeros_like(latents)
        mask = torch.ones((1, 1, 2, 2))
        step = SD2DenoisingStep(
            4,
            torch.tensor(5),
            "neutral",
            latents,
            noise,
            torch.zeros_like(latents),
            mask,
            mask,
            11,
            0.4,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            hook = CleanCCIGuidanceHook(
                scheduler=Scheduler(),
                vae=VAE(),
                target_evaluator=NaNTarget(),
                constraint_evaluators=(),
                controller=ConstraintFeedbackController(self.controller_spec()),
                desired_value=0,
                target_probability=0.8,
                trace_writer=JSONLTraceWriter(Path(tmpdir) / "trace.jsonl"),
            )
            first = hook(step)
            with self.assertRaisesRegex(FloatingPointError, "Two consecutive"):
                hook(step)

        self.assertTrue(torch.equal(first, noise))

    def test_final_target_correction_restores_margin_on_clean_latent(self):
        import torch

        from cci_diff.adapters.sd2_clean_cci import FinalTargetLatentCorrectionHook
        from cci_diff.sd2_bld_backend import SD2DenoisingStep

        class VAE:
            def decode(self, latent):
                return SimpleNamespace(sample=latent[:, :3])

        class Target:
            def logit(self, image):
                return image.mean()

        latents = torch.zeros((1, 4, 2, 2))
        mask = torch.ones((1, 1, 2, 2))
        step = SD2DenoisingStep(
            10,
            torch.tensor(0),
            "target",
            latents,
            torch.zeros_like(latents),
            latents,
            mask,
            mask,
            11,
            1.0,
        )
        hook = FinalTargetLatentCorrectionHook(
            vae=VAE(),
            target_evaluator=Target(),
            desired_value=1,
            target_probability=0.63,
            max_steps=4,
            step_radius=0.1,
        )

        corrected = hook(step)

        self.assertIsNotNone(corrected)
        self.assertGreater(
            hook.record["final_probability"],
            hook.record["initial_probability"],
        )
        self.assertGreater(hook.record["accepted_steps"], 0)

    def test_classifier_attribution_mask_is_semantically_bounded(self):
        import torch

        from cci_diff.adapters.sd2_clean_cci import classifier_attribution_mask

        image = torch.zeros((1, 3, 2, 2), requires_grad=True)
        importance = torch.tensor(
            [[[[1.0, 4.0], [2.0, 8.0]]]]
        ).expand_as(image)
        loss = (image * importance).sum()
        semantic_mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])

        support = classifier_attribution_mask(
            loss,
            image,
            semantic_mask,
            latent_size=(2, 2),
        )

        self.assertEqual(tuple(support.shape), (1, 1, 2, 2))
        self.assertTrue(torch.equal(support[:, :, 1], torch.zeros((1, 1, 2))))
        self.assertGreater(
            float(support[0, 0, 0, 1]),
            float(support[0, 0, 0, 0]),
        )
        self.assertAlmostEqual(float(support.max()), 1.0)

    def test_final_target_correction_uses_semantic_attribution_support(self):
        import torch

        from cci_diff.adapters.sd2_clean_cci import FinalTargetLatentCorrectionHook
        from cci_diff.sd2_bld_backend import SD2DenoisingStep

        class VAE:
            def decode(self, latent):
                return SimpleNamespace(sample=latent[:, :3])

        class Target:
            def logit(self, image):
                return image.mean()

        latents = torch.zeros((1, 4, 2, 2))
        generation_mask = torch.ones((1, 1, 2, 2))
        semantic_mask = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        step = SD2DenoisingStep(
            10,
            torch.tensor(0),
            "target",
            latents,
            torch.zeros_like(latents),
            latents,
            generation_mask,
            semantic_mask,
            11,
            1.0,
        )
        hook = FinalTargetLatentCorrectionHook(
            vae=VAE(),
            target_evaluator=Target(),
            desired_value=1,
            target_probability=0.65,
            max_steps=1,
            step_radius=0.1,
            mask_mode="semantic_attribution",
        )

        corrected = hook(step)

        self.assertIsNotNone(corrected)
        changed = (corrected - latents).abs().sum(dim=1, keepdim=True) > 0
        self.assertTrue(bool(changed[0, 0, 0, 0]))
        self.assertFalse(bool(changed[0, 0, 0, 1]))
        self.assertFalse(bool(changed[0, 0, 1, 0]))
        self.assertFalse(bool(changed[0, 0, 1, 1]))
        self.assertEqual(
            hook.record["mask"]["mode"],
            "semantic_attribution",
        )

    def test_final_target_correction_only_runs_at_last_step(self):
        import torch

        from cci_diff.adapters.sd2_clean_cci import FinalTargetLatentCorrectionHook
        from cci_diff.sd2_bld_backend import SD2DenoisingStep

        hook = FinalTargetLatentCorrectionHook(
            vae=object(),
            target_evaluator=object(),
            desired_value=0,
            target_probability=0.8,
            max_steps=4,
            step_radius=0.1,
        )
        latents = torch.zeros((1, 4, 2, 2))
        mask = torch.ones((1, 1, 2, 2))
        step = SD2DenoisingStep(
            5,
            torch.tensor(5),
            "target",
            latents,
            latents,
            latents,
            mask,
            mask,
            11,
            0.5,
        )

        self.assertIsNone(hook(step))


if __name__ == "__main__":
    unittest.main()
