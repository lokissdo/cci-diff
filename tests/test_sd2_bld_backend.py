import tempfile
import unittest
from unittest import mock


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape


class FakeTimestep:
    def item(self):
        return 17


class FakeTorch:
    float16 = "float16"
    float32 = "float32"


class FakeComponent:
    def __init__(self):
        self.eval_called = False
        self.requires_grad_value = None

    def to(self, device=None, dtype=None):
        self.device = device
        self.dtype = dtype
        return self

    def eval(self):
        self.eval_called = True
        return self

    def requires_grad_(self, value):
        self.requires_grad_value = value
        return self


class FakePipeline:
    def __init__(self):
        self.vae = FakeComponent()
        self.tokenizer = object()
        self.text_encoder = FakeComponent()
        self.unet = FakeComponent()
        self.loaded_lora_path = None

    def load_lora_weights(self, path):
        self.loaded_lora_path = path


class FakePipelineFactory:
    last_kwargs = None
    last_pipeline = None

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        cls.last_kwargs = {"model_path": model_path, **kwargs}
        cls.last_pipeline = FakePipeline()
        return cls.last_pipeline


class FakeScheduler:
    last_kwargs = None

    def __init__(self, **kwargs):
        FakeScheduler.last_kwargs = kwargs


class TestSD2BLDBackendContract(unittest.TestCase):
    def test_seeded_noise_like_replays_the_same_random_stream(self):
        import torch

        from cci_diff.sd2_bld_backend import seeded_noise_like

        reference = torch.zeros((1, 4, 2, 2))
        first_generator = torch.Generator().manual_seed(42)
        second_generator = torch.Generator().manual_seed(42)

        first = seeded_noise_like(reference, first_generator)
        second = seeded_noise_like(reference, second_generator)

        self.assertTrue(torch.equal(first, second))
    def test_require_sd2_dependencies_raises_helpful_error_when_missing(self):
        from cci_diff.sd2_bld_backend import require_sd2_dependencies

        with mock.patch.dict("sys.modules", {"torch": None, "diffusers": None}):
            with self.assertRaises(ImportError) as ctx:
                require_sd2_dependencies()

        self.assertIn("pip install -e '.[ml]'", str(ctx.exception))

    def test_blending_start_index_clamps_to_timestep_range(self):
        from cci_diff.sd2_bld_backend import blending_start_index

        self.assertEqual(blending_start_index(50, 0.25), 12)
        self.assertEqual(blending_start_index(50, -1.0), 0)
        self.assertEqual(blending_start_index(50, 2.0), 49)

    def test_apply_cci_guidance_uses_hook_result_when_present(self):
        from cci_diff.sd2_bld_backend import SD2DenoisingStep, apply_cci_guidance

        original_noise = object()
        guided_noise = object()
        step = SD2DenoisingStep(
            step_index=0,
            timestep=17,
            prompt="add smile",
            latents=FakeTensor((4, 4, 64, 64)),
            noise_pred=original_noise,
            source_latents=FakeTensor((1, 4, 64, 64)),
            latent_mask=FakeTensor((1, 1, 64, 64)),
        )

        result = apply_cci_guidance(original_noise, step, lambda ctx: guided_noise)

        self.assertIs(result, guided_noise)

    def test_apply_cci_guidance_keeps_original_noise_when_hook_returns_none(self):
        from cci_diff.sd2_bld_backend import SD2DenoisingStep, apply_cci_guidance

        original_noise = object()
        step = SD2DenoisingStep(
            step_index=0,
            timestep=17,
            prompt="add smile",
            latents=FakeTensor((4, 4, 64, 64)),
            noise_pred=original_noise,
            source_latents=FakeTensor((1, 4, 64, 64)),
            latent_mask=FakeTensor((1, 1, 64, 64)),
        )

        result = apply_cci_guidance(original_noise, step, lambda ctx: None)

        self.assertIs(result, original_noise)

    def test_apply_cci_latent_guidance_uses_hook_result_when_present(self):
        from cci_diff.sd2_bld_backend import (
            SD2DenoisingStep,
            apply_cci_latent_guidance_hook,
        )

        original_latents = object()
        guided_latents = object()
        step = SD2DenoisingStep(
            step_index=0,
            timestep=17,
            prompt="add blond hair",
            latents=original_latents,
            noise_pred=FakeTensor((4, 4, 64, 64)),
            source_latents=FakeTensor((1, 4, 64, 64)),
            latent_mask=FakeTensor((1, 1, 64, 64)),
        )

        result = apply_cci_latent_guidance_hook(
            original_latents,
            step,
            lambda ctx: guided_latents,
        )

        self.assertIs(result, guided_latents)

    def test_apply_cci_latent_guidance_keeps_original_when_hook_returns_none(self):
        from cci_diff.sd2_bld_backend import (
            SD2DenoisingStep,
            apply_cci_latent_guidance_hook,
        )

        original_latents = object()
        step = SD2DenoisingStep(
            step_index=0,
            timestep=17,
            prompt="add blond hair",
            latents=original_latents,
            noise_pred=FakeTensor((4, 4, 64, 64)),
            source_latents=FakeTensor((1, 4, 64, 64)),
            latent_mask=FakeTensor((1, 1, 64, 64)),
        )

        result = apply_cci_latent_guidance_hook(
            original_latents,
            step,
            lambda ctx: None,
        )

        self.assertIs(result, original_latents)

    def test_diffusion_state_from_step_records_phase_and_shapes(self):
        from cci_diff.sd2_bld_backend import SD2DenoisingStep, diffusion_state_from_step

        step = SD2DenoisingStep(
            step_index=3,
            timestep=FakeTimestep(),
            prompt="add smile",
            latents=FakeTensor((4, 4, 64, 64)),
            noise_pred=FakeTensor((4, 4, 64, 64)),
            source_latents=FakeTensor((1, 4, 64, 64)),
            latent_mask=FakeTensor((1, 1, 64, 64)),
        )

        state = diffusion_state_from_step(step, phase="blend")
        payload = state.to_dict()

        self.assertEqual(payload["phase"], "blend")
        self.assertEqual(payload["timestep"], 17)
        self.assertEqual(payload["latent_shape"], [4, 4, 64, 64])
        self.assertEqual(payload["extra"]["source_latent_shape"], [1, 4, 64, 64])
        self.assertEqual(payload["extra"]["mask_shape"], [1, 1, 64, 64])

    def test_blend_latents_masks_out_nan_values_from_preserved_region(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        import cci_diff.sd2_bld_backend as backend_module

        blend_latents = getattr(backend_module, "blend_latents", None)
        self.assertIsNotNone(blend_latents)
        if blend_latents is None:
            return

        latents = torch.tensor([float("nan"), 4.0]).reshape(1, 1, 1, 2)
        latent_mask = torch.tensor([0.0, 1.0]).reshape(1, 1, 1, 2)
        source_latents = torch.tensor([2.0, 3.0]).reshape(1, 1, 1, 2)

        blended = blend_latents(latents, latent_mask, source_latents)

        self.assertTrue(torch.isfinite(blended).all())
        self.assertEqual(blended[0, 0, 0, 0].item(), 2.0)
        self.assertEqual(blended[0, 0, 0, 1].item(), 4.0)

    def test_soft_blend_interpolates_edited_and_source_latents(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        from cci_diff.sd2_bld_backend import blend_soft_latents

        edited = torch.tensor([10.0, 10.0])
        source = torch.tensor([2.0, 2.0])
        mask = torch.tensor([0.0, 0.25])

        result = blend_soft_latents(edited, mask, source)

        self.assertTrue(torch.allclose(result, torch.tensor([2.0, 4.0])))

    def test_replace_nonfinite_latents_uses_fallback_values(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        import cci_diff.sd2_bld_backend as backend_module

        replace_nonfinite = getattr(backend_module, "replace_nonfinite_latents", None)
        self.assertIsNotNone(replace_nonfinite)
        if replace_nonfinite is None:
            return

        latents = torch.tensor([float("nan"), float("inf"), 4.0])
        fallback = torch.tensor([1.0, 2.0, 3.0])

        cleaned = replace_nonfinite(latents, fallback)

        self.assertTrue(torch.isfinite(cleaned).all())
        self.assertEqual(cleaned.tolist(), [1.0, 2.0, 4.0])

    def test_backend_constructor_loads_pipeline_lora_and_ddim_scheduler(self):
        import cci_diff.sd2_bld_backend as backend_module

        with mock.patch.object(
            backend_module,
            "require_sd2_dependencies",
            return_value=(FakeTorch, object(), object(), FakeScheduler, FakePipelineFactory),
        ), mock.patch(
            "transformers.utils.logging.is_progress_bar_enabled",
            return_value=True,
        ), mock.patch(
            "transformers.utils.logging.disable_progress_bar"
        ) as disable_progress, mock.patch(
            "transformers.utils.logging.enable_progress_bar"
        ) as enable_progress:
            backend = backend_module.BlendedLatentDiffusionSD2Backend(
                model_path="local-sd2",
                device="cuda",
                torch_dtype="float16",
                lora_path="local-lora",
                local_files_only=True,
            )

        self.assertEqual(backend.name, "sd2-bld")
        self.assertEqual(FakePipelineFactory.last_kwargs["model_path"], "local-sd2")
        self.assertEqual(FakePipelineFactory.last_kwargs["torch_dtype"], FakeTorch.float16)
        self.assertIsNone(FakePipelineFactory.last_kwargs["safety_checker"])
        self.assertTrue(FakePipelineFactory.last_kwargs["local_files_only"])
        self.assertEqual(FakePipelineFactory.last_pipeline.loaded_lora_path, "local-lora")
        self.assertEqual(FakeScheduler.last_kwargs["beta_schedule"], "scaled_linear")
        self.assertTrue(FakePipelineFactory.last_pipeline.vae.eval_called)
        self.assertEqual(FakePipelineFactory.last_pipeline.vae.dtype, FakeTorch.float32)
        self.assertFalse(FakePipelineFactory.last_pipeline.vae.requires_grad_value)
        self.assertTrue(FakePipelineFactory.last_pipeline.text_encoder.eval_called)
        self.assertFalse(
            FakePipelineFactory.last_pipeline.text_encoder.requires_grad_value
        )
        self.assertTrue(FakePipelineFactory.last_pipeline.unet.eval_called)
        self.assertFalse(FakePipelineFactory.last_pipeline.unet.requires_grad_value)
        disable_progress.assert_called_once_with()
        enable_progress.assert_called_once_with()

    def test_read_mask_uses_configured_float32_dtype(self):
        try:
            import torch
            from PIL import Image
        except ImportError:
            self.skipTest("ML dependencies are not installed")

        from cci_diff.sd2_bld_backend import BlendedLatentDiffusionSD2Backend

        backend = BlendedLatentDiffusionSD2Backend.__new__(
            BlendedLatentDiffusionSD2Backend
        )
        backend.device = "cpu"
        backend.torch_dtype = torch.float32

        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = f"{tmpdir}/mask.png"
            Image.new("L", (8, 8), color=255).save(mask_path)

            mask = backend._read_mask(mask_path, dest_size=(4, 4))

        self.assertEqual(mask.dtype, torch.float32)

    def test_read_mask_can_preserve_fractional_values(self):
        try:
            import torch
            from PIL import Image
        except ImportError:
            self.skipTest("ML dependencies are not installed")

        from cci_diff.sd2_bld_backend import BlendedLatentDiffusionSD2Backend

        backend = BlendedLatentDiffusionSD2Backend.__new__(
            BlendedLatentDiffusionSD2Backend
        )
        backend.device = "cpu"
        backend.torch_dtype = torch.float32

        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = f"{tmpdir}/mask.png"
            mask_image = Image.new("L", (3, 1))
            mask_image.putdata([0, 128, 255])
            mask_image.save(mask_path)

            mask = backend._read_mask(
                mask_path,
                dest_size=(3, 1),
                binary=False,
            )

        self.assertAlmostEqual(mask[0, 0, 0, 1].item(), 128 / 255)

    def test_denoising_progress_covers_selected_reverse_interval(self):
        from cci_diff.sd2_bld_backend import denoising_progress

        self.assertEqual(denoising_progress(0, 1), 0.0)
        self.assertEqual(denoising_progress(0, 5), 0.0)
        self.assertEqual(denoising_progress(2, 5), 0.5)
        self.assertEqual(denoising_progress(4, 5), 1.0)
        with self.assertRaisesRegex(ValueError, "total_steps"):
            denoising_progress(0, 0)
        with self.assertRaisesRegex(ValueError, "selected reverse interval"):
            denoising_progress(5, 5)

    def test_old_step_constructor_gets_safe_context_defaults(self):
        from cci_diff.sd2_bld_backend import SD2DenoisingStep

        step = SD2DenoisingStep(
            step_index=0,
            timestep=17,
            prompt="legacy",
            latents=object(),
            noise_pred=object(),
            source_latents=object(),
            latent_mask=object(),
        )

        self.assertIsNone(step.semantic_mask)
        self.assertEqual(step.total_steps, 1)
        self.assertEqual(step.progress, 0.0)

    def test_step_state_records_new_semantic_context(self):
        from cci_diff.sd2_bld_backend import (
            SD2DenoisingStep,
            diffusion_state_from_step,
        )

        step = SD2DenoisingStep(
            step_index=1,
            timestep=17,
            prompt="context",
            latents=FakeTensor((1, 4, 8, 8)),
            noise_pred=FakeTensor((1, 4, 8, 8)),
            source_latents=FakeTensor((1, 4, 8, 8)),
            latent_mask=FakeTensor((1, 1, 8, 8)),
            semantic_mask=FakeTensor((1, 1, 8, 8)),
            total_steps=3,
            progress=0.5,
        )
        extra = diffusion_state_from_step(step, phase="cci_guidance").extra

        self.assertEqual(extra["semantic_mask_shape"], (1, 1, 8, 8))
        self.assertEqual(extra["total_steps"], 3)
        self.assertEqual(extra["progress"], 0.5)

    def test_hook_and_blend_order_remains_pre_scheduler_then_post_scheduler(self):
        import inspect

        from cci_diff.sd2_bld_backend import BlendedLatentDiffusionSD2Backend

        source = inspect.getsource(BlendedLatentDiffusionSD2Backend.edit_image)
        self.assertLess(
            source.index("apply_cci_guidance("),
            source.index("self.scheduler.step("),
        )
        self.assertLess(
            source.index("self.scheduler.step("),
            source.index("apply_cci_latent_guidance_hook("),
        )
        self.assertLess(
            source.index("apply_cci_latent_guidance_hook("),
            source.index("blend_soft_latents("),
        )


if __name__ == "__main__":
    unittest.main()
