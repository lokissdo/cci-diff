import math
import sys
import unittest
from unittest import mock

from cci_diff.guidance import GuidanceTerms
from cci_diff.spec import GuidanceWeights


class TestSD2CCIAdapterContract(unittest.TestCase):
    def test_apply_cci_latent_guidance_decodes_composes_loss_and_updates_latents(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        from cci_diff.adapters.sd2_cci import apply_cci_latent_guidance

        calls = []
        latents = torch.tensor([1.0, -1.0])
        weights = GuidanceWeights(
            target=1.0,
            preservation=0.0,
            leakage=0.0,
            classifier=0.0,
            outside_mask=0.0,
        )

        def decode_fn(guided_latents):
            calls.append(("decode", guided_latents.requires_grad))
            return guided_latents * 2.0

        def loss_fn(decoded):
            calls.append(("loss", decoded.requires_grad))
            zero = decoded.sum() * 0.0
            return GuidanceTerms(
                target=decoded.sum(),
                preservation=zero,
                leakage=zero,
                classifier=zero,
                outside_mask=zero,
            )

        guided = apply_cci_latent_guidance(
            latents,
            decode_fn=decode_fn,
            loss_fn=loss_fn,
            weights=weights,
            step_size=0.1,
        )

        self.assertEqual(calls, [("decode", True), ("loss", True)])
        self.assertFalse(guided.requires_grad)
        self.assertTrue(torch.allclose(guided, torch.tensor([0.8, -1.2])))
        self.assertTrue(torch.allclose(latents, torch.tensor([1.0, -1.0])))

    def test_apply_cci_latent_guidance_respects_latent_mask(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        from cci_diff.adapters.sd2_cci import apply_cci_latent_guidance

        latents = torch.tensor([1.0, 1.0])
        latent_mask = torch.tensor([1.0, 0.0])
        weights = GuidanceWeights(
            target=1.0,
            preservation=0.0,
            leakage=0.0,
            classifier=0.0,
            outside_mask=0.0,
        )

        def loss_fn(decoded):
            zero = decoded.sum() * 0.0
            return GuidanceTerms(
                target=decoded.sum(),
                preservation=zero,
                leakage=zero,
                classifier=zero,
                outside_mask=zero,
            )

        guided = apply_cci_latent_guidance(
            latents,
            decode_fn=lambda guided_latents: guided_latents,
            loss_fn=loss_fn,
            weights=weights,
            step_size=0.1,
            latent_mask=latent_mask,
        )

        self.assertTrue(torch.allclose(guided, torch.tensor([0.9, 1.0])))

    def test_apply_cci_latent_guidance_can_normalize_gradient(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        from cci_diff.adapters.sd2_cci import apply_cci_latent_guidance

        latents = torch.tensor([1.0, 1.0])
        weights = GuidanceWeights(
            target=1.0,
            preservation=0.0,
            leakage=0.0,
            classifier=0.0,
            outside_mask=0.0,
        )

        def loss_fn(decoded):
            zero = decoded.sum() * 0.0
            return GuidanceTerms(
                target=decoded.sum(),
                preservation=zero,
                leakage=zero,
                classifier=zero,
                outside_mask=zero,
            )

        guided = apply_cci_latent_guidance(
            latents,
            decode_fn=lambda guided_latents: guided_latents,
            loss_fn=loss_fn,
            weights=weights,
            step_size=1.0,
            normalize_gradient=True,
        )

        expected = torch.tensor([1.0 - 1.0 / math.sqrt(2.0)] * 2)
        self.assertTrue(torch.allclose(guided, expected))

    def test_apply_cci_latent_guidance_raises_helpful_error_without_torch(self):
        sys.modules.pop("cci_diff.adapters.sd2_cci", None)
        with mock.patch.dict("sys.modules", {"torch": None}):
            from cci_diff.adapters.sd2_cci import apply_cci_latent_guidance

            with self.assertRaises(ImportError) as ctx:
                apply_cci_latent_guidance(
                    object(),
                    decode_fn=lambda value: value,
                    loss_fn=lambda value: value,
                    weights=GuidanceWeights(),
                    step_size=0.1,
                )

        self.assertIn("CCI SD2 adapter requires torch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
