import unittest

from cci_diff.guidance import GuidanceTerms, compose_guidance_loss
from cci_diff.spec import GuidanceWeights


class TestGuidanceObjective(unittest.TestCase):
    def test_compose_guidance_loss_weights_all_terms(self):
        terms = GuidanceTerms(
            target=0.20,
            preservation=0.10,
            leakage=0.40,
            classifier=0.30,
            outside_mask=0.05,
        )
        weights = GuidanceWeights(
            target=2.0,
            preservation=3.0,
            leakage=0.5,
            classifier=1.0,
            outside_mask=4.0,
        )

        self.assertAlmostEqual(compose_guidance_loss(terms, weights), 1.4)

    def test_compose_guidance_loss_weights_clip_and_smooth_terms(self):
        terms = GuidanceTerms(
            target=0.20,
            preservation=0.10,
            leakage=0.40,
            classifier=0.30,
            outside_mask=0.05,
            clip=0.60,
            smooth=0.20,
        )
        weights = GuidanceWeights(
            target=2.0,
            preservation=3.0,
            leakage=0.5,
            classifier=1.0,
            outside_mask=4.0,
            clip=5.0,
            smooth=6.0,
        )

        self.assertAlmostEqual(compose_guidance_loss(terms, weights), 5.6)


if __name__ == "__main__":
    unittest.main()
