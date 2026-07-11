import unittest

from cci_diff.spec import ConceptIntervention, GuidanceWeights


class TestConceptIntervention(unittest.TestCase):
    def test_intervention_requires_binary_desired_value(self):
        with self.assertRaises(ValueError):
            ConceptIntervention(target_concept="smile", desired_value=2)

    def test_intervention_rejects_target_inside_preserved_concepts(self):
        with self.assertRaises(ValueError):
            ConceptIntervention(
                target_concept="smile",
                desired_value=1,
                preserved_concepts=("identity", "smile"),
            )

    def test_default_audit_concepts_exclude_target_and_preserved(self):
        intervention = ConceptIntervention(
            target_concept="smile",
            desired_value=1,
            preserved_concepts=("identity", "age"),
            candidate_concepts=("smile", "identity", "age", "makeup", "glasses"),
        )

        self.assertEqual(intervention.audit_concepts, ("makeup", "glasses"))


class TestGuidanceWeights(unittest.TestCase):
    def test_guidance_weights_are_non_negative(self):
        with self.assertRaises(ValueError):
            GuidanceWeights(target=-0.1)

    def test_guidance_weights_have_practical_defaults(self):
        weights = GuidanceWeights()

        self.assertEqual(weights.target, 1.0)
        self.assertEqual(weights.preservation, 1.0)
        self.assertEqual(weights.leakage, 0.5)
        self.assertEqual(weights.classifier, 1.0)
        self.assertEqual(weights.outside_mask, 1.0)


if __name__ == "__main__":
    unittest.main()
