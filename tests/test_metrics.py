import unittest

from cci_diff.metrics import (
    bias_audit_matrix,
    causal_concept_effect,
    concept_delta,
    concept_leakage,
    counterfactual_purity,
    preservation_score,
    target_concept_success,
)


class TestConceptMetrics(unittest.TestCase):
    def test_concept_delta_uses_absolute_score_change(self):
        original = {"smile": 0.10, "makeup": 0.80}
        counterfactual = {"smile": 0.92, "makeup": 0.55}

        self.assertAlmostEqual(concept_delta(original, counterfactual, "smile"), 0.82)
        self.assertAlmostEqual(concept_delta(original, counterfactual, "makeup"), 0.25)

    def test_target_success_checks_desired_value_threshold(self):
        counterfactual = {"smile": 0.72, "makeup": 0.30}

        self.assertTrue(target_concept_success(counterfactual, "smile", desired_value=1))
        self.assertFalse(target_concept_success(counterfactual, "makeup", desired_value=1))
        self.assertTrue(target_concept_success(counterfactual, "makeup", desired_value=0))

    def test_preservation_score_is_high_when_preserved_concepts_are_stable(self):
        original = {"identity": 0.95, "age": 0.40, "makeup": 0.10}
        counterfactual = {"identity": 0.90, "age": 0.35, "makeup": 0.70}

        score = preservation_score(original, counterfactual, ["identity", "age"])

        self.assertAlmostEqual(score, 0.95)

    def test_concept_leakage_averages_non_target_changes(self):
        original = {"smile": 0.10, "makeup": 0.20, "age": 0.60}
        counterfactual = {"smile": 0.90, "makeup": 0.50, "age": 0.50}

        leakage = concept_leakage(original, counterfactual, ["makeup", "age"])

        self.assertAlmostEqual(leakage, 0.20)

    def test_counterfactual_purity_rewards_target_change_over_leakage(self):
        purity = counterfactual_purity(target_delta=0.80, leakage_deltas=[0.10, 0.10])

        self.assertAlmostEqual(purity, 0.80)

    def test_causal_concept_effect_is_mean_score_change(self):
        before = [0.20, 0.30, 0.50]
        after = [0.70, 0.60, 0.55]

        self.assertAlmostEqual(causal_concept_effect(before, after), 0.2833333333333333)

    def test_bias_audit_matrix_groups_mean_deltas_by_intervention_and_classifier(self):
        records = [
            {"intervention": "makeup", "classifier": "smile", "before": 0.20, "after": 0.50},
            {"intervention": "makeup", "classifier": "smile", "before": 0.10, "after": 0.30},
            {"intervention": "glasses", "classifier": "young", "before": 0.80, "after": 0.40},
        ]

        matrix = bias_audit_matrix(records)

        self.assertAlmostEqual(matrix["makeup"]["smile"], 0.25)
        self.assertAlmostEqual(matrix["glasses"]["young"], -0.40)


if __name__ == "__main__":
    unittest.main()
