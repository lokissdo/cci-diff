import unittest

from cci_diff.prompts import build_concept_prompt
from cci_diff.spec import ConceptIntervention


class TestPromptBuilder(unittest.TestCase):
    def test_build_prompt_names_target_and_preserved_concepts(self):
        intervention = ConceptIntervention(
            target_concept="smile",
            desired_value=1,
            preserved_concepts=("identity", "hair", "age-like appearance"),
        )

        prompt = build_concept_prompt(intervention)

        self.assertIn("add smile", prompt.positive)
        self.assertIn("same person", prompt.positive)
        self.assertIn("preserve hair", prompt.positive)
        self.assertIn("do not change hair", prompt.negative)
        self.assertIn("do not change age-like appearance", prompt.negative)

    def test_build_prompt_uses_remove_for_zero_intervention(self):
        intervention = ConceptIntervention(target_concept="smile", desired_value=0)

        prompt = build_concept_prompt(intervention)

        self.assertIn("remove smile", prompt.positive)
        self.assertIn("closed relaxed lips", prompt.positive)
        self.assertIn("no smile", prompt.positive)

    def test_build_prompt_describes_blond_hair_visually(self):
        intervention = ConceptIntervention(
            target_concept="Blond_Hair",
            desired_value=1,
        )

        prompt = build_concept_prompt(intervention)

        self.assertIn("add blond hair", prompt.positive)
        self.assertIn("natural warm blond hair", prompt.positive)
        self.assertNotIn("Blond_Hair", prompt.positive)


if __name__ == "__main__":
    unittest.main()
