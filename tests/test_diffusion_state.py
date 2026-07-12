import unittest

from cci_diff.diffusion_state import DiffusionRunResult, DiffusionState


class TestDiffusionState(unittest.TestCase):
    def test_diffusion_state_records_step_timestep_and_prompt(self):
        state = DiffusionState(
            step_index=3,
            timestep=42,
            prompt="add smile",
            latent_shape=(1, 4, 64, 64),
        )

        self.assertEqual(state.step_index, 3)
        self.assertEqual(state.timestep, 42)
        self.assertEqual(state.prompt, "add smile")
        self.assertEqual(state.latent_shape, (1, 4, 64, 64))

    def test_run_result_serializes_without_image_bytes(self):
        result = DiffusionRunResult(
            image_path="outputs/sample.ppm",
            prompt="add smile",
            backend="fake",
            states=[
                DiffusionState(
                    step_index=0,
                    timestep=1,
                    prompt="add smile",
                    latent_shape=(1, 4, 8, 8),
                )
            ],
        )

        payload = result.to_dict()

        self.assertEqual(payload["image_path"], "outputs/sample.ppm")
        self.assertEqual(payload["backend"], "fake")
        self.assertEqual(payload["states"][0]["latent_shape"], [1, 4, 8, 8])


if __name__ == "__main__":
    unittest.main()
