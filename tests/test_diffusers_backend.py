import unittest
from unittest import mock

from cci_diff.diffusers_backend import callback_state_from_kwargs, require_diffusers


class FakeTimestep:
    def item(self):
        return 42


class FakeLatents:
    shape = (1, 4, 8, 8)


class TestDiffusersBackend(unittest.TestCase):
    def test_require_diffusers_raises_helpful_error_when_missing(self):
        with mock.patch.dict("sys.modules", {"torch": None, "diffusers": None}):
            with self.assertRaises(ImportError) as ctx:
                require_diffusers()

        self.assertIn("pip install -e '.[ml]'", str(ctx.exception))

    def test_callback_state_reads_timestep_and_latent_shape(self):
        state = callback_state_from_kwargs(
            step_index=2,
            timestep=FakeTimestep(),
            prompt="add smile",
            callback_kwargs={"latents": FakeLatents()},
        )

        self.assertEqual(state.step_index, 2)
        self.assertEqual(state.timestep, 42)
        self.assertEqual(state.latent_shape, (1, 4, 8, 8))


if __name__ == "__main__":
    unittest.main()
