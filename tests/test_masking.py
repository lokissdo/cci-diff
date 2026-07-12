import tempfile
import unittest
from pathlib import Path


class TestSemanticMaskPreparation(unittest.TestCase):
    def _save_mask(self, path: Path, points, *, size=(16, 16)) -> None:
        from PIL import Image

        image = Image.new("L", size, color=0)
        pixels = image.load()
        for x, y in points:
            pixels[x, y] = 255
        image.save(path)

    def test_semantic_union_ors_components_and_feathers_edges(self):
        import numpy as np
        from PIL import Image

        from cci_diff.masking import prepare_semantic_masks

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mouth = root / "mouth.png"
            upper = root / "upper.png"
            lower = root / "lower.png"
            self._save_mask(mouth, [(7, 8), (8, 8)])
            self._save_mask(upper, [(7, 7)])
            self._save_mask(lower, [(8, 9)])

            artifacts = prepare_semantic_masks(
                [mouth, upper, lower],
                feather_radius=3,
                hard_output=root / "semantic.png",
                soft_output=root / "generation.png",
            )

            hard = np.array(Image.open(artifacts.semantic_path))
            soft = np.array(Image.open(artifacts.generation_path))

        self.assertEqual(int((hard >= 128).sum()), 4)
        self.assertAlmostEqual(artifacts.semantic_fraction, 4 / 256)
        self.assertTrue(((soft > 0) & (soft < 255)).any())

    def test_rejects_component_masks_with_different_sizes(self):
        from cci_diff.masking import prepare_semantic_masks

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.png"
            second = root / "second.png"
            self._save_mask(first, [(1, 1)], size=(16, 16))
            self._save_mask(second, [(1, 1)], size=(8, 8))

            with self.assertRaisesRegex(ValueError, "same dimensions"):
                prepare_semantic_masks(
                    [first, second],
                    feather_radius=3,
                    hard_output=root / "semantic.png",
                    soft_output=root / "generation.png",
                )

    def test_rejects_negative_feather_radius(self):
        from cci_diff.masking import prepare_semantic_masks

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mask = root / "mask.png"
            self._save_mask(mask, [(1, 1)])

            with self.assertRaisesRegex(ValueError, "non-negative"):
                prepare_semantic_masks(
                    [mask],
                    feather_radius=-1,
                    hard_output=root / "semantic.png",
                    soft_output=root / "generation.png",
                )


if __name__ == "__main__":
    unittest.main()
