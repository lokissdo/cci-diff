import unittest
from pathlib import Path


class TestCelebAResNet50(unittest.TestCase):
    def test_smile_resolves_to_canonical_celeba_index(self):
        from cci_diff.classifiers.celeba_resnet50 import (
            resolve_celeba_attribute_index,
        )

        self.assertEqual(resolve_celeba_attribute_index("smile"), 31)
        self.assertEqual(resolve_celeba_attribute_index("Smiling"), 31)
        with self.assertRaises(ValueError):
            resolve_celeba_attribute_index("unknown concept")

    def test_preprocessing_is_differentiable_and_normalized(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        from cci_diff.classifiers.celeba_resnet50 import (
            preprocess_classifier_images,
        )

        image = torch.full((1, 3, 8, 8), 0.5, requires_grad=True)
        result = preprocess_classifier_images(image, size=16)
        result.sum().backward()

        expected = torch.tensor(
            [
                (0.5 - 0.485) / 0.229,
                (0.5 - 0.456) / 0.224,
                (0.5 - 0.406) / 0.225,
            ]
        )
        self.assertEqual(tuple(result.shape), (1, 3, 16, 16))
        self.assertTrue(torch.allclose(result[0, :, 0, 0], expected))
        self.assertIsNotNone(image.grad)

    def test_local_checkpoint_loads_without_pretrained_download(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        from cci_diff.classifiers.celeba_resnet50 import load_celeba_resnet50

        checkpoint = Path("models/resnet50_multilabel_model.pth")
        if not checkpoint.exists():
            self.skipTest("local classifier checkpoint is not available")

        model = load_celeba_resnet50(
            checkpoint,
            device="cpu",
            dtype=torch.float32,
        )

        self.assertFalse(model.training)
        self.assertEqual(model.fc3.out_features, 40)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.parameters())
        )


if __name__ == "__main__":
    unittest.main()
