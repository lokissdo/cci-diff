import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


class MeanCelebAClassifier:
    def forward_logits(self, images):
        import torch

        smiling = images.mean(dim=(1, 2, 3))
        logits = torch.zeros(
            (images.shape[0], 40),
            device=images.device,
            dtype=images.dtype,
        )
        logits[:, 31] = smiling
        return logits


class TestSmileClassifierHook(unittest.TestCase):
    def test_classifier_terms_use_only_classifier_and_outside_losses(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        from scripts.run_sd2_bld_cci import build_classifier_guidance_terms

        decoded = torch.full((1, 3, 2, 2), 0.75, requires_grad=True)
        source = torch.full_like(decoded, 0.25)
        image_mask = torch.zeros((1, 1, 2, 2))
        image_mask[:, :, 0, 0] = 1.0

        terms = build_classifier_guidance_terms(
            decoded,
            classifier=MeanCelebAClassifier(),
            label_index=31,
            desired_value=0,
            image_mask=image_mask,
            source_image=source,
            input_size=2,
        )

        self.assertEqual(terms.target.item(), 0.0)
        self.assertEqual(terms.preservation.item(), 0.0)
        self.assertEqual(terms.leakage.item(), 0.0)
        self.assertGreater(terms.classifier.item(), 0.0)
        self.assertGreater(terms.outside_mask.item(), 0.0)
        terms.classifier.backward()
        self.assertIsNotNone(decoded.grad)

    def test_classifier_audit_metadata_records_smile_target(self):
        from cci_diff.config import load_cci_config
        from scripts.run_sd2_bld_cci import (
            ClassifierRuntime,
            classifier_audit_metadata,
        )

        args = Namespace(classifier_path="models/resnet50_multilabel_model.pth")
        runtime = ClassifierRuntime(
            model=None,
            path="models/resnet50_multilabel_model.pth",
            label_index=31,
            attribute="Smiling",
            input_size=512,
            device="cpu",
            applied_steps=[0, 2],
        )
        config = load_cci_config("examples/remove_smile_intervention.json")

        metadata = classifier_audit_metadata(
            args,
            config,
            runtime,
            source_probability=0.99,
            output_probabilities=[0.25],
        )

        self.assertEqual(metadata["attribute"], "Smiling")
        self.assertEqual(metadata["label_index"], 31)
        self.assertEqual(metadata["desired_value"], 0)
        self.assertEqual(metadata["input_size"], 512)
        self.assertEqual(metadata["source_probability"], 0.99)
        self.assertEqual(metadata["output_probabilities"], [0.25])
        self.assertEqual(metadata["applied_steps"], [0, 2])

    def test_classifier_grid_scoring_splits_batch_images(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")

        from scripts.run_sd2_bld_cci import score_classifier_image_grid

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "grid.png"
            grid = Image.new("RGB", (4, 2), color="black")
            grid.paste(Image.new("RGB", (2, 2), color="white"), (0, 0))
            grid.save(path)

            scores = score_classifier_image_grid(
                path,
                classifier=MeanCelebAClassifier(),
                label_index=31,
                input_size=2,
                device="cpu",
                batch_size=2,
                crop_width=2,
            )

        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1])

    def test_masked_image_change_metrics_separates_inside_and_outside(self):
        import numpy as np
        from PIL import Image

        from scripts.run_sd2_bld_cci import masked_image_change_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = np.zeros((2, 2, 3), dtype=np.uint8)
            output = source.copy()
            output[0, 0] = 255
            mask = np.zeros((2, 2), dtype=np.uint8)
            mask[0, 0] = 255
            Image.fromarray(source).save(root / "source.png")
            Image.fromarray(output).save(root / "output.png")
            Image.fromarray(mask).save(root / "mask.png")

            metrics = masked_image_change_metrics(
                root / "source.png",
                root / "output.png",
                root / "mask.png",
            )

        self.assertEqual(metrics["inside_mae"], 255.0)
        self.assertEqual(metrics["outside_mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
