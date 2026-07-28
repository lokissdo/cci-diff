import tempfile
import unittest
from pathlib import Path


class FakeDetector:
    def detect(self, image):
        import numpy as np

        return np.array([[1.0, 1.0, 7.0, 7.0]]), np.array([0.99])


class MeanEmbedder:
    def __call__(self, images):
        import torch

        mean = images.mean(dim=(2, 3))
        return torch.nn.functional.normalize(mean + 0.01, dim=1)


class TestFaceNetIdentity(unittest.TestCase):
    def test_build_face_detector_uses_bundled_opencv_cascade(self):
        from cci_diff.identity.facenet import build_face_detector

        detector = build_face_detector()
        self.assertIsNotNone(detector.cascade)

    def test_fixed_crop_is_differentiable_and_has_stable_size(self):
        import torch

        from cci_diff.identity.facenet import fixed_face_crop

        image = torch.rand((1, 3, 8, 8), requires_grad=True)
        crop = fixed_face_crop(image, (1, 1, 7, 7), size=4)
        crop.sum().backward()

        self.assertEqual(tuple(crop.shape), (1, 3, 4, 4))
        self.assertIsNotNone(image.grad)

    def test_identity_distance_is_zero_for_source_and_positive_after_change(self):
        import torch

        from cci_diff.constraints import ConstraintContext
        from cci_diff.identity.facenet import FaceNetIdentityConstraint

        source = torch.zeros((1, 3, 8, 8))
        mask = torch.ones((1, 1, 8, 8))
        evaluator = FaceNetIdentityConstraint(
            "identity",
            MeanEmbedder(),
            FakeDetector(),
            tolerance=0.08,
            crop_size=4,
        )
        evaluator.bind(ConstraintContext(source, mask, mask))

        self.assertAlmostEqual(evaluator.measure(source).item(), 0.0, places=6)
        changed = source.clone()
        changed[:, 0] = 1.0
        self.assertGreater(evaluator.measure(changed).item(), 0.0)

    def test_identity_casts_half_images_to_float_model_and_keeps_gradients(self):
        import torch

        from cci_diff.constraints import ConstraintContext
        from cci_diff.identity.facenet import FaceNetIdentityConstraint

        class Float32Embedder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Conv2d(3, 3, kernel_size=1, bias=False)

            def forward(self, images):
                return self.projection(images).mean(dim=(2, 3))

        source = torch.zeros((1, 3, 8, 8), dtype=torch.float16)
        mask = torch.ones((1, 1, 8, 8), dtype=torch.float16)
        evaluator = FaceNetIdentityConstraint(
            "identity",
            Float32Embedder().float().eval(),
            FakeDetector(),
            tolerance=0.08,
            crop_size=4,
        )
        evaluator.bind(ConstraintContext(source, mask, mask))
        changed = source.clone()
        changed[:, 0] = 1.0
        changed.requires_grad_(True)

        loss = evaluator.measure(changed)
        loss.backward()

        self.assertEqual(loss.dtype, torch.float32)
        self.assertIsNotNone(changed.grad)
        self.assertTrue(torch.isfinite(changed.grad).all())

    def test_detector_selects_largest_face_and_expands_it(self):
        import numpy as np
        import torch

        from cci_diff.identity.facenet import detect_largest_face_box

        class MultipleDetector:
            def detect(self, image):
                return (
                    np.array(
                        [
                            [2.0, 2.0, 4.0, 4.0],
                            [1.0, 1.0, 7.0, 7.0],
                        ]
                    ),
                    np.array([0.99, 0.80]),
                )

        box = detect_largest_face_box(
            MultipleDetector(),
            torch.zeros((1, 3, 8, 8)),
        )
        self.assertEqual(box, (0, 0, 8, 8))

    def test_missing_detection_and_invalid_crop_fail_clearly(self):
        import torch

        from cci_diff.identity.facenet import (
            detect_largest_face_box,
            fixed_face_crop,
        )

        class EmptyDetector:
            def detect(self, image):
                return None, None

        image = torch.zeros((1, 3, 8, 8))
        with self.assertRaisesRegex(ValueError, "could not detect"):
            detect_largest_face_box(EmptyDetector(), image)
        with self.assertRaisesRegex(ValueError, "Invalid fixed face box"):
            fixed_face_crop(image, (2, 2, 2, 4))

    def test_missing_checkpoint_fails_without_downloading(self):
        from cci_diff.identity.facenet import load_facenet_identity

        with self.assertRaisesRegex(FileNotFoundError, "Identity checkpoint not found"):
            load_facenet_identity("missing-facenet.ts", device="cpu")

    def test_torchscript_loader_freezes_parameters_without_module_requires_grad(self):
        import torch

        from cci_diff.identity.facenet import load_facenet_identity

        class TinyEmbedder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, images):
                return images.mean(dim=(2, 3)) * self.scale

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tiny.ts"
            traced = torch.jit.trace(TinyEmbedder(), torch.zeros((1, 3, 4, 4)))
            traced.save(str(path))
            loaded = load_facenet_identity(path, device="cpu")

        self.assertFalse(loaded.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))

    def test_export_keeps_parameters_movable_instead_of_freezing_constants(self):
        import torch

        from scripts.download_identity_model import export_torchscript

        model = torch.nn.Conv2d(3, 2, kernel_size=1).eval()
        exported = export_torchscript(model, torch.zeros((1, 3, 4, 4)))

        self.assertGreater(sum(1 for _ in exported.parameters()), 0)

    def test_export_manifest_verifies_checkpoint_digest(self):
        import hashlib
        import json

        from cci_diff.identity.facenet import load_identity_export_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "facenet.ts"
            checkpoint.write_bytes(b"model")
            Path(str(checkpoint) + ".json").write_text(
                json.dumps(
                    {
                        "facenet_pytorch_version": "2.6.0",
                        "export_torch_version": "2.2.0",
                        "sha256": hashlib.sha256(b"model").hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            manifest = load_identity_export_manifest(checkpoint)
            checkpoint.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "digest"):
                load_identity_export_manifest(checkpoint)

        self.assertEqual(manifest["facenet_pytorch_version"], "2.6.0")


if __name__ == "__main__":
    unittest.main()
