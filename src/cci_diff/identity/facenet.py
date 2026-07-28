"""FaceNet identity constraint with a local TorchScript model and fixed crop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cci_diff.concept_graph import sha256_file
from cci_diff.constraints import ConstraintContext


def load_facenet_identity(checkpoint_path: str | Path, *, device: str) -> Any:
    import torch

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Identity checkpoint not found: {path}")
    model = torch.jit.load(str(path), map_location="cpu")
    model.to(device=device, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_identity_export_manifest(checkpoint_path: str | Path) -> dict[str, Any]:
    import json

    checkpoint = Path(checkpoint_path)
    manifest_path = Path(str(checkpoint) + ".json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Identity export manifest not found: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"facenet_pytorch_version", "export_torch_version", "sha256"}
    if not required.issubset(payload):
        raise ValueError("Identity export manifest is missing required provenance")
    if payload["sha256"] != sha256_file(checkpoint):
        raise ValueError(
            "Identity TorchScript digest does not match its export manifest"
        )
    return payload


class OpenCVHaarDetector:
    def __init__(self) -> None:
        import cv2

        cascade_path = (
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        )
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise RuntimeError(f"Cannot load OpenCV face cascade: {cascade_path}")

    def detect(self, image: Any):
        import cv2
        import numpy as np

        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            return None, None
        boxes = np.asarray(
            [
                [x, y, x + width, y + height]
                for x, y, width, height in faces
            ],
            dtype=np.float32,
        )
        return boxes, np.ones((len(boxes),), dtype=np.float32)


def build_face_detector() -> Any:
    return OpenCVHaarDetector()


def detect_largest_face_box(
    detector: Any,
    source_image: Any,
) -> tuple[int, int, int, int]:
    from PIL import Image

    image = source_image.detach()[0].clamp(0, 1).mul(255).byte().cpu()
    pil = Image.fromarray(image.permute(1, 2, 0).numpy())
    boxes, probabilities = detector.detect(pil)
    if boxes is None or len(boxes) == 0:
        raise ValueError("FaceNet identity could not detect a source face")
    height, width = source_image.shape[-2:]
    candidates = []
    for box, probability in zip(boxes, probabilities):
        x1, y1, x2, y2 = [float(value) for value in box]
        area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        candidates.append((area, float(probability), x1, y1, x2, y2))
    _, _, x1, y1, x2, y2 = max(candidates)
    margin = 0.15 * max(x2 - x1, y2 - y1)
    return (
        max(0, int(round(x1 - margin))),
        max(0, int(round(y1 - margin))),
        min(width, int(round(x2 + margin))),
        min(height, int(round(y2 + margin))),
    )


def fixed_face_crop(
    images: Any,
    box: tuple[int, int, int, int],
    *,
    size: int = 160,
) -> Any:
    import torch.nn.functional as functional

    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid fixed face box: {box}")
    if size <= 0:
        raise ValueError("Face crop size must be positive")
    crop = images[:, :, y1:y2, x1:x2]
    return functional.interpolate(
        crop,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )


def standardize_face(crop: Any) -> Any:
    return (crop * 255.0 - 127.5) / 128.0


def model_input_dtype(model: Any, fallback: Any) -> Any:
    """Return the model parameter dtype, or the input dtype for parameterless models."""

    try:
        return next(model.parameters()).dtype
    except (AttributeError, StopIteration):
        return fallback


class FaceNetIdentityConstraint:
    def __init__(
        self,
        name: str,
        model: Any,
        detector: Any,
        *,
        tolerance: float,
        crop_size: int = 160,
    ) -> None:
        self.name = name
        self.model = model
        self.detector = detector
        self.tolerance = tolerance
        self.crop_size = crop_size
        self.face_box = None
        self._source_embedding = None

    def bind(self, context: ConstraintContext) -> None:
        import torch

        self.face_box = detect_largest_face_box(
            self.detector,
            context.source_image,
        )
        crop = fixed_face_crop(
            context.source_image,
            self.face_box,
            size=self.crop_size,
        )
        with torch.no_grad():
            standardized = standardize_face(crop).to(
                dtype=model_input_dtype(self.model, crop.dtype)
            )
            self._source_embedding = torch.nn.functional.normalize(
                self.model(standardized),
                dim=1,
            ).detach()

    def measure(self, image: Any) -> Any:
        import torch

        if self.face_box is None or self._source_embedding is None:
            raise RuntimeError(f"Constraint {self.name!r} is not bound to a source")
        crop = fixed_face_crop(image, self.face_box, size=self.crop_size)
        standardized = standardize_face(crop).to(
            dtype=model_input_dtype(self.model, crop.dtype)
        )
        embedding = torch.nn.functional.normalize(
            self.model(standardized),
            dim=1,
        )
        source = self._source_embedding.to(
            device=image.device,
            dtype=embedding.dtype,
        )
        return (
            1.0
            - torch.nn.functional.cosine_similarity(
                embedding,
                source,
                dim=1,
            )
        ).mean()
