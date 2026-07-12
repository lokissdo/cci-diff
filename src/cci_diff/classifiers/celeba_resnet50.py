"""Frozen CelebA ResNet50 adapter for differentiable CCI guidance."""

from __future__ import annotations

from pathlib import Path
from typing import Any


CELEBA_ATTRIBUTES = (
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
)


def resolve_celeba_attribute_index(concept: str) -> int:
    """Resolve a CCI concept name to its canonical CelebA output index."""

    normalized = concept.casefold().replace("_", " ").strip()
    normalized = {"smile": "smiling"}.get(normalized, normalized)
    for index, attribute in enumerate(CELEBA_ATTRIBUTES):
        if attribute.casefold().replace("_", " ") == normalized:
            return index
    raise ValueError(f"No CelebA classifier output for {concept!r}")


def preprocess_classifier_images(images: Any, *, size: int) -> Any:
    """Resize and ImageNet-normalize decoded images without breaking gradients."""

    if size <= 0:
        raise ValueError("classifier input size must be positive")

    import torch
    import torch.nn.functional as functional

    images = functional.interpolate(
        images.float(),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)
    return (images - mean) / std


def load_celeba_resnet50(
    checkpoint_path: str | Path,
    *,
    device: str,
    dtype: Any,
) -> Any:
    """Load the thesis ResNet50 checkpoint without training-only dependencies."""

    import torch
    from torch import nn
    from torchvision import models

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

    class CelebAResNet50(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_model = models.resnet50(weights=None)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
            self.fc1 = nn.Linear(num_features, 1024)
            self.bn1 = nn.BatchNorm1d(1024)
            self.relu1 = nn.ReLU()
            self.drop1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(1024, 512)
            self.bn2 = nn.BatchNorm1d(512)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(512, len(CELEBA_ATTRIBUTES))

        def forward_logits(self, images):
            features = self.base_model(images)
            features = self.relu1(self.bn1(self.fc1(features)))
            features = self.drop1(features)
            features = self.relu2(self.bn2(self.fc2(features)))
            return self.fc3(features)

        def forward(self, images):
            return torch.sigmoid(self.forward_logits(images))

    model = CelebAResNet50()
    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Classifier checkpoint is incompatible with {checkpoint_path}: {exc}"
        ) from exc

    model.to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)
    return model


def classifier_logits(model: Any, images: Any, *, size: int) -> Any:
    """Return differentiable CelebA logits for decoded images in [0, 1]."""

    normalized = preprocess_classifier_images(images, size=size)
    return model.forward_logits(normalized)


def classifier_probabilities(model: Any, images: Any, *, size: int) -> Any:
    """Return sigmoid CelebA probabilities for decoded images in [0, 1]."""

    import torch

    return torch.sigmoid(classifier_logits(model, images, size=size))
