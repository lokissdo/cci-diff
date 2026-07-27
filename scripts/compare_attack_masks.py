#!/usr/bin/env python3
"""Compare Grad-CAM++ and FacePart masks under identical targeted PGD."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from cci_diff.post_attack import gradcam_pp_saliency


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def binary_mask_from_saliency(
    saliency: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("saliency threshold must be in [0, 1]")
    return (np.asarray(saliency) >= threshold).astype(np.float32)


def normalize_images(images: Any) -> Any:
    import torch

    mean = torch.tensor(
        IMAGENET_MEAN,
        device=images.device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        IMAGENET_STD,
        device=images.device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)
    return (images - mean) / std


def unnormalize_images(images: Any) -> Any:
    import torch

    mean = torch.tensor(
        IMAGENET_MEAN,
        device=images.device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        IMAGENET_STD,
        device=images.device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)
    return images * std + mean


def targeted_masked_pgd(
    model: Any,
    image: Any,
    mask: Any,
    *,
    label_index: int,
    desired_value: int,
    epsilon: float,
    step_size: float,
    max_steps: int,
) -> tuple[Any, dict[str, float | int | bool]]:
    """Run the thesis targeted PGD update with a fixed spatial mask."""

    import torch
    import torch.nn.functional as functional

    if desired_value not in (0, 1):
        raise ValueError("desired_value must be 0 or 1")
    if epsilon <= 0 or step_size <= 0 or max_steps <= 0:
        raise ValueError("PGD epsilon, step_size, and max_steps must be positive")
    reference = image.detach()
    attack_mask = mask.to(device=image.device, dtype=image.dtype)
    if attack_mask.shape[1] == 1 and image.shape[1] != 1:
        attack_mask = attack_mask.expand(-1, image.shape[1], -1, -1)
    if attack_mask.shape != image.shape:
        raise ValueError("mask must be broadcastable across image channels")

    with torch.no_grad():
        before_probability = float(model(reference)[:, label_index].item())
    target = torch.full(
        (image.shape[0],),
        float(desired_value),
        device=image.device,
        dtype=image.dtype,
    )
    attacked = reference.clone()
    iterations = 0
    for iteration in range(max_steps):
        attacked = attacked.detach().requires_grad_(True)
        probability = model(attacked)[:, label_index]
        loss = functional.binary_cross_entropy(probability, target)
        gradient = torch.autograd.grad(loss, attacked)[0] * attack_mask
        candidate = attacked - step_size * gradient.sign()
        candidate = torch.maximum(
            torch.minimum(candidate, reference + epsilon),
            reference - epsilon,
        )
        attacked = candidate.detach()
        iterations = iteration + 1
        with torch.no_grad():
            current_probability = float(model(attacked)[:, label_index].item())
        if (
            desired_value == 1
            and current_probability >= 0.5
            or desired_value == 0
            and current_probability <= 0.5
        ):
            break

    with torch.no_grad():
        after_probability = float(model(attacked)[:, label_index].item())
    passed = (
        after_probability >= 0.5
        if desired_value == 1
        else after_probability <= 0.5
    )
    return attacked, {
        "before_probability": before_probability,
        "after_probability": after_probability,
        "desired_probability": (
            after_probability if desired_value == 1 else 1.0 - after_probability
        ),
        "iterations": iterations,
        "target_pass": passed,
    }


def perturbation_metrics(
    before: Any,
    after: Any,
    *,
    attack_mask: Any,
    facepart_mask: Any,
    pixel_threshold: float = 1.0 / 255.0,
) -> dict[str, float]:
    """Measure perturbation magnitude and leakage in display pixel space."""

    import torch

    delta = (after - before).detach().abs()
    attack_mask = attack_mask.to(device=delta.device, dtype=delta.dtype)
    facepart_mask = facepart_mask.to(device=delta.device, dtype=delta.dtype)
    pixel_delta = delta.amax(dim=1, keepdim=True)

    def outside_mae(mask):
        outside = 1.0 - mask
        denominator = (outside.sum() * delta.shape[1]).clamp_min(1.0)
        return float((delta * outside).sum().item() / denominator.item())

    return {
        "changed_fraction": float((pixel_delta > pixel_threshold).float().mean().item()),
        "mean_abs_change": float(delta.mean().item()),
        "l2": float(torch.linalg.vector_norm(delta.float()).item()),
        "linf": float(delta.max().item()),
        "outside_attack_mae": outside_mae(attack_mask),
        "outside_facepart_mae": outside_mae(facepart_mask),
    }


def load_rgb_tensor(path: str | Path, *, size: int, device: str):
    import torch
    import torch.nn.functional as functional
    from PIL import Image

    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = functional.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return tensor.to(device=device, dtype=torch.float32)


def load_binary_mask(path: str | Path, *, size: int, device: str):
    import torch
    from PIL import Image

    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy((array >= 0.5).astype(np.float32)).view(
        1,
        1,
        size,
        size,
    ).to(device)


def save_rgb_tensor(path: Path, tensor: Any) -> None:
    from PIL import Image

    array = (
        tensor.detach()[0]
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_gray_array(path: Path, array: np.ndarray) -> None:
    from PIL import Image

    values = np.clip(array, 0, 1)
    Image.fromarray((values * 255).round().astype(np.uint8), mode="L").save(path)


def identity_cosine(
    source: Any,
    output: Any,
    *,
    model: Any,
    detector: Any,
) -> float:
    import torch

    from cci_diff.identity.facenet import (
        detect_largest_face_box,
        fixed_face_crop,
        standardize_face,
    )

    box = detect_largest_face_box(detector, source)
    with torch.no_grad():
        source_embedding = torch.nn.functional.normalize(
            model(standardize_face(fixed_face_crop(source, box))),
            dim=1,
        )
        output_embedding = torch.nn.functional.normalize(
            model(standardize_face(fixed_face_crop(output, box))),
            dim=1,
        )
    return float(
        torch.nn.functional.cosine_similarity(
            source_embedding,
            output_embedding,
            dim=1,
        ).mean().item()
    )


def create_comparison_sheet(
    path: Path,
    *,
    source_path: Path,
    input_path: Path,
    facepart_mask_path: Path,
    facepart_output_path: Path,
    gradcam_mask_path: Path,
    gradcam_output_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    entries = (
        ("source", source_path),
        ("BLD input", input_path),
        ("FacePart mask", facepart_mask_path),
        ("FacePart PGD", facepart_output_path),
        ("Grad-CAM++ mask", gradcam_mask_path),
        ("Grad-CAM++ PGD", gradcam_output_path),
    )
    tiles = []
    for label, image_path in entries:
        image = Image.open(image_path).convert("RGB").resize((256, 256))
        tile = Image.new("RGB", (256, 282), "white")
        tile.paste(image, (0, 0))
        ImageDraw.Draw(tile).text((6, 262), label, fill="black")
        tiles.append(tile)
    sheet = Image.new("RGB", (3 * 256, 2 * 282), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 3) * 256, (index // 3) * 282))
    sheet.save(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for mask_type in ("facepart", "gradcam_pp"):
        selected = [row for row in rows if row["mask_type"] == mask_type]
        summary[mask_type] = {
            "count": len(selected),
            "target_passes": sum(bool(row["target_pass"]) for row in selected),
            "target_pass_rate": (
                sum(bool(row["target_pass"]) for row in selected) / len(selected)
            ),
            "mean_desired_probability": float(
                np.mean([row["desired_probability"] for row in selected])
            ),
            "mean_mask_fraction": float(
                np.mean([row["mask_fraction"] for row in selected])
            ),
            "mean_changed_fraction": float(
                np.mean([row["changed_fraction"] for row in selected])
            ),
            "mean_abs_change": float(
                np.mean([row["mean_abs_change"] for row in selected])
            ),
            "mean_iterations": float(
                np.mean([row["iterations"] for row in selected])
            ),
            "mean_outside_facepart_mae": float(
                np.mean([row["outside_facepart_mae"] for row in selected])
            ),
            "mean_identity_cosine": float(
                np.mean([row["identity_cosine"] for row in selected])
            ),
        }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        default="outputs/clean_cci_component_ablation_10",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_ids", nargs="+", type=int, required=True)
    parser.add_argument(
        "--classifier_path",
        default="models/resnet50_multilabel_model.pth",
    )
    parser.add_argument(
        "--identity_model_path",
        default="models/facenet_vggface2.ts",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--label_index", type=int, default=31)
    parser.add_argument("--desired_value", type=int, choices=[0, 1], default=0)
    parser.add_argument("--gradcam_threshold", type=float, default=0.4)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--step_size", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=500)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from cci_diff.classifiers.celeba_resnet50 import load_celeba_resnet50
    from cci_diff.identity.facenet import build_face_detector, load_facenet_identity

    if len(set(args.sample_ids)) != len(args.sample_ids):
        raise ValueError("sample_ids must be unique")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    model = load_celeba_resnet50(
        args.classifier_path,
        device=args.device,
        dtype=torch.float32,
    )
    identity_model = load_facenet_identity(
        args.identity_model_path,
        device=args.device,
    )
    detector = build_face_detector()
    rows = []

    for sample_id in args.sample_ids:
        sample_name = f"{sample_id:05d}"
        run_dir = Path(args.input_root) / "smile" / sample_name / "A9"
        audit_path = run_dir / "audit.json"
        input_path = run_dir / "sd2_bld_grid.png"
        if not audit_path.is_file() or not input_path.is_file():
            raise FileNotFoundError(f"Incomplete A9 sample: {run_dir}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        source_path = Path(audit["cci"]["source_image"])
        semantic_path = Path(audit["cci"]["mask_artifacts"]["semantic_path"])

        source = load_rgb_tensor(
            source_path,
            size=args.image_size,
            device=args.device,
        )
        generated = load_rgb_tensor(
            input_path,
            size=args.image_size,
            device=args.device,
        )
        source_normalized = normalize_images(source)
        generated_normalized = normalize_images(generated)
        facepart_mask = load_binary_mask(
            semantic_path,
            size=args.image_size,
            device=args.device,
        )
        with torch.no_grad():
            source_probability = float(
                model(source_normalized)[:, args.label_index].item()
            )
            generated_probability = float(
                model(generated_normalized)[:, args.label_index].item()
            )
        saliency = gradcam_pp_saliency(
            model,
            source_normalized,
            label_index=args.label_index,
            original_present=source_probability >= 0.5,
        )
        gradcam_mask_np = binary_mask_from_saliency(
            saliency,
            threshold=args.gradcam_threshold,
        )
        gradcam_mask = torch.from_numpy(gradcam_mask_np).view(
            1,
            1,
            args.image_size,
            args.image_size,
        ).to(args.device)

        sample_output = output_root / sample_name
        sample_output.mkdir(parents=True, exist_ok=True)
        facepart_mask_path = sample_output / "facepart_mask.png"
        gradcam_heatmap_path = sample_output / "gradcam_pp_heatmap.png"
        gradcam_mask_path = sample_output / "gradcam_pp_mask.png"
        save_gray_array(
            facepart_mask_path,
            facepart_mask[0, 0].cpu().numpy(),
        )
        save_gray_array(gradcam_heatmap_path, saliency)
        save_gray_array(gradcam_mask_path, gradcam_mask_np)

        output_paths = {}
        for mask_type, attack_mask in (
            ("facepart", facepart_mask),
            ("gradcam_pp", gradcam_mask),
        ):
            attacked_normalized, attack_record = targeted_masked_pgd(
                model,
                generated_normalized,
                attack_mask,
                label_index=args.label_index,
                desired_value=args.desired_value,
                epsilon=args.epsilon,
                step_size=args.step_size,
                max_steps=args.max_steps,
            )
            attacked = unnormalize_images(attacked_normalized).clamp(0, 1)
            output_path = sample_output / f"{mask_type}_pgd.png"
            save_rgb_tensor(output_path, attacked)
            saved_attacked = load_rgb_tensor(
                output_path,
                size=args.image_size,
                device=args.device,
            )
            with torch.no_grad():
                saved_probability = float(
                    model(normalize_images(saved_attacked))[
                        :, args.label_index
                    ].item()
                )
            saved_desired_probability = (
                saved_probability
                if args.desired_value == 1
                else 1.0 - saved_probability
            )
            saved_target_pass = (
                saved_probability >= 0.5
                if args.desired_value == 1
                else saved_probability <= 0.5
            )
            output_paths[mask_type] = output_path
            metrics = perturbation_metrics(
                generated,
                saved_attacked,
                attack_mask=attack_mask,
                facepart_mask=facepart_mask,
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "mask_type": mask_type,
                    "source_smile_probability": source_probability,
                    "generated_smile_probability": generated_probability,
                    **attack_record,
                    "saved_smile_probability": saved_probability,
                    "desired_probability": saved_desired_probability,
                    "target_pass": saved_target_pass,
                    "mask_fraction": float(attack_mask.float().mean().item()),
                    **metrics,
                    "identity_cosine": identity_cosine(
                        source,
                        saved_attacked,
                        model=identity_model,
                        detector=detector,
                    ),
                    "input_path": str(input_path),
                    "mask_path": str(
                        facepart_mask_path
                        if mask_type == "facepart"
                        else gradcam_mask_path
                    ),
                    "output_path": str(output_path),
                }
            )

        create_comparison_sheet(
            sample_output / "comparison.jpg",
            source_path=source_path,
            input_path=input_path,
            facepart_mask_path=facepart_mask_path,
            facepart_output_path=output_paths["facepart"],
            gradcam_mask_path=gradcam_mask_path,
            gradcam_output_path=output_paths["gradcam_pp"],
        )

    summary = {
        "configuration": {
            "input_root": args.input_root,
            "sample_ids": args.sample_ids,
            "classifier_path": args.classifier_path,
            "identity_model_path": args.identity_model_path,
            "label_index": args.label_index,
            "desired_value": args.desired_value,
            "gradcam_method": "GradCAMPlusPlus",
            "target_layer": "base_model.layer4[-1]",
            "gradcam_threshold": args.gradcam_threshold,
            "epsilon_normalized": args.epsilon,
            "step_size_normalized": args.step_size,
            "max_steps": args.max_steps,
        },
        "masks": aggregate_rows(rows),
    }
    write_csv(output_root / "results.csv", rows)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
