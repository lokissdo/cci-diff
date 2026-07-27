from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from cci_diff.post_attack import (
    gradcam_pp_saliency,
    soft_anatomical_mask,
    targeted_smooth_boundary_attack,
)

try:
    from scripts.compare_attack_masks import (
        identity_cosine,
        load_binary_mask,
        load_rgb_tensor,
        normalize_images,
        perturbation_metrics,
        save_gray_array,
        save_rgb_tensor,
        unnormalize_images,
        write_csv,
    )
except ModuleNotFoundError:
    from compare_attack_masks import (
        identity_cosine,
        load_binary_mask,
        load_rgb_tensor,
        normalize_images,
        perturbation_metrics,
        save_gray_array,
        save_rgb_tensor,
        unnormalize_images,
        write_csv,
    )


def residual_total_variation(before: Any, after: Any, facepart_mask: Any) -> float:
    """Measure channel-normalized residual variation on mask-interior edges."""

    import torch

    mask = facepart_mask.to(device=before.device, dtype=before.dtype)
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    residual = (after - before) * mask
    horizontal_weights = mask[:, :, :, :-1] * mask[:, :, :, 1:]
    vertical_weights = mask[:, :, :-1, :] * mask[:, :, 1:, :]
    horizontal = (
        torch.abs(residual[:, :, :, 1:] - residual[:, :, :, :-1])
        * horizontal_weights
    ).sum()
    vertical = (
        torch.abs(residual[:, :, 1:, :] - residual[:, :, :-1, :])
        * vertical_weights
    ).sum()
    edge_count = horizontal_weights.sum() + vertical_weights.sum()
    denominator = (before.shape[1] * edge_count).clamp_min(1)
    return float(((horizontal + vertical) / denominator).item())


def aggregate_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = {}
    for method in ("without_attack", "with_attack"):
        selected = [row for row in rows if row["method"] == method]
        if not selected:
            raise ValueError(f"no rows for method {method}")
        methods[method] = {
            "count": len(selected),
            "target_passes": sum(bool(row["target_pass"]) for row in selected),
            "target_pass_rate": float(
                np.mean([bool(row["target_pass"]) for row in selected])
            ),
            "mean_smile_probability": float(
                np.mean([row["smile_probability"] for row in selected])
            ),
            "mean_desired_probability": float(
                np.mean([row["desired_probability"] for row in selected])
            ),
            "mean_identity_cosine": float(
                np.mean([row["identity_cosine"] for row in selected])
            ),
            "mean_abs_change": float(
                np.mean([row["mean_abs_change"] for row in selected])
            ),
            "mean_linf": float(np.mean([row["linf"] for row in selected])),
            "mean_changed_fraction": float(
                np.mean([row["changed_fraction"] for row in selected])
            ),
            "mean_residual_tv": float(
                np.mean([row["residual_tv"] for row in selected])
            ),
            "mean_outside_facepart_mae": float(
                np.mean([row["outside_facepart_mae"] for row in selected])
            ),
            "mean_iterations": float(
                np.mean([row["iterations"] for row in selected])
            ),
            "mean_boundary_iterations": float(
                np.mean([row["boundary_iterations"] for row in selected])
            ),
        }

    before_by_id = {
        int(row["sample_id"]): row
        for row in rows
        if row["method"] == "without_attack"
    }
    after_by_id = {
        int(row["sample_id"]): row
        for row in rows
        if row["method"] == "with_attack"
    }
    if before_by_id.keys() != after_by_id.keys():
        raise ValueError("before and after sample IDs must match")
    pairs = [(before_by_id[key], after_by_id[key]) for key in sorted(before_by_id)]
    attempted = [
        (before, after)
        for before, after in pairs
        if not bool(before["target_pass"])
    ]

    def attempted_mean(key: str, *, delta: bool = False) -> float:
        if not attempted:
            return 0.0
        if delta:
            values = [after[key] - before[key] for before, after in attempted]
        else:
            values = [after[key] for _, after in attempted]
        return float(np.mean(values))

    paired = {
        "count": len(pairs),
        "new_target_passes": sum(
            not bool(before["target_pass"]) and bool(after["target_pass"])
            for before, after in pairs
        ),
        "lost_target_passes": sum(
            bool(before["target_pass"]) and not bool(after["target_pass"])
            for before, after in pairs
        ),
        "target_pass_rate_delta": (
            methods["with_attack"]["target_pass_rate"]
            - methods["without_attack"]["target_pass_rate"]
        ),
        "desired_probability_delta": float(
            np.mean(
                [
                    after["desired_probability"] - before["desired_probability"]
                    for before, after in pairs
                ]
            )
        ),
        "identity_delta": float(
            np.mean(
                [
                    after["identity_cosine"] - before["identity_cosine"]
                    for before, after in pairs
                ]
            )
        ),
        "mean_abs_change": methods["with_attack"]["mean_abs_change"],
        "mean_linf": methods["with_attack"]["mean_linf"],
        "mean_changed_fraction": methods["with_attack"][
            "mean_changed_fraction"
        ],
        "mean_residual_tv": methods["with_attack"]["mean_residual_tv"],
        "attempted_count": len(attempted),
        "attempted_target_passes": sum(
            bool(after["target_pass"]) for _, after in attempted
        ),
        "attempted_target_pass_rate": (
            float(np.mean([bool(after["target_pass"]) for _, after in attempted]))
            if attempted
            else 0.0
        ),
        "attempted_desired_probability_delta": attempted_mean(
            "desired_probability",
            delta=True,
        ),
        "attempted_identity_delta": attempted_mean(
            "identity_cosine",
            delta=True,
        ),
        "attempted_mean_abs_change": attempted_mean("mean_abs_change"),
        "attempted_mean_linf": attempted_mean("linf"),
        "attempted_mean_changed_fraction": attempted_mean("changed_fraction"),
        "attempted_mean_residual_tv": attempted_mean("residual_tv"),
    }
    return {
        "without_attack": methods["without_attack"],
        "with_attack": methods["with_attack"],
        "paired": paired,
    }


def _target_pass(probability: float, desired_value: int) -> bool:
    return probability >= 0.5 if desired_value == 1 else probability <= 0.5


def _desired_probability(probability: float, desired_value: int) -> float:
    return probability if desired_value == 1 else 1.0 - probability


def _mouth_crop(image_path: Path, mask: Any, *, size: int = 512):
    from PIL import Image

    values = mask[0, 0].detach().cpu().numpy()
    rows, columns = np.where(values > 0)
    image = Image.open(image_path).convert("RGB")
    if len(rows) == 0:
        return image.resize((size, size))
    padding = max(8, int(0.05 * max(image.size)))
    left = max(0, int(columns.min()) - padding)
    top = max(0, int(rows.min()) - padding)
    right = min(image.width, int(columns.max()) + padding + 1)
    bottom = min(image.height, int(rows.max()) + padding + 1)
    return image.crop((left, top, right, bottom)).resize((size, size))


def create_comparison_sheet(
    path: Path,
    *,
    source_path: Path,
    baseline_path: Path,
    soft_mask_path: Path,
    attacked_path: Path,
    residual_path: Path,
    facepart_mask: Any,
) -> None:
    from PIL import Image, ImageDraw

    entries = [
        ("source", Image.open(source_path).convert("RGB")),
        ("CCI-BLD without attack", Image.open(baseline_path).convert("RGB")),
        ("soft FacePart x Grad-CAM++", Image.open(soft_mask_path).convert("RGB")),
        ("with smooth boundary attack", Image.open(attacked_path).convert("RGB")),
        ("absolute residual x16", Image.open(residual_path).convert("RGB")),
        ("attacked mouth crop", _mouth_crop(attacked_path, facepart_mask)),
    ]
    tiles = []
    for label, image in entries:
        image = image.resize((256, 256))
        tile = Image.new("RGB", (256, 282), "white")
        tile.paste(image, (0, 0))
        ImageDraw.Draw(tile).text((6, 262), label, fill="black")
        tiles.append(tile)
    sheet = Image.new("RGB", (3 * 256, 2 * 282), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 3) * 256, (index // 3) * 282))
    sheet.save(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        default="outputs/clean_cci_component_ablation_10",
    )
    parser.add_argument("--variant", default="A9")
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
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--step_size", type=float, default=0.005)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--decision_threshold", type=float, default=0.5)
    parser.add_argument("--boundary_margin", type=float, default=0.01)
    parser.add_argument("--boundary_steps", type=int, default=16)
    parser.add_argument("--gaussian_kernel_size", type=int, default=5)
    parser.add_argument("--gaussian_sigma", type=float, default=1.0)
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
    rows: list[dict[str, Any]] = []

    for sample_id in args.sample_ids:
        sample_name = f"{sample_id:05d}"
        run_dir = (
            Path(args.input_root)
            / "smile"
            / sample_name
            / args.variant
            / "candidates"
            / "x4_y4_f3"
        )
        if not run_dir.is_dir():
            run_dir = (
                Path(args.input_root)
                / "smile"
                / sample_name
                / args.variant
            )
        audit_path = run_dir / "audit.json"
        baseline_path = run_dir / "sd2_bld_grid.png"
        if not audit_path.is_file() or not baseline_path.is_file():
            raise FileNotFoundError(
                f"Incomplete {args.variant} sample: {run_dir}"
            )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        source_path = Path(audit["cci"]["source_image"])
        semantic_path = Path(audit["cci"]["mask_artifacts"]["semantic_path"])

        source = load_rgb_tensor(
            source_path,
            size=args.image_size,
            device=args.device,
        )
        baseline = load_rgb_tensor(
            baseline_path,
            size=args.image_size,
            device=args.device,
        )
        facepart_mask = load_binary_mask(
            semantic_path,
            size=args.image_size,
            device=args.device,
        )
        source_normalized = normalize_images(source)
        baseline_normalized = normalize_images(baseline)
        with torch.no_grad():
            source_probability = float(
                model(source_normalized)[:, args.label_index].item()
            )
            baseline_probability = float(
                model(baseline_normalized)[:, args.label_index].item()
            )
        saliency = gradcam_pp_saliency(
            model,
            source_normalized,
            label_index=args.label_index,
            original_present=source_probability >= 0.5,
        )
        saliency_tensor = torch.from_numpy(saliency).view(
            1,
            1,
            args.image_size,
            args.image_size,
        ).to(device=args.device, dtype=baseline.dtype)
        soft_mask = soft_anatomical_mask(facepart_mask, saliency_tensor)

        attacked_normalized, attack_record = targeted_smooth_boundary_attack(
            model,
            baseline_normalized,
            soft_mask,
            label_index=args.label_index,
            desired_value=args.desired_value,
            epsilon=args.epsilon,
            step_size=args.step_size,
            max_steps=args.max_steps,
            decision_threshold=args.decision_threshold,
            boundary_margin=args.boundary_margin,
            boundary_steps=args.boundary_steps,
            kernel_size=args.gaussian_kernel_size,
            sigma=args.gaussian_sigma,
        )
        attacked = unnormalize_images(attacked_normalized).clamp(0, 1)

        sample_output = output_root / sample_name
        sample_output.mkdir(parents=True, exist_ok=True)
        facepart_path = sample_output / "facepart_mask.png"
        heatmap_path = sample_output / "gradcam_pp_heatmap.png"
        soft_mask_path = sample_output / "soft_anatomical_mask.png"
        attacked_path = sample_output / "smooth_boundary_attack.png"
        residual_path = sample_output / "absolute_residual_x16.png"
        save_gray_array(
            facepart_path,
            facepart_mask[0, 0].detach().cpu().numpy(),
        )
        save_gray_array(heatmap_path, saliency)
        save_gray_array(
            soft_mask_path,
            soft_mask[0, 0].detach().cpu().numpy(),
        )
        save_rgb_tensor(attacked_path, attacked)
        saved_attacked = load_rgb_tensor(
            attacked_path,
            size=args.image_size,
            device=args.device,
        )
        residual = torch.abs(saved_attacked - baseline)
        save_rgb_tensor(residual_path, (16.0 * residual).clamp(0, 1))

        with torch.no_grad():
            saved_probability = float(
                model(normalize_images(saved_attacked))[
                    :, args.label_index
                ].item()
            )
        baseline_identity = identity_cosine(
            source,
            baseline,
            model=identity_model,
            detector=detector,
        )
        attacked_identity = identity_cosine(
            source,
            saved_attacked,
            model=identity_model,
            detector=detector,
        )
        zero_metrics = {
            "mean_abs_change": 0.0,
            "l2": 0.0,
            "linf": 0.0,
            "changed_fraction": 0.0,
            "outside_attack_mae": 0.0,
            "outside_facepart_mae": 0.0,
        }
        attack_metrics = perturbation_metrics(
            baseline,
            saved_attacked,
            attack_mask=(soft_mask > 0).to(dtype=soft_mask.dtype),
            facepart_mask=facepart_mask,
        )
        common = {
            "sample_id": sample_id,
            "source_smile_probability": source_probability,
            "input_path": str(baseline_path),
            "source_path": str(source_path),
            "facepart_mask_path": str(facepart_path),
            "gradcam_heatmap_path": str(heatmap_path),
            "soft_mask_path": str(soft_mask_path),
        }
        rows.append(
            {
                **common,
                "method": "without_attack",
                "smile_probability": baseline_probability,
                "desired_probability": _desired_probability(
                    baseline_probability,
                    args.desired_value,
                ),
                "target_pass": _target_pass(
                    baseline_probability,
                    args.desired_value,
                ),
                "identity_cosine": baseline_identity,
                **zero_metrics,
                "residual_tv": 0.0,
                "iterations": 0,
                "boundary_iterations": 0,
                "internal_smile_probability": baseline_probability,
                "margin_pass": _target_pass(
                    baseline_probability,
                    args.desired_value,
                ),
                "output_path": str(baseline_path),
            }
        )
        rows.append(
            {
                **common,
                "method": "with_attack",
                "smile_probability": saved_probability,
                "desired_probability": _desired_probability(
                    saved_probability,
                    args.desired_value,
                ),
                "target_pass": _target_pass(
                    saved_probability,
                    args.desired_value,
                ),
                "identity_cosine": attacked_identity,
                **attack_metrics,
                "residual_tv": residual_total_variation(
                    baseline,
                    saved_attacked,
                    facepart_mask,
                ),
                "iterations": attack_record["iterations"],
                "boundary_iterations": attack_record["boundary_iterations"],
                "internal_smile_probability": attack_record[
                    "after_probability"
                ],
                "margin_pass": attack_record["margin_pass"],
                "output_path": str(attacked_path),
            }
        )
        create_comparison_sheet(
            sample_output / "comparison.jpg",
            source_path=source_path,
            baseline_path=baseline_path,
            soft_mask_path=soft_mask_path,
            attacked_path=attacked_path,
            residual_path=residual_path,
            facepart_mask=facepart_mask,
        )

    write_csv(output_root / "results.csv", rows)
    summary = {
        "configuration": {
            "input_root": args.input_root,
            "variant": args.variant,
            "sample_ids": args.sample_ids,
            "classifier_path": args.classifier_path,
            "identity_model_path": args.identity_model_path,
            "label_index": args.label_index,
            "desired_value": args.desired_value,
            "epsilon_normalized": args.epsilon,
            "step_size_normalized": args.step_size,
            "max_steps": args.max_steps,
            "decision_threshold": args.decision_threshold,
            "boundary_margin": args.boundary_margin,
            "boundary_steps": args.boundary_steps,
            "gaussian_kernel_size": args.gaussian_kernel_size,
            "gaussian_sigma": args.gaussian_sigma,
        },
        "comparison": aggregate_comparison(rows),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    summary = run(build_arg_parser().parse_args())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
