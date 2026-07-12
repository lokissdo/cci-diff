#!/usr/bin/env python3
"""Build a binary edit mask from a CelebAMask-HQ label map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


ATTRIBUTE_ALIASES = {
    "smile": "Smiling",
    "smiling": "Smiling",
}

PART_LABEL_IDS = {
    "background": 0,
    "skin": 1,
    "nose": 2,
    "eye_glasses": 3,
    "eye_g": 3,
    "left_eye": 4,
    "l_eye": 4,
    "right_eye": 5,
    "r_eye": 5,
    "left_brow": 6,
    "l_brow": 6,
    "right_brow": 7,
    "r_brow": 7,
    "left_ear": 8,
    "l_ear": 8,
    "right_ear": 9,
    "r_ear": 9,
    "mouth": 10,
    "upper_lip": 11,
    "u_lip": 11,
    "lower_lip": 12,
    "l_lip": 12,
    "hair": 13,
    "hat": 14,
    "ear_ring": 15,
    "ear_r": 15,
    "necklace": 16,
    "neck_l": 16,
    "neck": 17,
    "cloth": 18,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cci_config", required=True)
    parser.add_argument("--label_map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--parts",
        nargs="+",
        default=None,
        help="CelebAMask part names to mask, e.g. hair or mouth upper_lip lower_lip.",
    )
    parser.add_argument(
        "--attribute2parts",
        default="../thesis_2025/face_parts_retrieval/CelebAMask-HQ/face_parsing/attribute2parts.json",
    )
    return parser


def make_mask_from_config(
    *,
    config_path: str | Path,
    label_map_path: str | Path,
    attribute2parts_path: str | Path,
    output_path: str | Path,
    parts_override: list[str] | None = None,
) -> list[int]:
    config = _load_json(config_path)
    attribute2parts = _load_json(attribute2parts_path)
    target_concept = str(config["target_concept"])
    parts = parts_override or resolve_parts(target_concept, attribute2parts)
    selected_ids = sorted({label_id_for_part(part) for part in parts})

    label_map = np.array(Image.open(label_map_path).convert("L"))
    mask = np.isin(label_map, selected_ids).astype(np.uint8) * 255
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_path)
    return selected_ids


def resolve_parts(target_concept: str, attribute2parts: dict[str, list[str]]) -> list[str]:
    attribute_name = resolve_attribute_name(target_concept, attribute2parts)
    if attribute_name in attribute2parts:
        return attribute2parts[attribute_name]
    if _normalize(target_concept) in PART_LABEL_IDS:
        return [target_concept]
    raise SystemExit(
        f"Cannot map target concept {target_concept!r} to CelebAMask parts."
    )


def resolve_attribute_name(
    target_concept: str,
    attribute2parts: dict[str, list[str]],
) -> str:
    normalized_target = _normalize(target_concept)
    if normalized_target in ATTRIBUTE_ALIASES:
        return ATTRIBUTE_ALIASES[normalized_target]
    for attribute_name in attribute2parts:
        if _normalize(attribute_name) == normalized_target:
            return attribute_name
    return target_concept


def label_id_for_part(part: str) -> int:
    normalized_part = _normalize(part)
    if normalized_part not in PART_LABEL_IDS:
        raise SystemExit(f"Unknown CelebAMask part {part!r}.")
    return PART_LABEL_IDS[normalized_part]


def _load_json(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _normalize(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def main() -> int:
    args = build_arg_parser().parse_args()
    selected_ids = make_mask_from_config(
        config_path=args.cci_config,
        label_map_path=args.label_map,
        attribute2parts_path=args.attribute2parts,
        output_path=args.output,
        parts_override=args.parts,
    )
    print(f"{args.output}: labels {','.join(str(label_id) for label_id in selected_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
