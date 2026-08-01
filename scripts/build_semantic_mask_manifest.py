#!/usr/bin/env python3
"""Freeze exact semantic-mask bytes for a predeclared selector cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cci_diff.concept_graph import sha256_file  # noqa: E402
from cci_diff.individual_region_selection import (  # noqa: E402
    load_frozen_influence_policy,
)
from cci_diff.region_screening import celebamask_component_path  # noqa: E402


def build_semantic_mask_manifest(
    influence_graph: str | Path,
    sample_ids: list[int] | tuple[int, ...],
    mask_root: str | Path,
) -> dict[str, object]:
    policy = load_frozen_influence_policy(influence_graph)
    ids = tuple(sorted({int(value) for value in sample_ids}))
    if not ids:
        raise ValueError("sample_ids must be non-empty")
    sample_masks = {}
    for sample_id in ids:
        regions = {}
        for region in policy.verified_regions:
            path = celebamask_component_path(mask_root, sample_id, region)
            if not path.is_file():
                raise FileNotFoundError(
                    f"semantic mask not found for {sample_id}/{region}: {path}"
                )
            regions[region] = sha256_file(path)
        sample_masks[str(sample_id)] = regions
    return {
        "version": 1,
        "artifact_type": "semantic_mask_bytes_v1",
        "influence_graph_sha256": policy.graph_sha256,
        "verified_regions": list(policy.verified_regions),
        "sample_ids": list(ids),
        "sample_masks": sample_masks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--influence_graph", required=True)
    parser.add_argument("--sample_ids", nargs="+", type=int, required=True)
    parser.add_argument("--mask_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build_semantic_mask_manifest(
        args.influence_graph, args.sample_ids, args.mask_root
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
