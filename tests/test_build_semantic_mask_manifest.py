import json

from scripts.build_semantic_mask_manifest import build_semantic_mask_manifest
from tests.test_run_individual_region_cci import make_file_tree


def test_build_manifest_binds_every_verified_component(tmp_path):
    _, influence, _, mask_root, _, _, _ = make_file_tree(
        tmp_path, sample_ids=(0, 1)
    )

    payload = build_semantic_mask_manifest(influence, [1, 0], mask_root)

    assert payload["sample_ids"] == [0, 1]
    assert payload["verified_regions"] == ["lower_lip", "mouth"]
    assert set(payload["sample_masks"]) == {"0", "1"}
    assert all(
        set(regions) == {"lower_lip", "mouth"}
        for regions in payload["sample_masks"].values()
    )
    json.dumps(payload, allow_nan=False)
