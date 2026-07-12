import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.make_celebamask_mask import make_mask_from_config


class TestMakeCelebAMaskMask(unittest.TestCase):
    def test_make_mask_from_smile_config_uses_mouth_and_lip_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "cci.json"
            parts_path = tmp_path / "attribute2parts.json"
            label_map_path = tmp_path / "labels.png"
            output_path = tmp_path / "mask.png"

            config_path.write_text(
                json.dumps({"target_concept": "smile", "desired_value": 1}),
                encoding="utf-8",
            )
            parts_path.write_text(
                json.dumps({"Smiling": ["mouth", "upper_lip", "lower_lip"]}),
                encoding="utf-8",
            )
            labels = np.array(
                [
                    [0, 10, 11],
                    [12, 13, 1],
                ],
                dtype=np.uint8,
            )
            Image.fromarray(labels).save(label_map_path)

            selected_ids = make_mask_from_config(
                config_path=config_path,
                label_map_path=label_map_path,
                attribute2parts_path=parts_path,
                output_path=output_path,
            )

            mask = np.array(Image.open(output_path))
            self.assertEqual(selected_ids, [10, 11, 12])
            np.testing.assert_array_equal(
                mask,
                np.array(
                    [
                        [0, 255, 255],
                        [255, 0, 0],
                    ],
                    dtype=np.uint8,
                ),
            )

    def test_make_mask_uses_explicit_part_override_instead_of_config_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "cci.json"
            parts_path = tmp_path / "attribute2parts.json"
            label_map_path = tmp_path / "labels.png"
            output_path = tmp_path / "mask.png"

            config_path.write_text(
                json.dumps({"target_concept": "smile", "desired_value": 1}),
                encoding="utf-8",
            )
            parts_path.write_text(
                json.dumps({"Smiling": ["mouth", "upper_lip", "lower_lip"]}),
                encoding="utf-8",
            )
            labels = np.array(
                [
                    [0, 10, 13],
                    [13, 11, 12],
                ],
                dtype=np.uint8,
            )
            Image.fromarray(labels).save(label_map_path)

            selected_ids = make_mask_from_config(
                config_path=config_path,
                label_map_path=label_map_path,
                attribute2parts_path=parts_path,
                output_path=output_path,
                parts_override=["hair"],
            )

            mask = np.array(Image.open(output_path))
            self.assertEqual(selected_ids, [13])
            np.testing.assert_array_equal(
                mask,
                np.array(
                    [
                        [0, 0, 255],
                        [255, 0, 0],
                    ],
                    dtype=np.uint8,
                ),
            )


if __name__ == "__main__":
    unittest.main()
