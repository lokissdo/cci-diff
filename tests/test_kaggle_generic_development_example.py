from pathlib import Path

from examples.kaggle_generic_development import (
    KagglePaths,
    build_kaggle_command,
)


def fake_kaggle_paths():
    return KagglePaths(
        repo_root=Path("/kaggle/working/cci-diff"),
        image_root=Path("/kaggle/input/celeba/CelebA-HQ-img"),
        mask_root=Path("/kaggle/input/celeba/CelebAMask-HQ-mask-anno"),
        model_path=Path("/kaggle/input/sd2/sd2-1-base"),
        classifier_path=Path("/kaggle/input/models/classifier.pth"),
        identity_model_path=Path("/kaggle/input/models/facenet.ts"),
        template_graph=Path("/kaggle/input/config/graph.json"),
        generation_policy=Path("/kaggle/input/config/a11-policy.json"),
        eligible_ids_manifest=Path("/kaggle/input/config/candidate-ids.json"),
        evaluation_ids_manifest=Path("/kaggle/input/config/evaluation-ids.json"),
        working_root=Path("/kaggle/working/cci-generic"),
    )


def test_kaggle_example_uses_shared_cli_and_only_data_size_parameter():
    command = build_kaggle_command(
        data_size=30, paths=fake_kaggle_paths()
    )

    assert command[1].endswith("run_generic_region_development.py")
    assert command[command.index("--data_size") + 1] == "30"
    assert "/kaggle/working" in " ".join(command)
    assert "smoke" not in " ".join(command).lower()
    assert command[command.index("--device") + 1] == "auto"
