"""Bridge helpers for reusing the ESWA thesis SD2 script."""

from __future__ import annotations

from pathlib import Path

from cci_diff.prompts import ConceptPrompt


def build_legacy_sd2_command(
    *,
    legacy_script: Path,
    init_image: Path,
    mask: Path,
    classifier_path: Path,
    output_dir: Path,
    prompt: ConceptPrompt,
    batch_size: int = 4,
    device: str = "cuda",
    lora_path: Path | None = None,
) -> list[str]:
    """Build a subprocess command for the old `text_editing_SD2.py` script."""

    prompt_text = f"{prompt.positive}, negative: {prompt.negative}"
    command = [
        "python3",
        str(legacy_script),
        "--prompt",
        prompt_text,
        "--init_image",
        str(init_image),
        "--mask",
        str(mask),
        "--classifier_path",
        str(classifier_path),
        "--batch_size",
        str(batch_size),
        "--device",
        device,
        "--output_path",
        str(output_dir / "res.jpg"),
        "--output_path_1",
        str(output_dir / "res_1.jpg"),
        "--output_path_2",
        str(output_dir / "res_2.jpg"),
        "--output_path_3",
        str(output_dir / "res_3.jpg"),
        "--output_path_4",
        str(output_dir / "res_4.jpg"),
    ]
    if lora_path is not None:
        command.extend(["--lora_path", str(lora_path)])
    return command
