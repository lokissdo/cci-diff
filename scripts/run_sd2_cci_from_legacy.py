#!/usr/bin/env python3
"""Run or print an ESWA SD2 command built from a CCI config."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from cci_diff.config import load_cci_config
from cci_diff.legacy import build_legacy_sd2_command
from cci_diff.prompts import build_concept_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cci_config", required=True)
    parser.add_argument("--legacy_script", required=True)
    parser.add_argument("--init_image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--classifier_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_cci_config(args.cci_config)
    prompt = build_concept_prompt(config.intervention)
    command = build_legacy_sd2_command(
        legacy_script=Path(args.legacy_script),
        init_image=Path(args.init_image),
        mask=Path(args.mask),
        classifier_path=Path(args.classifier_path),
        output_dir=Path(args.output_dir),
        prompt=prompt,
        batch_size=args.batch_size,
        device=args.device,
        lora_path=Path(args.lora_path) if args.lora_path else None,
    )
    if args.dry_run:
        print(shlex.join(command))
        return 0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
