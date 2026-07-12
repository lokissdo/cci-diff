#!/usr/bin/env python3
"""Run a small CCI-Diff smoke generation."""

from __future__ import annotations

import argparse

from cci_diff.runner import run_diffusion_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cci_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backend", choices=["fake", "diffusers"], default="fake")
    parser.add_argument("--model_id", default="hf-internal-testing/tiny-stable-diffusion-pipe")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch_dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--num_inference_steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_diffusion_smoke(
        config_path=args.cci_config,
        output_dir=args.output_dir,
        backend_name=args.backend,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
    )
    print(result.image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
