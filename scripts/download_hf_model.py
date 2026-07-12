#!/usr/bin/env python3
"""Download a Hugging Face model snapshot for offline/local inference."""

from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-2-base")
    parser.add_argument("--local_dir", default="checkpoints/sd2-base")
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. You can also run `huggingface-cli login`.",
    )
    return parser


def download_model(args: argparse.Namespace) -> str:
    validate_model_id(args.model_id)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Model download requires huggingface_hub. Run: "
            "pip install -r requirements.txt"
        ) from exc

    try:
        return snapshot_download(
            repo_id=args.model_id,
            local_dir=args.local_dir,
            revision=args.revision,
            token=args.token,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        raise SystemExit(format_download_error(args, exc)) from exc


def validate_model_id(model_id: str) -> None:
    if model_id == "stabilityai/stable-diffusion-2-1-base":
        raise SystemExit(
            "Model id 'stabilityai/stable-diffusion-2-1-base' does not resolve. "
            "Use 'stabilityai/stable-diffusion-2-base' for the 512px SD2 base "
            "model, or 'stabilityai/stable-diffusion-2-1' if you specifically "
            "want SD 2.1."
        )


def format_download_error(args: argparse.Namespace, exc: Exception) -> str:
    return (
        f"Could not download {args.model_id!r} to {args.local_dir!r}.\n"
        "If Hugging Face says 'Repository Not Found' for a StabilityAI model, "
        "the common causes are gated/private access or a token that cannot see "
        "the repo.\n"
        "Fix:\n"
        "1. Open the model page in a browser and accept access if required.\n"
        "2. Run `huggingface-cli login` with a read token, or pass "
        "`--token hf_...` to this script.\n"
        "3. Re-run the download command.\n"
        f"Original error: {exc}"
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    print(download_model(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
