"""Upload a fine-tuned Donut model to the Hugging Face Hub.

First-time setup: see docs/huggingface_upload.md.

Only the inference files are uploaded — training checkpoints, optimizer state,
and RNG/scheduler files are excluded (they are large and not needed to run the
model). Large files (pytorch_model.bin) are pushed via Git LFS automatically.
"""
import argparse
import os

from huggingface_hub import HfApi, create_repo


# Training artifacts that are not needed for inference.
IGNORE_PATTERNS = [
    "checkpoint-*/**",
    "checkpoint-*",
    "optimizer.pt",
    "rng_state*.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "*.tmp",
]


def main():
    ap = argparse.ArgumentParser(description="Upload a Donut model to the HF Hub.")
    ap.add_argument("--model-dir", required=True, help="Local fine-tuned model directory")
    ap.add_argument("--repo-id", required=True, help="Target repo, e.g. username/donut-docai")
    ap.add_argument("--private", action="store_true", help="Create a private repo")
    ap.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF write token (or set HF_TOKEN env var, or run `huggingface-cli login`)",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.model_dir):
        raise SystemExit(f"Model dir not found: {args.model_dir}")

    create_repo(args.repo_id, repo_type="model", private=args.private,
                exist_ok=True, token=args.token)

    api = HfApi()
    api.upload_folder(
        folder_path=args.model_dir,
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=IGNORE_PATTERNS,
        token=args.token,
        commit_message="Upload fine-tuned Donut document-AI model",
    )
    print(f"[ok] uploaded to https://huggingface.co/{args.repo_id}")
    print("Load it with:")
    print(f'    DonutPredictor(cfg, model_path="{args.repo_id}")')


if __name__ == "__main__":
    main()
