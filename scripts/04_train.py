"""CLI: fine-tune Donut on the prepared dataset."""
import argparse

from donut_docai import load_config
from donut_docai.train import train


def main():
    ap = argparse.ArgumentParser(description="Fine-tune Donut.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset-dir", help="Override paths.dataset_dir")
    ap.add_argument("--output-dir", help="Override paths.output_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dataset_dir:
        cfg.paths.dataset_dir = args.dataset_dir
    if args.output_dir:
        cfg.paths.output_dir = args.output_dir

    train(cfg)


if __name__ == "__main__":
    main()
