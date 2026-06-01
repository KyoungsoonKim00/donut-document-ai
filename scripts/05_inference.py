"""CLI: run a fine-tuned Donut model over a folder of images."""
import argparse

from donut_docai import load_config
from donut_docai.inference import DonutPredictor


def main():
    ap = argparse.ArgumentParser(description="Run Donut inference on images.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", help="Override model path (default: paths.output_dir)")
    ap.add_argument("--image-dir", help="Override paths.image_dir")
    ap.add_argument("--out", help="Override paths.prediction_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    predictor = DonutPredictor(cfg, model_path=args.model)
    n = predictor.predict_folder(
        args.image_dir or cfg.paths.image_dir,
        args.out or cfg.paths.prediction_dir,
    )
    print(f"Done. {n} predictions parsed.")


if __name__ == "__main__":
    main()
