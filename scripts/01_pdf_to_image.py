"""CLI: convert PDFs in paths.pdf_dir to PNGs in paths.image_dir."""
import argparse

from donut_docai import load_config
from donut_docai.data import convert_folder


def main():
    ap = argparse.ArgumentParser(description="Convert PDFs to PNG images.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pdf-dir", help="Override paths.pdf_dir")
    ap.add_argument("--image-dir", help="Override paths.image_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pdf_dir = args.pdf_dir or cfg.paths.pdf_dir
    image_dir = args.image_dir or cfg.paths.image_dir

    n = convert_folder(
        pdf_dir, image_dir,
        dpi=cfg.pdf.dpi, first_page=cfg.pdf.first_page, last_page=cfg.pdf.last_page,
    )
    print(f"Done. {n} images written to {image_dir}")


if __name__ == "__main__":
    main()
