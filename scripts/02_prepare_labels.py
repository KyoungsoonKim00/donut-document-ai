"""CLI: seed the labeling workbook with PDF filenames (column A)."""
import argparse

from donut_docai import load_config
from donut_docai.data import write_filenames_to_excel


def main():
    ap = argparse.ArgumentParser(description="Write PDF filenames into the label Excel.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pdf-dir", help="Override paths.pdf_dir")
    ap.add_argument("--excel", help="Override paths.excel_path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    write_filenames_to_excel(
        args.pdf_dir or cfg.paths.pdf_dir,
        args.excel or cfg.paths.excel_path,
    )


if __name__ == "__main__":
    main()
