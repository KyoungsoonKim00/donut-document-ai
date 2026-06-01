"""CLI: convert the labeled Excel workbook into Donut JSON labels."""
import argparse

from donut_docai import load_config
from donut_docai.data import excel_to_donut_json


def main():
    ap = argparse.ArgumentParser(description="Convert labeled Excel to Donut JSON.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--excel", help="Override paths.excel_path")
    ap.add_argument("--out", help="Override paths.dataset_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    excel_to_donut_json(
        args.excel or cfg.paths.excel_path,
        args.out or cfg.paths.dataset_dir,
        schema=cfg.schema,
    )


if __name__ == "__main__":
    main()
