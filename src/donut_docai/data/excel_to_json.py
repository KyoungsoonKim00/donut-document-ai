"""Convert a labeling Excel workbook into Donut-format JSON labels.

Each non-empty row becomes one ``<name>.json`` file shaped as::

    {"file_name": "<doc>.pdf", "ground_truth": {"gt_parse": {<field>: <value>, ...}}}

Handles Korean transaction-statement quirks: Excel "',"-prefixed numbers,
comma-separated numeric lists, dates, and item-level fields that do not apply
to 입고서류 (incoming-goods) documents.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas._libs.tslibs.nattype import NaTType

from ..config import SchemaCfg


def convert_value(v: Any) -> Any:
    """Serialize dates to ISO strings; pass through everything else."""
    if isinstance(v, (pd.Timestamp, datetime)):
        if v.hour == 0 and v.minute == 0 and v.second == 0:
            return v.date().isoformat()
        return v.isoformat()
    return v


def smart_number(value: Any) -> Any:
    """Parse Excel cell values into int / float / list, robust to formatting.

    - "1,2,3"      -> [1, 2, 3]   (comma-separated numeric list)
    - "'00123"     -> 123          (strips text-format apostrophe)
    - "1.0"        -> 1            (collapses integral floats)
    - "" / NaN/Inf -> ""           (treated as empty)
    Non-numeric strings are returned unchanged.
    """
    try:
        if isinstance(value, str):
            value = value.strip().lstrip("'")
            if "," in value:
                result = []
                for x in value.split(","):
                    x = x.strip()
                    if x == "":
                        continue
                    f = float(x)
                    result.append(int(f) if f.is_integer() else f)
                return result
            if value == "":
                return ""
            f = float(value)
            return int(f) if f.is_integer() else f
        elif isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                return ""
            return int(value) if isinstance(value, float) and value.is_integer() else value
    except (ValueError, TypeError):
        return value
    return value


def stringify_if_list(value: Any) -> Any:
    """Join list values into a comma-separated string."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return value


def _normalize_file_name(raw: Any) -> str:
    """Coerce a raw cell into a ``*.pdf`` filename."""
    file_name = str(raw).strip()
    try:
        if file_name.replace(".", "", 1).isdigit() and float(file_name).is_integer():
            file_name = str(int(float(file_name)))
    except (ValueError, TypeError):
        pass
    if not file_name.lower().endswith(".pdf"):
        file_name += ".pdf"
    return file_name


def excel_to_donut_json(
    excel_path: str | Path,
    output_dir: str | Path,
    schema: SchemaCfg,
    only_pdf_rows: bool = True,
) -> int:
    """Convert ``excel_path`` rows into Donut JSON files in ``output_dir``.

    ``schema`` drives numeric parsing and the 입고서류 field-exclusion rule.
    Returns the number of JSON files written.
    """
    excel_path, output_dir = Path(excel_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(excel_path, header=0)
    df[df.columns[0]] = df[df.columns[0]].astype(str)
    df = df.rename(columns={df.columns[0]: "file_name"})

    if only_pdf_rows:
        mask = df["file_name"].astype(str).str.strip().str.lower().str.endswith(".pdf")
        df = df[mask]

    df = df.fillna("")

    for col in schema.numeric_fields:
        if col in df.columns:
            df[col] = df[col].apply(smart_number)

    written = 0
    for idx, row in df.iterrows():
        raw_dict = row.to_dict()
        file_name = _normalize_file_name(raw_dict.pop("file_name"))
        if not file_name or file_name == ".pdf":
            print(f"[skip] row {idx + 2}: missing file_name")
            continue

        doc_type = str(raw_dict.get(schema.doc_type_field, "")).strip()
        drop_items = doc_type == schema.exclude_doc_type

        gt_parse = {}
        for k, v in raw_dict.items():
            key = str(k)
            if str(v).strip() == "":
                continue
            if drop_items and key in schema.exclude_fields_for_doc_type:
                continue
            val = convert_value(v)
            if schema.stringify_list_fields:
                val = stringify_if_list(val)
            gt_parse[key] = val

        donut_json = {
            "file_name": file_name,
            "ground_truth": {"gt_parse": gt_parse},
        }

        out_path = output_dir / (os.path.splitext(file_name)[0] + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(donut_json, f, ensure_ascii=False, indent=4)
        written += 1

    print(f"[ok] wrote {written} Donut JSON files to {output_dir}")
    return written
