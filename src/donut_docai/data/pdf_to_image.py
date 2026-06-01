"""Convert PDF documents to PNG images for Donut.

Donut consumes a single image per document, so only the first page is rendered
by default. Requires the `poppler` binaries on PATH (pdf2image dependency).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pdf2image import convert_from_path


def convert_pdf_to_png(
    pdf_path: str | Path,
    output_path: str | Path,
    dpi: int = 300,
    first_page: int = 1,
    last_page: int = 1,
) -> Optional[Path]:
    """Render the first page of a PDF to a PNG (RGB).

    Returns the output path, or None if the PDF produced no pages.
    """
    pages = convert_from_path(
        str(pdf_path), dpi=dpi, first_page=first_page, last_page=last_page
    )
    if not pages:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].convert("RGB").save(output_path, "PNG")
    return output_path


def convert_folder(
    pdf_dir: str | Path,
    image_dir: str | Path,
    dpi: int = 300,
    first_page: int = 1,
    last_page: int = 1,
    skip_existing: bool = True,
) -> int:
    """Convert every PDF in ``pdf_dir`` to a PNG in ``image_dir``.

    Returns the number of images written.
    """
    pdf_dir, image_dir = Path(pdf_dir), Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for pdf_file in sorted(os.listdir(pdf_dir)):
        if not pdf_file.lower().endswith(".pdf"):
            continue
        out = image_dir / (Path(pdf_file).stem + ".png")
        if skip_existing and out.exists():
            continue
        try:
            if convert_pdf_to_png(pdf_dir / pdf_file, out, dpi, first_page, last_page):
                count += 1
                print(f"[ok] {pdf_file} -> {out.name}")
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"[fail] {pdf_file}: {e}")
    return count
