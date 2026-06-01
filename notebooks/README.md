# Reference notebooks

The original capstone work was built in Jupyter. These notebooks are kept for
reference — they show the exploration behind each pipeline stage. **Outputs are
cleared and all paths are anonymized** (`data/...`, `outputs/...`).

For production use prefer the `scripts/` CLI + `src/donut_docai` package, which
are the cleaned, configurable versions of these notebooks.

| Notebook | Stage | Package equivalent |
|----------|-------|--------------------|
| `01_pdf_to_image.ipynb` | PDF → PNG | `data/pdf_to_image.py` |
| `02_prepare_labels.ipynb` | Seed labeling Excel | `data/filename_to_excel.py` |
| `03_excel_to_json.ipynb` | Excel → Donut JSON | `data/excel_to_json.py` |
| `04_finetune.ipynb` | Fine-tuning | `train.py` |
| `05_inference.ipynb` | Inference | `inference.py` |
