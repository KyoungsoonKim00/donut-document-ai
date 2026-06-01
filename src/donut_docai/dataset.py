"""Build a Hugging Face Dataset from Donut JSON labels + images."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

from datasets import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

from .config import Config


def json_to_token_string(json_obj: dict, task_prompt: str) -> str:
    """Serialize a gt_parse dict into the Donut target sequence."""
    return task_prompt + json.dumps(json_obj, ensure_ascii=False) + "</s>"


def load_records(dataset_dir: str | Path) -> List[dict]:
    """Load ``{image_path, ground_truth}`` records from a dataset directory.

    A record is included only when both ``<name>.json`` and ``<name>.png`` exist.
    """
    dataset_dir = Path(dataset_dir)
    records = []
    for json_file in sorted(os.listdir(dataset_dir)):
        if not json_file.endswith(".json"):
            continue
        base = json_file[: -len(".json")]
        image_path = dataset_dir / f"{base}.png"
        if not image_path.exists():
            continue
        with open(dataset_dir / json_file, "r", encoding="utf-8") as f:
            label = json.load(f)
        records.append(
            {
                "image_path": str(image_path),
                "ground_truth": label["ground_truth"]["gt_parse"],
            }
        )
    return records


def build_datasets(cfg: Config, processor) -> Tuple[Dataset, Dataset]:
    """Load records, split train/val, and tokenize into model-ready tensors."""
    records = load_records(cfg.paths.dataset_dir)
    if not records:
        raise RuntimeError(
            f"No (json, png) pairs found in {cfg.paths.dataset_dir}. "
            "Run the PDF->image and Excel->JSON steps first."
        )

    train_data, val_data = train_test_split(
        records, test_size=cfg.train.val_split, random_state=cfg.train.seed
    )

    def preprocess(example):
        image = Image.open(example["image_path"]).convert("RGB")
        image_encoding = processor.image_processor(
            images=image,
            return_tensors="pt",
            size={"height": cfg.model.image_height, "width": cfg.model.image_width},
        )
        target_text = json_to_token_string(
            example["ground_truth"], cfg.model.task_prompt
        )
        labels = processor.tokenizer(
            target_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.model.max_length,
        ).input_ids[0]
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return {
            "pixel_values": image_encoding["pixel_values"][0],
            "labels": labels,
        }

    train_ds = Dataset.from_list(train_data).map(preprocess)
    val_ds = Dataset.from_list(val_data).map(preprocess)
    train_ds = train_ds.remove_columns(["image_path", "ground_truth"])
    val_ds = val_ds.remove_columns(["image_path", "ground_truth"])
    return train_ds, val_ds
