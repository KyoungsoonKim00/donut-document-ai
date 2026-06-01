"""Configuration loading: YAML -> dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class Paths:
    pdf_dir: str = "data/raw_pdf"
    image_dir: str = "data/images"
    excel_path: str = "data/labeling.xlsx"
    dataset_dir: str = "data/dataset"
    output_dir: str = "outputs/donut_finetuned"
    prediction_dir: str = "outputs/predictions"


@dataclass
class ModelCfg:
    name: str = "naver-clova-ix/donut-base"
    task_prompt: str = "<s_gt_parse>"
    image_height: int = 720
    image_width: int = 960
    max_length: int = 512


@dataclass
class PdfCfg:
    dpi: int = 300
    first_page: int = 1
    last_page: int = 1


@dataclass
class TrainCfg:
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    num_train_epochs: int = 15
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    fp16: bool = True
    val_split: float = 0.2
    seed: int = 42


@dataclass
class InferenceCfg:
    num_beams: int = 5
    early_stopping: bool = True


@dataclass
class SchemaCfg:
    numeric_fields: List[str] = field(default_factory=list)
    doc_type_field: str = "서류특성.서류종류"
    exclude_doc_type: str = "입고서류"
    exclude_fields_for_doc_type: List[str] = field(default_factory=list)
    stringify_list_fields: bool = True


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    model: ModelCfg = field(default_factory=ModelCfg)
    pdf: PdfCfg = field(default_factory=PdfCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    inference: InferenceCfg = field(default_factory=InferenceCfg)
    schema: SchemaCfg = field(default_factory=SchemaCfg)


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a Config dataclass.

    Unknown keys are ignored; missing keys fall back to dataclass defaults.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    def build(cls, data):
        valid = {k: v for k, v in (data or {}).items() if k in cls.__annotations__}
        return cls(**valid)

    return Config(
        paths=build(Paths, raw.get("paths")),
        model=build(ModelCfg, raw.get("model")),
        pdf=build(PdfCfg, raw.get("pdf")),
        train=build(TrainCfg, raw.get("train")),
        inference=build(InferenceCfg, raw.get("inference")),
        schema=build(SchemaCfg, raw.get("schema")),
    )
