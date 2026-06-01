"""Fine-tune Donut on the transaction-statement dataset."""
from __future__ import annotations

import os

import torch
from transformers import (
    DonutProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    default_data_collator,
)

from .config import Config
from .dataset import build_datasets


def build_model_and_processor(cfg: Config):
    """Load the base Donut model + processor and wire up Donut-specific config."""
    processor = DonutProcessor.from_pretrained(cfg.model.name)
    model = VisionEncoderDecoderModel.from_pretrained(cfg.model.name)

    # Donut decodes JSON starting from the task prompt token.
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        cfg.model.task_prompt
    )
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.use_cache = False  # incompatible with gradient checkpointing
    model.gradient_checkpointing_enable()  # ~20% VRAM savings
    return model, processor


def train(cfg: Config) -> str:
    """Run fine-tuning and save the model + processor to ``cfg.paths.output_dir``.

    Returns the output directory path.
    """
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    model, processor = build_model_and_processor(cfg)
    train_ds, val_ds = build_datasets(cfg, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.paths.output_dir,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        max_grad_norm=cfg.train.max_grad_norm,
        num_train_epochs=cfg.train.num_train_epochs,
        logging_dir=os.path.join(cfg.paths.output_dir, "logs"),
        logging_steps=cfg.train.logging_steps,
        eval_strategy=cfg.train.eval_strategy,
        save_strategy=cfg.train.save_strategy,
        predict_with_generate=True,
        remove_unused_columns=False,
        fp16=cfg.train.fp16,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    model.save_pretrained(cfg.paths.output_dir)
    processor.save_pretrained(cfg.paths.output_dir)
    print(f"[ok] saved model + processor to {cfg.paths.output_dir}")
    return cfg.paths.output_dir
