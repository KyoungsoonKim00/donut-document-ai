"""Run a fine-tuned Donut model on document images."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from .config import Config


class DonutPredictor:
    """Load a fine-tuned Donut model once and run inference on many images."""

    def __init__(self, cfg: Config, model_path: str | Path | None = None):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        path = str(model_path or cfg.paths.output_dir)

        self.processor = DonutProcessor.from_pretrained(path)
        self.model = VisionEncoderDecoderModel.from_pretrained(path).to(self.device)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.model.eval()

    def predict(self, image_path: str | Path) -> Tuple[str, dict | None]:
        """Return ``(raw_text, parsed_json_or_None)`` for one image."""
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(
            self.device
        )
        decoder_input_ids = self.processor.tokenizer(
            self.cfg.model.task_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device)

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.cfg.model.max_length,
            early_stopping=self.cfg.inference.early_stopping,
            num_beams=self.cfg.inference.num_beams,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )

        raw = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        json_text = raw.replace(self.cfg.model.task_prompt, "").replace("</s>", "").strip()
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            parsed = None
        return raw, parsed

    def predict_folder(self, image_dir: str | Path, output_dir: str | Path) -> int:
        """Predict every PNG in ``image_dir``; write parsed JSON to ``output_dir``.

        Returns the count of successfully parsed predictions.
        """
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        parsed_count = 0
        for file_name in sorted(os.listdir(image_dir)):
            if not file_name.lower().endswith(".png"):
                continue
            raw, parsed = self.predict(image_dir / file_name)
            stem = os.path.splitext(file_name)[0]
            if parsed is not None:
                with open(output_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=4)
                parsed_count += 1
                print(f"[ok] {file_name} -> {stem}.json")
            else:
                print(f"[warn] {file_name}: JSON parse failed | raw: {raw[:120]}")
        return parsed_count
