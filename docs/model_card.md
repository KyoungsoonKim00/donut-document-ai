---
license: mit
language:
- ko
library_name: transformers
pipeline_tag: image-to-text
base_model: naver-clova-ix/donut-base
tags:
- donut
- document-understanding
- ocr-free
- vision-encoder-decoder
- korean
---

# donut-docai — Korean Transaction-Statement Parser

Fine-tuned [Donut](https://github.com/clovaai/donut)
(`naver-clova-ix/donut-base`) that reads a Korean transaction statement
(거래명세표 / 계산서) **image** and outputs structured JSON — no OCR + rule engine.

Code & full pipeline: https://github.com/KyoungsoonKim00/donut-document-ai

## Usage

```python
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("ksk00/donut-docai")
model = VisionEncoderDecoderModel.from_pretrained("ksk00/donut-docai")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

image = Image.open("document.png").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
decoder_input_ids = processor.tokenizer(
    "<s_gt_parse>", return_tensors="pt", add_special_tokens=False
).input_ids.to(device)

outputs = model.generate(
    pixel_values, decoder_input_ids=decoder_input_ids,
    max_length=512, num_beams=5,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
)
print(processor.batch_decode(outputs, skip_special_tokens=True)[0])
```

## Output schema

| Group | Fields |
|-------|--------|
| `서류특성.*` | 서류종류, 거래일, 합계금액 |
| `피공급자.*` | 이름, 거래전미지급금, 입금액, 현잔액 |
| `품목.*` | 품목명, 코드, 단위, 수량, 단가, 공급가액, 세액, 수량합계, 공급가액합계, 세액합계 |

## Training

- Base: `naver-clova-ix/donut-base` (Swin-B encoder + mBART decoder)
- Image size 720×960, task prompt `<s_gt_parse>`, max length 512
- AdamW lr 5e-5, weight decay 0.01, warmup 5%, 15 epochs, fp16, gradient checkpointing

## Limitations

Trained on a small in-house dataset (tens of documents). The model overfits and
can collapse into repeated tokens on unseen layouts. Treat as a proof-of-concept,
not production-ready. See the GitHub repo for improvement directions.
