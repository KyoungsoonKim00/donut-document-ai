# Hugging Face Hub 업로드 가이드 (처음 사용자용)

파인튜닝한 Donut 모델 가중치(`pytorch_model.bin` 등 약 815MB)는 GitHub에 올리지
않고 Hugging Face Hub에 올린 뒤, 코드에서 repo id로 불러온다.

---

## 1. 계정 + 액세스 토큰

1. https://huggingface.co/join 에서 무료 가입.
2. 로그인 후 우상단 프로필 → **Settings** → **Access Tokens**
   (직접 링크: https://huggingface.co/settings/tokens).
3. **Create new token** → Type을 **Write**로 선택 → 생성된 토큰(`hf_...`) 복사.
   - 이 토큰은 비밀번호와 같다. 외부에 노출하지 말 것. GitHub에 커밋 금지.

## 2. 라이브러리 설치

```bash
pip install -U huggingface_hub
```

## 3. 로그인 (둘 중 하나)

**방법 A — CLI 로그인 (권장):**

```bash
huggingface-cli login
# 프롬프트에 1번에서 복사한 hf_... 토큰 붙여넣기
```

**방법 B — 환경변수 (이번 세션만):**

```powershell
# PowerShell
$env:HF_TOKEN = "hf_여기에토큰"
```

## 4. 업로드 실행

`--model-dir`은 로컬 모델 폴더, `--repo-id`는 `사용자명/모델이름` 형식.

```bash
python scripts/upload_to_hf.py `
  --model-dir "C:\Users\rudtn\Desktop\jupyter notebook\Donut\donut_finetuned_O" `
  --repo-id  "ksk00/donut-docai"
```

- 비공개로 올리려면 `--private` 추가.
- 학습 체크포인트(`checkpoint-*`)·옵티마이저 파일은 자동 제외된다(추론에 불필요).
- 대용량 파일은 Git LFS로 자동 처리된다. 첫 업로드는 회선에 따라 수 분~수십 분.

## 5. 확인

업로드 후 출력되는 주소(`https://huggingface.co/<username>/donut-docai`)에서
파일 목록 확인. `pytorch_model.bin`, `config.json`, `tokenizer.json`,
`sentencepiece.bpe.model` 등이 보이면 성공.

## 6. 코드에서 불러오기

`configs/default.yaml`의 모델 경로를 repo id로 바꾸거나, 추론 시 직접 지정:

```bash
python scripts/05_inference.py --model ksk00/donut-docai
```

```python
from donut_docai import load_config
from donut_docai.inference import DonutPredictor

cfg = load_config("configs/default.yaml")
predictor = DonutPredictor(cfg, model_path="ksk00/donut-docai")
```

## 7. (선택) 모델 카드 작성

Hub repo의 `README.md`(모델 카드)에 용도·학습 데이터·한계를 적어두면 좋다.
이 저장소의 루트 README "Known limitations" 내용을 그대로 옮겨 적기를 권장.
