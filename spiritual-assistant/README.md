# Spiritual AI Assistant (English | हिंदी | ಕನ್ನಡ)

A compassionate spiritual AI assistant that chats in English, Hindi, and Kannada. Includes an interactive Streamlit UI and an end-to-end fine-tuning pipeline (LoRA + Transformers + PEFT) for domain adaptation.

## Features
- Multilingual: English, Hindi, Kannada (auto-detect or force a language)
- Spiritual persona tuned for reflective, compassionate guidance
- Local inference with Hugging Face models (default: `Qwen/Qwen2.5-3B-Instruct`)
- Fine-tuning via LoRA adapters on your dataset (JSONL)
- Simple evaluation script

## 1) Prerequisites
- Python 3.10+
- Recommended: a GPU with CUDA for faster inference/fine-tuning
- A Hugging Face token if the model requires auth

## 2) Setup
```bash
cd /workspace/spiritual-assistant
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional: set a default base model and HF token
```bash
cp .env.example .env
# edit .env to set BASE_MODEL_ID and HF_TOKEN
```

## 3) Run the Chat UI
```bash
source .venv/bin/activate
streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
```
- Open the URL it prints. Choose language: Auto, English, हिंदी (Hindi), ಕನ್ನಡ (Kannada).
- Customize generation params in the sidebar.

## 4) Dataset Format for Fine-tuning
Provide a JSONL file where each line is an object with fields:
```json
{"instruction": "Ask your question here", "input": "(optional context)", "output": "Assistant answer", "language": "en|hi|kn"}
```
- `language` is optional; used to pick a localized system prompt. If missing, we auto-detect from `instruction + input`.
- Place your dataset at `training/data/train.jsonl` and `training/data/val.jsonl` (you can change paths via CLI flags).

## 5) Fine-tuning (LoRA)
```bash
source .venv/bin/activate
python training/finetune.py \
  --base_model_id "Qwen/Qwen2.5-3B-Instruct" \
  --train_file training/data/train.jsonl \
  --eval_file training/data/val.jsonl \
  --output_dir training/outputs/qwen2_5_3b_spiritual_lora \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 2 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --fp16 --use_4bit
```
Notes:
- Use `--bf16` on newer GPUs.
- Use `--use_8bit` if 4-bit is problematic.
- The script saves LoRA adapters to `output_dir`.

## 6) Use the Fine-tuned Adapter in the UI
- In the Streamlit sidebar, set:
  - Base model: same as used for training
  - LoRA adapter path: `training/outputs/qwen2_5_3b_spiritual_lora` (leave empty to use base model only)

## 7) Merge LoRA into a Full Model (optional)
```bash
python training/finetune.py \
  --base_model_id "Qwen/Qwen2.5-3B-Instruct" \
  --lora_path training/outputs/qwen2_5_3b_spiritual_lora \
  --merge_lora \
  --output_dir training/outputs/qwen2_5_3b_spiritual_merged
```
The merged model can be loaded without PEFT.

## 8) Quick Evaluation
```bash
python eval/evaluate.py \
  --base_model_id "Qwen/Qwen2.5-3B-Instruct" \
  --lora_path training/outputs/qwen2_5_3b_spiritual_lora \
  --prompt "जीवन का उद्देश्य क्या है?" \
  --language hi
```

## 9) Notes on Languages
- The assistant replies in the detected language or the forced selection from the UI.
- You can expand `app/prompting.py` to add more localized tone/style and additional languages later.

## 10) Safety & Scope
- The assistant offers spiritual reflections and general guidance only.
- It is not a medical, legal, or mental health professional. It should advise users to consult qualified experts for those domains.

## 11) Repo Structure
```
spiritual-assistant/
  app/
    __init__.py
    chatbot.py
    prompting.py
  eval/
    evaluate.py
  training/
    finetune.py
    data_format.md
    data/
      train.jsonl   # you add
      val.jsonl     # you add
  ui/
    app.py
  .env.example
  requirements.txt
  README.md
```

## 12) Troubleshooting
- CUDA OOM: reduce sequence length, batch size, or use 4-bit/8-bit loading.
- Missing CUDA: run on CPU (slow) by omitting 4bit/8bit flags and ensuring device map uses CPU.
- Model auth: set `HF_TOKEN` in `.env` or run `huggingface-cli login`.