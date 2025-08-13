#!/usr/bin/env python3
import os
from pathlib import Path
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("FINE_TUNE_BASE_MODEL", "gpt-4o-mini")
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "backend" / "data" / "finetune" / "finetune_dataset.jsonl"

if not OPENAI_API_KEY:
    raise SystemExit("Please set OPENAI_API_KEY in your environment")

if not DATASET.exists():
    raise SystemExit(f"Dataset not found at {DATASET}. Run scripts/prepare_finetune.py first.")

client = OpenAI(api_key=OPENAI_API_KEY)

print("Uploading dataset...")
file_obj = client.files.create(file=DATASET.open("rb"), purpose="fine-tune")

print("Creating fine-tune job...")
job = client.fine_tuning.jobs.create(training_file=file_obj.id, model=MODEL)
print("Job created:", job.id)
print("Track status with: openai fine_tuning.jobs.retrieve - or poll via API.")