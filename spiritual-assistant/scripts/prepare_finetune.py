#!/usr/bin/env python3
import os
import json
import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "backend" / "data" / "finetune"
OUT_FILE = DATA_DIR / "finetune_dataset.jsonl"

EXAMPLE = {
    "messages": [
        {"role": "system", "content": "You are a compassionate Spiritual Guide."},
        {"role": "user", "content": "How can I calm my mind before sleep?"},
        {"role": "assistant", "content": "Try a 4-7-8 breathing pattern and a short gratitude reflection."}
    ]
}


def convert_csv(csv_path: Path) -> int:
    n = 0
    with csv_path.open("r", encoding="utf-8") as f, OUT_FILE.open("w", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        if not {"question", "answer"}.issubset(reader.fieldnames or set()):
            raise ValueError("CSV must have 'question' and 'answer' columns")
        for row in reader:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if not q or not a:
                continue
            record = {
                "messages": [
                    {"role": "system", "content": "You are a compassionate Spiritual Guide."},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_candidates = list(DATA_DIR.glob("*.csv"))
    if not csv_candidates:
        EXAMPLE_CSV = DATA_DIR / "example.csv"
        with EXAMPLE_CSV.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["question", "answer"])
            w.writerow(["I feel anxious.", "Place a gentle hand on your heart, breathe slowly, and name three things you are grateful for."])
            w.writerow(["How to start meditation?", "Begin with 2 minutes of mindful breathing. Focus on inhales and exhales; extend gradually."])
        print(f"No CSV found. Wrote example at {EXAMPLE_CSV}. Re-run after adding your data.")
        return

    rows = 0
    for csv_path in csv_candidates:
        rows += convert_csv(csv_path)
    print(f"Wrote {rows} examples to {OUT_FILE}")


if __name__ == "__main__":
    main()