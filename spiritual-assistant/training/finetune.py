import os
import json
import argparse
from typing import Optional, Dict, Any, List, Tuple

import torch
from datasets import load_dataset
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
	Trainer,
	TrainingArguments,
	DataCollatorForLanguageModeling,
	set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel

# Local prompt helpers
from app.prompting import get_system_prompt, detect_language_code


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="LoRA fine-tune for Spiritual Assistant")
	parser.add_argument("--base_model_id", type=str, default=os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct"))
	parser.add_argument("--train_file", type=str, required=False, default="training/data/train.jsonl")
	parser.add_argument("--eval_file", type=str, required=False, default="training/data/val.jsonl")
	parser.add_argument("--output_dir", type=str, required=True)
	parser.add_argument("--seed", type=int, default=42)

	# LoRA
	parser.add_argument("--lora_r", type=int, default=16)
	parser.add_argument("--lora_alpha", type=int, default=32)
	parser.add_argument("--lora_dropout", type=float, default=0.05)
	parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")

	# Quant / precision
	parser.add_argument("--use_4bit", action="store_true")
	parser.add_argument("--use_8bit", action="store_true")
	parser.add_argument("--fp16", action="store_true")
	parser.add_argument("--bf16", action="store_true")

	# Train params
	parser.add_argument("--per_device_train_batch_size", type=int, default=2)
	parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
	parser.add_argument("--learning_rate", type=float, default=2e-4)
	parser.add_argument("--num_train_epochs", type=int, default=2)
	parser.add_argument("--max_steps", type=int, default=-1)
	parser.add_argument("--warmup_ratio", type=float, default=0.03)
	parser.add_argument("--logging_steps", type=int, default=10)
	parser.add_argument("--save_steps", type=int, default=200)
	parser.add_argument("--eval_steps", type=int, default=200)
	parser.add_argument("--save_total_limit", type=int, default=3)
	parser.add_argument("--gradient_checkpointing", action="store_true")
	parser.add_argument("--max_seq_length", type=int, default=2048)

	# Merging
	parser.add_argument("--lora_path", type=str, default=None, help="Path to existing LoRA adapters")
	parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA into base and save")

	return parser.parse_args()


def load_model_and_tokenizer(base_model_id: str, use_4bit: bool, use_8bit: bool) -> Tuple[Any, Any]:
	quant_config = None
	if use_4bit or use_8bit:
		quant_config = BitsAndBytesConfig(
			load_in_4bit=use_4bit,
			load_in_8bit=use_8bit and not use_4bit,
			bnb_4bit_compute_dtype=torch.float16,
		)

	tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, trust_remote_code=True)
	model = AutoModelForCausalLM.from_pretrained(
		base_model_id,
		device_map="auto",
		quantization_config=quant_config,
		trust_remote_code=True,
	)
	return model, tokenizer


def format_example_to_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
	instruction = example.get("instruction", "").strip()
	input_text = (example.get("input", "") or "").strip()
	answer = example.get("output", "").strip()
	lang = example.get("language") or detect_language_code(" ".join([instruction, input_text]))
	system = get_system_prompt(lang)
	user = instruction if not input_text else f"{instruction}\n\n{input_text}"
	messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": user},
		{"role": "assistant", "content": answer},
	]
	return messages


def tokenize_with_labels(tokenizer, messages: List[Dict[str, str]], max_length: int) -> Dict[str, Any]:
	# Prompt ids without assistant content (generation prompt)
	prompt_messages = messages[:-1]
	prompt_ids = tokenizer.apply_chat_template(
		prompt_messages,
		add_generation_prompt=True,
		return_tensors="pt",
	)[0]

	# Full ids including assistant content
	full_ids = tokenizer.apply_chat_template(
		messages,
		add_generation_prompt=False,
		return_tensors="pt",
	)[0]

	# Labels mask out the prompt
	labels = full_ids.clone()
	assistant_start = prompt_ids.shape[0]
	labels[:assistant_start] = -100

	# Truncate from left if necessary
	if full_ids.shape[0] > max_length:
		overflow = full_ids.shape[0] - max_length
		full_ids = full_ids[overflow:]
		labels = labels[overflow:]

	return {
		"input_ids": full_ids,
		"labels": labels,
		"attention_mask": torch.ones_like(full_ids),
	}


def build_dataset(tokenizer, train_file: Optional[str], eval_file: Optional[str], max_length: int):
	data_files = {}
	if train_file and os.path.exists(train_file):
		data_files["train"] = train_file
	if eval_file and os.path.exists(eval_file):
		data_files["validation"] = eval_file
	if not data_files:
		raise FileNotFoundError("Provide at least one of train_file or eval_file with valid path")

	ds = load_dataset("json", data_files=data_files)

	def _map_fn(example: Dict[str, Any]) -> Dict[str, Any]:
		messages = format_example_to_messages(example)
		return tokenize_with_labels(tokenizer, messages, max_length)

	columns = ["input_ids", "labels", "attention_mask"]
	mapped = ds.map(_map_fn, remove_columns=ds[list(ds.keys())[0]].column_names)
	mapped.set_format(type="torch", columns=columns)
	return mapped


def train(args: argparse.Namespace):
	set_seed(args.seed)

	if args.merge_lora:
		if not args.lora_path:
			raise ValueError("--merge_lora requires --lora_path")
		model, tokenizer = load_model_and_tokenizer(args.base_model_id, args.use_4bit, args.use_8bit)
		model = PeftModel.from_pretrained(model, args.lora_path)
		merged = model.merge_and_unload()
		os.makedirs(args.output_dir, exist_ok=True)
		merged.save_pretrained(args.output_dir)
		tokenizer.save_pretrained(args.output_dir)
		print(f"Merged model saved to {args.output_dir}")
		return

	model, tokenizer = load_model_and_tokenizer(args.base_model_id, args.use_4bit, args.use_8bit)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
	peft_config = LoraConfig(
		r=args.lora_r,
		lora_alpha=args.lora_alpha,
		lora_dropout=args.lora_dropout,
		target_modules=target_modules,
		task_type="CAUSAL_LM",
	)
	model = get_peft_model(model, peft_config)

	dataset = build_dataset(tokenizer, args.train_file, args.eval_file, args.max_seq_length)

	fp16 = args.fp16 and not args.bf16
	bf16 = args.bf16 and torch.cuda.is_available()

	training_args = TrainingArguments(
		output_dir=args.output_dir,
		overwrite_output_dir=True,
		num_train_epochs=args.num_train_epochs,
		per_device_train_batch_size=args.per_device_train_batch_size,
		per_device_eval_batch_size=args.per_device_eval_batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		learning_rate=args.learning_rate,
		warmup_ratio=args.warmup_ratio,
		logging_steps=args.logging_steps,
		evaluation_strategy="steps" if "validation" in dataset else "no",
		eval_steps=args.eval_steps,
		save_steps=args.save_steps,
		save_total_limit=args.save_total_limit,
		bf16=bf16,
		fp16=fp16,
		gradient_checkpointing=args.gradient_checkpointing,
		report_to=["none"],
		max_steps=args.max_steps,
	)

	collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=dataset.get("train"),
		eval_dataset=dataset.get("validation"),
		data_collator=collator,
	)

	trainer.train()
	trainer.save_model(args.output_dir)
	tokenizer.save_pretrained(args.output_dir)
	print(f"LoRA adapters saved to {args.output_dir}")


if __name__ == "__main__":
	args = parse_args()
	train(args)