import os
from dataclasses import dataclass
from typing import List, Optional, Literal

import torch
from transformers import (
	AutoTokenizer,
	AutoModelForCausalLM,
	BitsAndBytesConfig,
)
from peft import PeftModel

from .prompting import get_system_prompt, detect_language_code, LanguageCode


@dataclass
class GenerationConfig:
	temperature: float = 0.7
	top_p: float = 0.9
	max_new_tokens: int = 512
	repetition_penalty: float = 1.05


class SpiritualAssistant:
	"""Loads a base LLM (optionally with LoRA) and generates responses with a spiritual persona.

	- Base model can be set via env `BASE_MODEL_ID` or init arg
	- Optional LoRA adapter via `lora_path`
	- Language can be auto-detected or forced via arg
	"""

	def __init__(
		self,
		base_model_id: Optional[str] = None,
		lora_path: Optional[str] = None,
		use_4bit: bool = True,
		use_8bit: bool = False,
		trust_remote_code: bool = True,
		hf_token: Optional[str] = None,
	):
		self.base_model_id = base_model_id or os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
		self.lora_path = lora_path
		self.hf_token = hf_token or os.getenv("HF_TOKEN")

		quant_config = None
		if use_4bit or use_8bit:
			quant_config = BitsAndBytesConfig(
				load_in_4bit=use_4bit,
				load_in_8bit=use_8bit and not use_4bit,
				bnb_4bit_compute_dtype=torch.float16,
			)

		self.tokenizer = AutoTokenizer.from_pretrained(
			self.base_model_id,
			use_fast=True,
			trust_remote_code=trust_remote_code,
			token=self.hf_token,
		)

		self.model = AutoModelForCausalLM.from_pretrained(
			self.base_model_id,
			quantization_config=quant_config,
			device_map="auto",
			trust_remote_code=trust_remote_code,
			token=self.hf_token,
		)

		if self.lora_path:
			self.model = PeftModel.from_pretrained(self.model, self.lora_path)

		self.model.eval()

	def _resolve_language(self, forced_language: Optional[LanguageCode], text_for_detection: str) -> LanguageCode:
		if forced_language:
			return forced_language
		return detect_language_code(text_for_detection)

	def _build_messages(self, language: LanguageCode, history: List[dict], user_message: str) -> List[dict]:
		messages: List[dict] = []
		messages.append({"role": "system", "content": get_system_prompt(language)})
		for turn in history:
			if "role" in turn and "content" in turn:
				messages.append(turn)
		messages.append({"role": "user", "content": user_message})
		return messages

	def generate(
		self,
		history: List[dict],
		user_message: str,
		gen_config: Optional[GenerationConfig] = None,
		language: Optional[LanguageCode] = None,
	) -> str:
		resolved_language = self._resolve_language(language, f"{user_message}")
		messages = self._build_messages(resolved_language, history, user_message)

		prompt = self.tokenizer.apply_chat_template(
			messages,
			add_generation_prompt=True,
			return_tensors="pt",
		)
		prompt = prompt.to(self.model.device)

		gen = gen_config or GenerationConfig()
		with torch.no_grad():
			output_ids = self.model.generate(
				prompt,
				max_new_tokens=gen.max_new_tokens,
				temperature=gen.temperature,
				top_p=gen.top_p,
				repetition_penalty=gen.repetition_penalty,
				do_sample=True,
			)

		new_token_ids = output_ids[0][prompt.shape[-1]:]
		assistant_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
		return assistant_text