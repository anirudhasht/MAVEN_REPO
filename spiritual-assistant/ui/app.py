import os
import streamlit as st
from typing import Optional
from dotenv import load_dotenv

from app.chatbot import SpiritualAssistant, GenerationConfig
from app.prompting import LanguageCode

load_dotenv()

st.set_page_config(page_title="Spiritual Assistant (EN | HI | KN)", page_icon="🕊️", layout="centered")

if "history" not in st.session_state:
	st.session_state["history"] = []  # list of {role, content}

if "assistant" not in st.session_state:
	st.session_state["assistant"] = None

st.title("🕊️ Spiritual AI Assistant")
st.caption("Compassionate guidance in English, हिंदी, and ಕನ್ನಡ")

with st.sidebar:
	st.header("Settings")
	base_model_id = st.text_input("Base model ID", os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct"))
	lora_path = st.text_input("LoRA adapter path (optional)", "")
	use_4bit = st.checkbox("4-bit quantization", value=True)
	use_8bit = st.checkbox("8-bit quantization", value=False, help="Ignored if 4-bit is enabled")

	language_choice = st.selectbox(
		"Language",
		["Auto", "English", "हिंदी", "ಕನ್ನಡ"],
		index=0,
	)
	lang_map = {"Auto": None, "English": "en", "हिंदी": "hi", "ಕನ್ನಡ": "kn"}
	forced_language: Optional[LanguageCode] = lang_map[language_choice]  # type: ignore

	temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
	top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
	max_new_tokens = st.slider("Max new tokens", 64, 2048, 512, 32)
	repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.05, 0.01)

	if st.button("Clear conversation"):
		st.session_state["history"] = []
		st.experimental_rerun()


def get_assistant() -> SpiritualAssistant:
	if st.session_state["assistant"] is None:
		st.session_state["assistant"] = SpiritualAssistant(
			base_model_id=base_model_id,
			lora_path=lora_path or None,
			use_4bit=use_4bit,
			use_8bit=use_8bit,
		)
	return st.session_state["assistant"]


# Render chat history
for turn in st.session_state["history"]:
	with st.chat_message(turn["role"]):
		st.markdown(turn["content"])

user_input = st.chat_input("Ask or share what's on your heart…")
if user_input:
	st.session_state["history"].append({"role": "user", "content": user_input})
	with st.chat_message("user"):
		st.markdown(user_input)

	with st.chat_message("assistant"):
		with st.spinner("Reflecting…"):
			assistant = get_assistant()
			gen_cfg = GenerationConfig(
				temperature=temperature,
				top_p=top_p,
				max_new_tokens=max_new_tokens,
				repetition_penalty=repetition_penalty,
			)
			reply = assistant.generate(
				st.session_state["history"][:-1],
				user_input,
				gen_config=gen_cfg,
				language=forced_language,
			)
			st.markdown(reply)
			st.session_state["history"].append({"role": "assistant", "content": reply})