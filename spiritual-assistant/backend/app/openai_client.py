from typing import Iterable, List, Dict, Any, Optional
import os
from openai import OpenAI, OpenAIError
from .config import OPENAI_API_KEY, OPENAI_MODEL, API_BASE

_client_kwargs: Dict[str, Any] = {}
if OPENAI_API_KEY:
    _client_kwargs["api_key"] = OPENAI_API_KEY
if API_BASE:
    _client_kwargs["base_url"] = API_BASE

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = _client_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIError("Missing OPENAI_API_KEY. Set it in environment or .env")
        kwargs = dict(_client_kwargs)
        kwargs["api_key"] = api_key
        _client = OpenAI(**kwargs)
    return _client


def build_messages(system_prompt: str, history: List[Dict[str, str]] | None, user_message: str, context_blocks: List[str] | None = None) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if context_blocks:
        context_text = "\n\n".join([f"[Context]\n{c.strip()}" for c in context_blocks if c.strip()])
        messages.append({"role": "system", "content": f"Use the following context if relevant.\n{context_text}"})
    if history:
        for m in history:
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


def stream_chat_completions(messages: List[Dict[str, str]], model: str | None = None) -> Iterable[str]:
    chosen_model = model or OPENAI_MODEL
    client = get_client()
    stream = client.chat.completions.create(
        model=chosen_model,
        messages=messages,
        temperature=0.7,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
        if delta:
            yield delta