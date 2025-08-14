import argparse
from app.chatbot import SpiritualAssistant, GenerationConfig
from app.prompting import LanguageCode


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser()
	p.add_argument("--base_model_id", type=str, default=None)
	p.add_argument("--lora_path", type=str, default=None)
	p.add_argument("--prompt", type=str, required=True)
	p.add_argument("--language", type=str, default=None, choices=["en", "hi", "kn", None])
	p.add_argument("--max_new_tokens", type=int, default=512)
	p.add_argument("--temperature", type=float, default=0.7)
	p.add_argument("--top_p", type=float, default=0.9)
	return p.parse_args()


def main():
	args = parse_args()
	assistant = SpiritualAssistant(base_model_id=args.base_model_id, lora_path=args.lora_path)
	gen_cfg = GenerationConfig(
		max_new_tokens=args.max_new_tokens,
		temperature=args.temperature,
		top_p=args.top_p,
	)
	reply = assistant.generate(history=[], user_message=args.prompt, gen_config=gen_cfg, language=args.language)  # type: ignore
	print("\n=== Assistant ===\n")
	print(reply)


if __name__ == "__main__":
	main()