# Data format for fine-tuning

Each line in `train.jsonl` and `val.jsonl` must be a JSON object with:

- `instruction` (string): the main user query or instruction
- `input` (string, optional): extra context
- `output` (string): the assistant's ideal answer
- `language` (string, optional): `en`, `hi`, or `kn`

Example lines:
```json
{"instruction": "How can I calm my mind?", "input": "", "output": "Try a gentle breath practice: inhale 4, exhale 6...", "language": "en"}
{"instruction": "जीवन का अर्थ क्या है?", "input": "", "output": "अर्थ अक्सर अनुभव और संबंधों में पाया जाता है...", "language": "hi"}
{"instruction": "ನಾನು ಆತಂಕವನ್ನು ಹೇಗೆ ಕಡಿಮೆ ಮಾಡಬಹುದು?", "input": "", "output": "ದೀರ್ಘ ಉಸಿರಾಟ ಮತ್ತು ನೆಲಗಟ್ಟುವಿಕೆ ಅಭ್ಯಾಸ ಮಾಡಿ...", "language": "kn"}
```