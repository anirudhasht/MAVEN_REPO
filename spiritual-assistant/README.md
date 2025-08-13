# Spiritual AI Assistant

Interactive spiritual guidance chatbot with a FastAPI backend and a Tailwind chat UI.

## Quick start

1) Create and activate the virtual environment (already prepared at `/workspace/spiritual-assistant/venv` in this environment):

```bash
source /workspace/spiritual-assistant/venv/bin/activate
```

2) Set environment variables (copy and edit):

```bash
cp .env.example .env
# Edit .env to add OPENAI_API_KEY
```

3) Run the backend:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

4) Open the UI. If serving statics directly from a local server, use a simple file server, or run:

```bash
python -m http.server 5173 -d frontend
```

Then visit http://localhost:5173 and ensure the backend is available at http://localhost:8000

## Fine-tuning

- Place your Q/A CSVs under `backend/data/finetune/*.csv` with columns: `question,answer`
- Prepare dataset:

```bash
python scripts/prepare_finetune.py
```

- Start a fine-tune job (requires billing-enabled OpenAI account):

```bash
OPENAI_API_KEY=... python scripts/start_finetune.py
```

Once complete, set `OPENAI_MODEL` in `.env` to your fine-tuned model name.

## Knowledge files (RAG)

Drop `.txt` or `.md` files into `backend/data/knowledge`. The assistant will retrieve relevant snippets automatically.