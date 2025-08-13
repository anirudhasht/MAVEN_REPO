from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Iterator
import os
from openai import OpenAIError
from .schemas import ChatRequest
from .config import SYSTEM_PROMPT
from .rag import retriever
from .openai_client import build_messages, stream_chat_completions

app = FastAPI(title="Spiritual Assistant API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    context_blocks = retriever.retrieve(req.message, top_k=3) if req.use_rag else []
    messages = build_messages(SYSTEM_PROMPT, [m.model_dump() for m in (req.history or [])], req.message, context_blocks)

    def token_generator() -> Iterator[bytes]:
        for token in stream_chat_completions(messages):
            yield token.encode("utf-8")

    try:
        return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")
    except OpenAIError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# Serve the frontend
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")