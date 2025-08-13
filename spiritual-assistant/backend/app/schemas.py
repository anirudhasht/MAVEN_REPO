from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="The latest user message")
    history: Optional[List[Message]] = Field(default=None, description="Prior messages in the conversation")
    use_rag: bool = Field(default=True, description="Whether to include retrieved context from local knowledge")


class ChatResponse(BaseModel):
    response: str