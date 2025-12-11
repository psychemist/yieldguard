from typing import Any

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    context: dict[str, Any] | None = None  # For passing wallet data, etc.


class ChatResponse(BaseModel):
    response: str
    metadata: dict[str, Any] | None = None
