"""
Krusch Agentic Proxy — Pydantic Request/Response Models.

Provides schema validation and body size constraints for all API endpoints.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = Field(..., description="Message role: system, user, assistant, or tool")
    content: Optional[str] = Field(None, description="Message content")
    name: Optional[str] = Field(None, description="Optional name for the message author")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls from assistant")
    tool_call_id: Optional[str] = Field(None, description="ID of the tool call this message responds to")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(default="qwen2.5-coder:7b", description="Model identifier")
    messages: List[ChatMessage] = Field(..., min_length=1, description="Conversation messages")
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=32768)
    stream: Optional[bool] = Field(default=False)
    tools: Optional[List[Dict[str, Any]]] = Field(default=None)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None)
    response_format: Optional[Dict[str, Any]] = Field(default=None)
    force_autonomous: Optional[bool] = Field(default=False)

    @field_validator("messages")
    @classmethod
    def validate_messages_size(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        """Reject oversized message payloads to prevent OOM."""
        total_chars = sum(len(msg.content or "") for msg in v)
        if total_chars > 500_000:  # ~500KB text limit
            raise ValueError(
                f"Total message content ({total_chars} chars) exceeds the 500,000 character limit."
            )
        return v


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: Dict[str, Any]
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None
