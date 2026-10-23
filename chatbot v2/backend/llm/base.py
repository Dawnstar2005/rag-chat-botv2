"""Common interface every LLM provider implements."""

from collections.abc import AsyncIterator
from typing import Protocol


class LLMError(Exception):
    """A provider failure with a user-facing (Mongolian) message and an HTTP status."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.message = message
        self.status = status


# Messages are plain dicts: {"role": "user" | "assistant", "content": str}.
Message = dict[str, str]


class Provider(Protocol):
    id: str
    name: str
    default_model: str
    key_hint: str  # e.g. "sk-ant-..."
    key_url: str  # where the user can create a key

    async def list_models(self, api_key: str) -> list[dict]:
        """Return [{"id": ..., "name": ...}]; also serves as an API key check."""
        ...

    def stream_chat(
        self, api_key: str, model: str, system: str, messages: list[Message]
    ) -> AsyncIterator[str]:
        """Yield response text fragments as they arrive. Raises LLMError."""
        ...
