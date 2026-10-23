"""Provider lookup and server-side API key fallback."""

import os

from .anthropic_provider import AnthropicProvider
from .base import LLMError, Provider
from .gemini_provider import GeminiProvider
from .nvidia_provider import NvidiaProvider
from .openai_provider import OpenAIProvider

# In the order the settings dialog shows them; the first is the default.
PROVIDERS: dict[str, Provider] = {
    p.id: p for p in (NvidiaProvider(), AnthropicProvider(), OpenAIProvider(), GeminiProvider())
}

# Used only when the browser does not send a key (e.g. a shared demo machine with a .env file).
_ENV_KEYS = {
    "nvidia": ("NVIDIA_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


def get_provider(provider_id: str) -> Provider:
    provider = PROVIDERS.get(provider_id)
    if provider is None:
        raise LLMError(f"'{provider_id}' нэртэй provider алга.", 400)
    return provider


def _env_key(provider_id: str) -> str:
    # A blank or whitespace-only variable counts as unset, so the next name is still tried.
    return next((v for n in _ENV_KEYS.get(provider_id, ()) if (v := os.environ.get(n, "").strip())), "")


def resolve_api_key(provider_id: str, header_key: str | None) -> str:
    key = (header_key or "").strip() or _env_key(provider_id)
    if not key:
        raise LLMError("API түлхүүр оруулаагүй байна. Тохиргоо (⚙) хэсэгт API түлхүүрээ оруулна уу.", 401)
    return key


def has_env_key(provider_id: str) -> bool:
    return bool(_env_key(provider_id))
