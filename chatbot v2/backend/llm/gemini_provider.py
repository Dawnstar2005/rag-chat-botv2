"""Google Gemini provider (official `google-genai` SDK, async streaming)."""

import os
import ssl
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import cache

import anyio
import certifi
import httpx  # the SDK's async HTTP client (unless the optional aiohttp extra is installed)
from google import genai
from google.genai import errors, types
from google.genai.client import AsyncClient

from .base import LLMError, Message

_PREFERRED = ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash")
_LIST_TIMEOUT_MS = 30_000
_CHAT_TIMEOUT_MS = 600_000  # same 10 minutes the Anthropic and OpenAI SDKs default to

_FILTERED = {
    types.FinishReason.SAFETY, types.FinishReason.PROHIBITED_CONTENT, types.FinishReason.BLOCKLIST,
    types.FinishReason.SPII, types.FinishReason.RECITATION,
}
# Network failures: httpx errors, plus OSError/TimeoutError raised by the aiohttp transport.
_NETWORK_ERRORS = (httpx.TransportError, OSError, TimeoutError)
_ERRORS = (errors.APIError, errors.UnknownApiResponseError, *_NETWORK_ERRORS)


def _to_llm_error(exc: Exception) -> LLMError:
    if isinstance(exc, errors.APIError):
        code = exc.code or 500
        detail = exc.message or str(exc)
        if code in (401, 403) or "API_KEY" in str(exc) or "API key" in detail:
            return LLMError("Gemini API түлхүүр буруу байна. Тохиргооноос шалгана уу.", 401)
        if code == 404:
            return LLMError("Сонгосон загвар олдсонгүй. Өөр загвар сонгоно уу.", 404)
        if code == 429:
            return LLMError("Хүсэлтийн хязгаар (квот) хэтэрлээ. Түр хүлээгээд дахин оролдоно уу.", 429)
        if code == 400:
            return LLMError(f"Хүсэлт буруу байна: {detail}", 400)
        if code >= 500:
            return LLMError("Gemini сервер түр ачаалалтай байна. Дахин оролдоно уу.", 503)
        return LLMError(f"Gemini API алдаа ({code}): {detail}", 502)
    if isinstance(exc, _NETWORK_ERRORS):
        return LLMError("Gemini API-тай холбогдож чадсангүй. Интернэт холболтоо шалгана уу.", 503)
    return LLMError(f"Gemini API-аас ойлгомжгүй хариу ирлээ: {exc}", 502)


@cache
def _ssl_context() -> ssl.SSLContext:
    # Same CA settings the SDK uses. Without a shared context every genai.Client loads the CA
    # bundle twice (~0.3 s), blocking the event loop on each request.
    return ssl.create_default_context(
        cafile=os.environ.get("SSL_CERT_FILE", certifi.where()), capath=os.environ.get("SSL_CERT_DIR")
    )


@asynccontextmanager
async def _client(api_key: str, timeout_ms: int) -> AsyncIterator[AsyncClient]:
    """A client for one request; its HTTP connections are closed afterwards."""
    ctx = _ssl_context()
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=timeout_ms,
            client_args={"verify": ctx},
            async_client_args={"verify": ctx, "ssl": ctx},  # "verify" for httpx, "ssl" for aiohttp/websockets
        ),
    )
    try:
        yield client.aio
    finally:
        with anyio.CancelScope(shield=True):  # still close when the browser disconnected mid-stream
            await client.aio.aclose()
        client.close()


class GeminiProvider:
    id = "gemini"
    name = "Google Gemini"
    default_model = "gemini-2.5-flash"
    key_hint = "AIza..."
    key_url = "https://aistudio.google.com/apikey"

    async def list_models(self, api_key: str) -> list[dict]:
        models = []
        try:
            async with _client(api_key, _LIST_TIMEOUT_MS) as client:
                async for model in await client.models.list():
                    name = (model.name or "").removeprefix("models/")
                    actions = model.supported_actions or []
                    if "gemini" in name and "generateContent" in actions and "embedding" not in name:
                        models.append({"id": name, "name": model.display_name or name})
        except _ERRORS as exc:
            raise _to_llm_error(exc) from exc
        preferred = [m for p in _PREFERRED for m in models if m["id"] == p]
        rest = sorted((m for m in models if m not in preferred), key=lambda m: m["id"], reverse=True)
        return preferred + rest

    async def stream_chat(
        self, api_key: str, model: str, system: str, messages: list[Message]
    ) -> AsyncIterator[str]:
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            # No tools are declared; skip the SDK's function-calling loop.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        finish_reason = None
        blocked = False
        try:
            async with _client(api_key, _CHAT_TIMEOUT_MS) as client:
                async for chunk in await client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                ):
                    if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                        blocked = True  # the question itself was rejected; no candidates follow
                    for candidate in chunk.candidates or []:
                        if candidate.finish_reason:
                            finish_reason = candidate.finish_reason
                        for part in (candidate.content.parts if candidate.content else None) or []:
                            # Skip thought summaries; only stream the visible answer.
                            if part.text and not part.thought:
                                yield part.text
        except _ERRORS as exc:
            raise _to_llm_error(exc) from exc

        if blocked:
            yield "> ⚠️ Gemini энэ асуултыг аюулгүй байдлын шүүлтүүрээр хаасан тул хариулт өгсөнгүй."
        elif finish_reason == types.FinishReason.MAX_TOKENS:
            yield "\n\n> ⚠️ Хариулт хэт урт тул таслагдлаа."
        elif finish_reason in _FILTERED:
            yield "\n\n> ⚠️ Хариулт аюулгүй байдлын шүүлтүүрээр зогсоогдлоо."
