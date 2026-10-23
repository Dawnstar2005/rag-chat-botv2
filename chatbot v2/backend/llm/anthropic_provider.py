"""Anthropic Claude provider (official `anthropic` SDK, async streaming)."""

from collections.abc import AsyncIterator

import anthropic
import httpx2  # the SDK's HTTP client; a connection dropped mid-stream surfaces as a raw httpx2 error

from .base import LLMError, Message

# Models whose safety classifiers can decline a request; for these we opt in to
# server-side fallbacks so a declined request is re-run on a recommended model
# instead of returning a refusal.
_FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5-1"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_DEFAULT_MAX_TOKENS = 64000
_SAFE_MAX_TOKENS = 8192  # retried with this when an older model rejects the default

_ERRORS = (anthropic.AnthropicError, httpx2.TransportError)


def _detail(exc: anthropic.APIError) -> str:
    """The API's own error text, without the SDK's "Error code: 400 - {...}" wrapper."""
    error = exc.body.get("error") if isinstance(exc.body, dict) else None
    return str(error.get("message") or exc.message) if isinstance(error, dict) else exc.message


def _to_llm_error(exc: Exception) -> LLMError:
    # An error event inside an already started stream arrives with HTTP 200, so look at its type too.
    kind = getattr(exc, "type", None)
    if isinstance(exc, anthropic.AuthenticationError):
        return LLMError("Anthropic API түлхүүр буруу байна. Тохиргооноос шалгана уу.", 401)
    if isinstance(exc, anthropic.PermissionDeniedError):
        return LLMError("Энэ API түлхүүрт уг загварыг ашиглах эрх алга.", 403)
    if isinstance(exc, anthropic.NotFoundError):
        return LLMError("Сонгосон загвар олдсонгүй. Өөр загвар сонгоно уу.", 404)
    if isinstance(exc, anthropic.RateLimitError) or kind == "rate_limit_error":
        return LLMError("Хүсэлтийн хязгаар хэтэрлээ эсвэл кредит дууссан байна. Түр хүлээгээд дахин оролдоно уу.", 429)
    if isinstance(exc, anthropic.BadRequestError):
        return LLMError(f"Хүсэлт буруу байна: {_detail(exc)}", 400)
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code >= 500 or kind in ("overloaded_error", "api_error", "timeout_error"):
            return LLMError("Anthropic сервер түр ачаалалтай байна. Дахин оролдоно уу.", 503)
        return LLMError(f"Anthropic API алдаа ({exc.status_code}): {_detail(exc)}", 502)
    if isinstance(exc, (anthropic.APIConnectionError, httpx2.TransportError)):
        return LLMError("Anthropic API-тай холбогдож чадсангүй. Интернэт холболтоо шалгана уу.", 503)
    if isinstance(exc, anthropic.APIError):
        return LLMError(f"Anthropic API алдаа: {_detail(exc)}", 502)
    return LLMError(f"Тодорхойгүй алдаа: {exc}", 500)


class AnthropicProvider:
    id = "anthropic"
    name = "Anthropic Claude"
    default_model = "claude-opus-5"
    key_hint = "sk-ant-..."
    key_url = "https://console.anthropic.com/settings/keys"

    def __init__(self):
        self._max_tokens: dict[str, int] = {}  # per-model output cap, learned from /v1/models

    async def list_models(self, api_key: str) -> list[dict]:
        models = []
        try:
            async with anthropic.AsyncAnthropic(api_key=api_key, max_retries=1, timeout=30) as client:
                async for model in client.models.list(limit=100):
                    if model.max_tokens:
                        self._max_tokens[model.id] = model.max_tokens
                    models.append({"id": model.id, "name": model.display_name or model.id})
        except _ERRORS as exc:
            raise _to_llm_error(exc) from exc
        return models

    async def stream_chat(
        self, api_key: str, model: str, system: str, messages: list[Message]
    ) -> AsyncIterator[str]:
        max_tokens = min(self._max_tokens.get(model, _DEFAULT_MAX_TOKENS), _DEFAULT_MAX_TOKENS)
        async with anthropic.AsyncAnthropic(api_key=api_key) as client:
            for attempt in range(2):
                produced = False
                try:
                    if model in _FALLBACK_MODELS:
                        manager = client.beta.messages.stream(
                            model=model,
                            max_tokens=max_tokens,
                            system=system,
                            messages=messages,
                            betas=[_FALLBACK_BETA],
                            fallbacks="default",
                        )
                    else:
                        manager = client.messages.stream(
                            model=model, max_tokens=max_tokens, system=system, messages=messages
                        )
                    async with manager as stream:
                        async for text in stream.text_stream:
                            produced = True
                            yield text
                        final = await stream.get_final_message()
                except anthropic.BadRequestError as exc:
                    # Older models have a lower output cap than our default.
                    if attempt == 0 and not produced and "max_tokens" in str(exc.message):
                        max_tokens = _SAFE_MAX_TOKENS
                        continue
                    raise _to_llm_error(exc) from exc
                except _ERRORS as exc:
                    raise _to_llm_error(exc) from exc
                break

        if final.stop_reason == "refusal":
            yield "\n\n> ⚠️ Загвар аюулгүй байдлын шалтгаанаар энэ хүсэлтэд хариулахаас татгалзлаа."
        elif final.stop_reason == "max_tokens":
            yield "\n\n> ⚠️ Хариулт хэт урт тул таслагдлаа."
