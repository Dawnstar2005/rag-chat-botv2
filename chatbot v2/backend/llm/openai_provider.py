"""OpenAI provider (official `openai` SDK, Chat Completions streaming)."""

from collections.abc import AsyncIterator

import openai

from .base import LLMError, Message

# Model families in /v1/models that cannot hold a text chat. "-pro" models (o1-pro, o3-pro,
# gpt-5-pro) are served only by the Responses API, not Chat Completions.
_NON_CHAT = (
    "audio", "realtime", "tts", "transcribe", "whisper", "embedding", "dall-e",
    "image", "moderation", "search", "davinci", "babbage", "instruct", "codex", "computer-use", "-pro",
)
_CHAT_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")
# Preferred defaults, best first; the first one the key can access is pre-selected.
_PREFERRED = ("gpt-5.1", "gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini")


def _detail(exc: openai.APIError) -> str:
    """The API's own error text, without the SDK's "Error code: 400 - {...}" wrapper."""
    body = exc.body
    return str(body["message"]) if isinstance(body, dict) and body.get("message") else exc.message


def _to_llm_error(exc: Exception) -> LLMError:
    if isinstance(exc, openai.AuthenticationError):
        return LLMError("OpenAI API түлхүүр буруу байна. Тохиргооноос шалгана уу.", 401)
    if isinstance(exc, openai.PermissionDeniedError):
        return LLMError("Энэ API түлхүүрт уг загварыг ашиглах эрх алга.", 403)
    if isinstance(exc, openai.NotFoundError):
        return LLMError("Сонгосон загвар олдсонгүй. Өөр загвар сонгоно уу.", 404)
    if isinstance(exc, openai.RateLimitError):
        return LLMError("Хүсэлтийн хязгаар хэтэрлээ эсвэл кредит дууссан байна. Түр хүлээгээд дахин оролдоно уу.", 429)
    if isinstance(exc, openai.BadRequestError):
        return LLMError(f"Хүсэлт буруу байна: {_detail(exc)}", 400)
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code >= 500:
            return LLMError("OpenAI сервер түр ачаалалтай байна. Дахин оролдоно уу.", 503)
        return LLMError(f"OpenAI API алдаа ({exc.status_code}): {_detail(exc)}", 502)
    if isinstance(exc, openai.APIConnectionError):
        return LLMError("OpenAI API-тай холбогдож чадсангүй. Интернэт холболтоо шалгана уу.", 503)
    if isinstance(exc, openai.APIError):  # an error event inside an already started stream
        if getattr(exc, "type", None) == "server_error":  # only set when the body is a dict
            return LLMError("OpenAI сервер түр ачаалалтай байна. Дахин оролдоно уу.", 503)
        return LLMError(f"OpenAI API алдаа: {_detail(exc)}", 502)
    return LLMError(f"Тодорхойгүй алдаа: {exc}", 500)


def _is_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return lowered.startswith(_CHAT_PREFIXES) and not any(word in lowered for word in _NON_CHAT)


class OpenAIProvider:
    id = "openai"
    name = "OpenAI GPT"
    default_model = "gpt-5-mini"
    key_hint = "sk-..."
    key_url = "https://platform.openai.com/api-keys"

    async def list_models(self, api_key: str) -> list[dict]:
        try:
            async with openai.AsyncOpenAI(api_key=api_key, max_retries=1, timeout=30) as client:
                ids = [m.id async for m in client.models.list() if _is_chat_model(m.id)]
        except openai.OpenAIError as exc:
            raise _to_llm_error(exc) from exc
        preferred = [p for p in _PREFERRED if p in ids]
        ordered = preferred + sorted((i for i in ids if i not in preferred), reverse=True)
        return [{"id": i, "name": i} for i in ordered]

    async def stream_chat(
        self, api_key: str, model: str, system: str, messages: list[Message]
    ) -> AsyncIterator[str]:
        try:
            async with openai.AsyncOpenAI(api_key=api_key) as client:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system}, *messages],
                    stream=True,
                )
                finish_reason = None
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.delta and choice.delta.content:
                        yield choice.delta.content
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
        except openai.OpenAIError as exc:
            raise _to_llm_error(exc) from exc

        if finish_reason == "length":
            yield "\n\n> ⚠️ Хариулт хэт урт тул таслагдлаа."
        elif finish_reason == "content_filter":
            yield "\n\n> ⚠️ Хариулт агуулгын шүүлтүүрээр зогсоогдлоо."
