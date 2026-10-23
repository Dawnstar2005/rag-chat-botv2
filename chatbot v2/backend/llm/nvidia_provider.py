"""NVIDIA API provider (build.nvidia.com): an OpenAI-compatible endpoint, used through the `openai` SDK.

The hosted catalog differs from OpenAI's in a few ways handled here:
- GET /v1/models is public and lists the whole catalog, yet most entries answer 404
  ("Function ... not found for account") for a given free account, and some popular ones sit
  in a queue for minutes. Listing therefore sends every model a one-token request: missing
  models are dropped, slow ones are marked unavailable, and a rejected key is reported.
- A wrong key is answered with 403 "Authorization failed", a retired model with 410 Gone.
- Error bodies carry `detail` (or a bare string), not OpenAI's `error.message`.
- Some reasoning models put their thinking in the answer text inside <think> tags.
"""

import asyncio
import hashlib
import logging
import time
from collections.abc import AsyncIterator

import openai

from .base import LLMError, Message

log = logging.getLogger(__name__)

BASE_URL = "https://integrate.api.nvidia.com/v1"

_BAD_KEY = "NVIDIA API түлхүүр буруу байна (эсвэл энэ загварт эрх алга). Тохиргооноос шалгана уу."

# Catalog entries that are not text chat models: embedders, rerankers, guard/safety
# classifiers, vision and parsing models, code-completion models.
_NON_CHAT = (
    "embed", "rerank", "reward", "guard", "safety", "topic-control", "jailbreak", "pii",
    "clip", "vila", "neva", "kosmos", "deplot", "fuyu", "vision", "vlm", "-vl-", "omni", "cosmos",
    "parse", "translate", "detector", "calibration", "diffusion",
    "starcoder", "codegemma", "codellama", "codestral", "coder", "code-instruct",
)
# Best first, from Mongolian RAG questions on a free account (2026-10-24): minimax-m3.1-flash
# answered correctly with citations in 1-6 s, nemotron-3-ultra well in 5-12 s, muse-glimmer
# well but slower, nemotron-3.5-bolt very fast but terser. The rest did not answer then or wrote
# poor Mongolian. Prefixes rather than exact ids, because the catalog renames versions often.
_PREFERRED = (
    "minimaxai/minimax-m", "nvidia/nemotron-3-ultra", "meta/muse-glimmer", "nvidia/nemotron-3.5-bolt",
    "deepseek-ai/deepseek-v", "moonshotai/kimi-k", "z-ai/glm-", "stepfun-ai/step-", "nvidia/nemotron-3",
    "mistralai/mistral-large", "openai/gpt-oss",
)

_MAX_TOKENS = 8192  # reasoning models spend part of this on thinking
_TEMPERATURE = 0.3  # low: answers should stay close to the retrieved excerpts
_CHAT_TIMEOUT = 90  # s without a byte from a queued model before giving up (the SDK default is 10 min)
_PROBE_TIMEOUT = 10  # s; a model that takes longer to answer one token is listed as unavailable
_PROBE_CONCURRENCY = 12
_CACHE_SECONDS = 600  # reuse a key's probe results while the settings dialog is reopened

_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def _detail(exc: openai.APIError) -> str:
    body = exc.body
    if isinstance(body, dict):
        error = body.get("error")
        text = body.get("detail") or body.get("message") or (error.get("message") if isinstance(error, dict) else error)
        if text:
            return str(text)
    if isinstance(body, str) and body.strip():
        return body.strip()
    return exc.message


def _to_llm_error(exc: Exception) -> LLMError:
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return LLMError(_BAD_KEY, 401)
    if isinstance(exc, openai.NotFoundError):
        return LLMError("Сонгосон загвар олдсонгүй. Тохиргооноос өөр загвар сонгоно уу.", 404)
    if isinstance(exc, openai.RateLimitError):
        return LLMError("NVIDIA-ийн хүсэлтийн хязгаар хэтэрлээ (үнэгүй эрх минутад ~40 хүсэлт). Түр хүлээгээд дахин оролдоно уу.", 429)
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code == 410:
            return LLMError("Сонгосон загварыг NVIDIA зогсоосон байна. Тохиргооноос өөр загвар сонгоно уу.", 404)
        if exc.status_code in (400, 422):
            return LLMError(f"Хүсэлт буруу байна: {_detail(exc)}", 400)
        if exc.status_code >= 500:
            return LLMError("NVIDIA сервер түр ачаалалтай байна. Дахин оролдоно уу.", 503)
        return LLMError(f"NVIDIA API алдаа ({exc.status_code}): {_detail(exc)}", 502)
    if isinstance(exc, openai.APITimeoutError):  # before APIConnectionError, its base class
        return LLMError(
            f"Загвар {_CHAT_TIMEOUT} секундэд хариу өгсөнгүй — NVIDIA дээр дараалалд хүлээгдэж байна. "
            "Тохиргооноос өөр загвар сонгоно уу.",
            504,
        )
    if isinstance(exc, openai.APIConnectionError):
        return LLMError("NVIDIA API-тай холбогдож чадсангүй. Интернэт холболтоо шалгана уу.", 503)
    if isinstance(exc, openai.APIError):  # an error event inside an already started stream
        return LLMError(f"NVIDIA API алдаа: {_detail(exc)}", 502)
    return LLMError(f"Тодорхойгүй алдаа: {exc}", 500)


def _client(api_key: str, **options) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key=api_key, base_url=BASE_URL, **options)


def _is_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(word in lowered for word in _NON_CHAT)


def _order(ids: list[str]) -> list[str]:
    """Preferred families first (newest-looking id first within a family), then the rest."""
    preferred = []
    for prefix in _PREFERRED:
        preferred += sorted((i for i in ids if i.startswith(prefix) and i not in preferred), reverse=True)
    return preferred + sorted(i for i in ids if i not in preferred)


class _ThinkFilter:
    """Drops a <think>...</think> block that opens the answer; everything else passes through."""

    def __init__(self):
        self._held = ""  # text kept back until it is clear whether it belongs to a think block
        self._state = "start"  # start -> thinking -> answer, or start -> answer

    def feed(self, text: str) -> str:
        if self._state == "answer":
            return text
        self._held += text
        if self._state == "start":
            head = self._held.lstrip()
            if head.startswith(_THINK_OPEN):
                self._state = "thinking"
            elif _THINK_OPEN.startswith(head):
                return ""  # empty, or a prefix such as "<th" that may still become the tag
            else:
                self._state = "answer"
                text, self._held = self._held, ""
                return text
        end = self._held.find(_THINK_CLOSE)
        if end < 0:
            self._held = self._held[-len(_THINK_CLOSE):]  # enough to spot a tag split across chunks
            return ""
        self._state = "answer"
        text, self._held = self._held[end + len(_THINK_CLOSE):].lstrip(), ""
        return text

    def flush(self) -> str:
        # Text held at the start was answer text after all (e.g. a lone "<"); unfinished thinking is dropped.
        text, self._held = (self._held if self._state == "start" else ""), ""
        return text


async def _probe(client: openai.AsyncOpenAI, model: str) -> str:
    """One-token request: "ready", "busy" (queued, rate limited, failing), "gone" or "denied"."""
    try:
        await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=1, timeout=_PROBE_TIMEOUT
        )
    except (openai.AuthenticationError, openai.PermissionDeniedError):
        return "denied"
    except openai.NotFoundError:
        return "gone"
    except (openai.BadRequestError, openai.UnprocessableEntityError):
        return "ready"  # it answered; it just wants other options, which chat retries without
    except openai.APIStatusError as exc:
        return "gone" if exc.status_code == 410 else "busy"
    except openai.OpenAIError:  # timeout or connection trouble
        return "busy"
    return "ready"


class NvidiaProvider:
    id = "nvidia"
    name = "NVIDIA"
    default_model = "minimaxai/minimax-m3.1-flash"
    key_hint = "nvapi-..."
    key_url = "https://build.nvidia.com/settings/api-keys"

    def __init__(self):
        self._cache: dict[str, tuple[float, list[dict]]] = {}  # sha256(key) -> (time, models)

    async def list_models(self, api_key: str) -> list[dict]:
        """Chat models this key can use, answering ones first; `available` is False for models
        that exist but did not answer within a few seconds."""
        digest = hashlib.sha256(api_key.encode()).hexdigest()
        if (cached := self._cache.get(digest)) and time.monotonic() - cached[0] < _CACHE_SECONDS:
            return cached[1]
        try:
            async with _client(api_key, max_retries=0, timeout=30) as client:
                ids = _order([m.id async for m in client.models.list() if _is_chat_model(m.id)])
                limit = asyncio.Semaphore(_PROBE_CONCURRENCY)

                async def probe(model: str) -> str:
                    async with limit:
                        return await _probe(client, model)

                states = dict(zip(ids, await asyncio.gather(*(probe(m) for m in ids))))
        except openai.OpenAIError as exc:
            raise _to_llm_error(exc) from exc
        if "denied" in states.values() and "ready" not in states.values():
            raise LLMError(_BAD_KEY, 401)  # a wrong key is refused on every model, before any queue
        models = [{"id": i, "name": i, "available": states[i] == "ready"} for i in ids if states[i] in ("ready", "busy")]
        models.sort(key=lambda m: not m["available"])  # stable: keeps the preferred order within each group
        log.info("NVIDIA: %d of %d models answer for this key", sum(m["available"] for m in models), len(ids))
        self._cache[digest] = (time.monotonic(), models)
        return models

    async def stream_chat(
        self, api_key: str, model: str, system: str, messages: list[Message]
    ) -> AsyncIterator[str]:
        options = {"max_tokens": _MAX_TOKENS, "temperature": _TEMPERATURE}
        think = _ThinkFilter()
        finish_reason = None
        # No SDK retries: they would repeat a queued request's wait.
        async with _client(api_key, max_retries=0, timeout=_CHAT_TIMEOUT) as client:
            for attempt in range(2):
                produced = False
                try:
                    stream = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": system}, *messages],
                        stream=True,
                        **options,
                    )
                    async for chunk in stream:
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        # Reasoning models stream their thinking in a separate `reasoning_content`
                        # field; only the answer (`content`) is shown.
                        if choice.delta and choice.delta.content:
                            produced = True
                            if text := think.feed(choice.delta.content):
                                yield text
                        if choice.finish_reason:
                            finish_reason = choice.finish_reason
                except (openai.BadRequestError, openai.UnprocessableEntityError) as exc:
                    # A small model may reject our max_tokens (it exceeds its context) or the
                    # temperature; retry once with the server's defaults.
                    if attempt == 0 and not produced:
                        log.info("NVIDIA %s rejected the options, retrying without them: %s", model, exc)
                        options = {}
                        continue
                    raise _to_llm_error(exc) from exc
                except openai.OpenAIError as exc:
                    raise _to_llm_error(exc) from exc
                break

        if text := think.flush():
            yield text
        if finish_reason == "length":
            yield "\n\n> ⚠️ Хариулт хэт урт тул таслагдлаа."
        elif finish_reason == "content_filter":
            yield "\n\n> ⚠️ Хариулт агуулгын шүүлтүүрээр зогсоогдлоо."
