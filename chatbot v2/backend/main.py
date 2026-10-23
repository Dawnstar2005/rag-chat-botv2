"""FastAPI application: REST + Server-Sent Events API and the static web UI."""

import json
import logging
import mimetypes
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .config import Settings, settings as default_settings
from .llm.base import LLMError
from .llm.prompts import SYSTEM_PROMPT, build_messages, retrieval_query
from .llm.registry import PROVIDERS, get_provider, has_env_key, resolve_api_key
from .rag.embeddings import Embedder
from .rag.folder import import_folder
from .rag.loaders import SUPPORTED_EXTENSIONS, DocumentLoadError, load_document
from .rag.store import DocumentStore

log = logging.getLogger("chatbot")

# On Windows, mimetypes takes types from the registry, where some software maps .js to text/plain;
# browsers then refuse to run the UI's <script type="module">. Pin the types the UI depends on.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")

MAX_MESSAGE_CHARS = 20_000


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    provider: str = "nvidia"
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)  # the UI sends at most 50
    top_k: int | None = Field(default=None, ge=1, le=20)  # None -> TOP_K setting
    use_rag: bool = True

    @field_validator("messages")
    @classmethod
    def last_message_is_user(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        if messages[-1].role != "user":
            raise ValueError("сүүлийн мессеж хэрэглэгчийнх байх ёстой")
        return messages


class ModelsRequest(BaseModel):
    provider: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    top_k: int | None = Field(default=None, ge=1, le=20)  # None -> TOP_K setting


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(settings: Settings = default_settings) -> FastAPI:
    embedder = Embedder(settings.embedding_model, str(settings.models_dir), settings.embeddings_enabled)
    store = DocumentStore(settings.db_path, embedder, settings.chunk_size, settings.chunk_overlap)

    def warm_up() -> None:
        # Folder files first, so keyword search covers them before the embedding model is loaded;
        # the back-fill then embeds them along with anything else that is missing.
        import_folder(store, settings.docs_dir, settings.max_upload_mb * 1024 * 1024)
        if embedder.load():
            store.backfill_embeddings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        worker = threading.Thread(target=warm_up, name="warmup", daemon=True)
        worker.start()
        yield
        # A still-running back-fill sees the closed store and stops on its own.
        await run_in_threadpool(worker.join, 5)
        store.close()

    app = FastAPI(title="RAG Chatbot", version="2.0", lifespan=lifespan)
    app.state.store = store
    app.state.embedder = embedder

    @app.exception_handler(LLMError)
    async def llm_error_handler(_, exc: LLMError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    # ---------- system ----------

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "embeddings": {"status": embedder.status, "model": embedder.model_name, "error": embedder.error},
            **store.stats(),
        }

    @app.get("/api/providers")
    def providers():
        return {
            "providers": [
                {
                    "id": p.id,
                    "name": p.name,
                    "default_model": p.default_model,
                    "key_hint": p.key_hint,
                    "key_url": p.key_url,
                    "server_key": has_env_key(p.id),
                }
                for p in PROVIDERS.values()
            ]
        }

    @app.post("/api/models")
    async def models(req: ModelsRequest, x_api_key: str | None = Header(default=None)):
        provider = get_provider(req.provider)
        api_key = resolve_api_key(req.provider, x_api_key)
        found = await provider.list_models(api_key)
        # A provider may mark models that exist but are not answering right now ("available": False).
        ids = [m["id"] for m in found]
        ready = [m["id"] for m in found if m.get("available", True)] or ids
        default = provider.default_model if provider.default_model in ready else (ready[0] if ready else None)
        return {"models": found, "default": default}

    # ---------- documents ----------

    @app.get("/api/documents")
    def list_documents():
        return {"documents": store.list_documents(), "embeddings": embedder.status}

    @app.post("/api/documents")
    async def upload_documents(files: list[UploadFile] = File(...)):
        limit = settings.max_upload_mb * 1024 * 1024
        results = []
        for upload in files:
            name = upload.filename or "нэргүй"
            data = await upload.read(limit + 1)
            if len(data) > limit:
                results.append({"name": name, "status": "error", "error": f"Файл {settings.max_upload_mb} MB-аас их байна."})
                continue
            try:
                pages = await run_in_threadpool(load_document, name, data)
                doc, created = await run_in_threadpool(store.add_document, name, data, pages)
            except DocumentLoadError as exc:
                results.append({"name": name, "status": "error", "error": str(exc)})
                continue
            except Exception:
                log.exception("Failed to index %s", name)
                results.append({"name": name, "status": "error", "error": "Файлыг боловсруулахад алдаа гарлаа."})
                continue
            results.append({"name": name, "status": "added" if created else "duplicate", "document": doc})
        return {"results": results}

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: str):
        if not store.delete_document(doc_id):
            raise HTTPException(status_code=404, detail="Баримт олдсонгүй.")
        return {"ok": True}

    @app.get("/api/formats")
    def formats():
        return {"extensions": sorted(SUPPORTED_EXTENSIONS), "max_upload_mb": settings.max_upload_mb}

    @app.post("/api/search")
    async def search(req: SearchRequest):
        return {"results": await run_in_threadpool(store.search, req.query, req.top_k or settings.top_k)}

    # ---------- chat ----------

    @app.post("/api/chat")
    async def chat(req: ChatRequest, x_api_key: str | None = Header(default=None)):
        # Validate everything that can fail fast before opening the stream,
        # so these errors arrive as normal HTTP status codes.
        provider = get_provider(req.provider)
        api_key = resolve_api_key(req.provider, x_api_key)
        model = (req.model or "").strip() or provider.default_model

        history = [m.model_dump() for m in req.messages]
        chunks = []
        if req.use_rag:
            chunks = await run_in_threadpool(store.search, retrieval_query(history), req.top_k or settings.top_k)
        messages = build_messages(history, chunks, settings.max_history_messages)

        sources = [
            {"index": i, "doc_id": c["doc_id"], "doc_name": c["doc_name"], "page": c["page"], "text": c["text"], "score": c["score"]}
            for i, c in enumerate(chunks, start=1)
        ]

        async def events():
            yield _sse("sources", {"sources": sources, "model": model})
            try:
                async for text in provider.stream_chat(api_key, model, SYSTEM_PROMPT, messages):
                    yield _sse("delta", {"text": text})
            except LLMError as exc:
                yield _sse("error", {"message": exc.message, "status": exc.status})
                return
            except Exception:
                log.exception("Chat stream failed")
                yield _sse("error", {"message": "Серверийн дотоод алдаа гарлаа.", "status": 500})
                return
            yield _sse("done", {})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # The web UI. Mounted last so /api routes take precedence.
    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    return app


app = create_app()
