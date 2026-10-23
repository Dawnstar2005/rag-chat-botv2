"""Document store: SQLite for persistence, an in-memory snapshot for fast hybrid search."""

import hashlib
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25, tokenize
from .chunker import split_text
from .embeddings import Embedder
from .loaders import DocumentLoadError, Page

log = logging.getLogger(__name__)

RRF_K = 60  # standard Reciprocal Rank Fusion constant
CANDIDATES = 30  # how many hits each retriever contributes before fusion
MIN_SIMILARITY = 0.30  # cosine floor below which a dense hit is treated as unrelated

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    size INTEGER NOT NULL,
    pages INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    char_count INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    page INTEGER,
    text TEXT NOT NULL,
    embedding BLOB,
    embedding_model TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


@dataclass
class _Snapshot:
    chunk_ids: list[int]
    doc_ids: list[str]
    doc_names: list[str]
    pages: list[int | None]
    texts: list[str]
    bm25: BM25
    matrix: np.ndarray | None  # N x dim, zero rows where a chunk has no vector yet
    has_vector: np.ndarray  # bool mask, N


class DocumentStore:
    def __init__(self, db_path: Path, embedder: Embedder, chunk_size: int, chunk_overlap: int):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)
        self._snapshot: _Snapshot | None = None
        self._closed = False

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._conn.close()

    @property
    def closed(self) -> bool:
        return self._closed

    # ---------- documents ----------

    def has_document(self, data: bytes) -> bool:
        """Whether a file with exactly these bytes is already stored (checked before parsing it)."""
        digest = hashlib.sha256(data).hexdigest()
        with self._lock:
            return self._conn.execute("SELECT 1 FROM documents WHERE sha256 = ?", (digest,)).fetchone() is not None

    def add_document(self, name: str, data: bytes, pages: list[Page]) -> tuple[dict, bool]:
        """Chunk, embed and store a document. Returns (document, created)."""
        digest = hashlib.sha256(data).hexdigest()
        with self._lock:
            row = self._conn.execute("SELECT * FROM documents WHERE sha256 = ?", (digest,)).fetchone()
        if row:
            return dict(row), False

        chunks: list[tuple[int | None, str]] = []
        for page in pages:
            for text in split_text(page.text, self.chunk_size, self.chunk_overlap):
                chunks.append((page.number, text))
        if not chunks:
            # e.g. a file of NUL bytes: non-blank to the loader, empty after normalisation.
            raise DocumentLoadError("Файлаас текст олдсонгүй.")

        # Embedding is the slow part; do it outside the lock.
        vectors = self.embedder.embed([text for _, text in chunks])
        model = self.embedder.model_name if vectors is not None else None

        doc = {
            "id": uuid.uuid4().hex,
            "name": name,
            "sha256": digest,
            "size": len(data),
            "pages": len(pages),
            "chunk_count": len(chunks),
            "char_count": sum(len(t) for _, t in chunks),
            "created_at": time.time(),
        }
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        "INSERT INTO documents VALUES (:id, :name, :sha256, :size, :pages, :chunk_count, :char_count, :created_at)",
                        doc,
                    )
                    self._conn.executemany(
                        "INSERT INTO chunks (doc_id, ord, page, text, embedding, embedding_model) VALUES (?, ?, ?, ?, ?, ?)",
                        [
                            (doc["id"], i, page, text, vectors[i].tobytes() if vectors is not None else None, model)
                            for i, (page, text) in enumerate(chunks)
                        ],
                    )
            except sqlite3.IntegrityError:
                # The same file was uploaded twice concurrently; the other request won.
                row = self._conn.execute("SELECT * FROM documents WHERE sha256 = ?", (digest,)).fetchone()
                return dict(row), False
            self._snapshot = None
        if vectors is None and self.embedder.ready:
            # The model finished loading while this file was being processed.
            self.backfill_embeddings()
        return doc, True

    def list_documents(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT d.*, SUM(c.embedding IS NOT NULL AND c.embedding_model = ?) AS embedded
                   FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id
                   GROUP BY d.id ORDER BY d.created_at DESC""",
                (self.embedder.model_name,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: str) -> bool:
        with self._lock, self._conn:
            deleted = self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,)).rowcount
            if deleted:
                self._snapshot = None
        return bool(deleted)

    def stats(self) -> dict:
        with self._lock:
            docs = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": docs, "chunks": chunks}

    def backfill_embeddings(self, batch: int = 128) -> int:
        """Embed chunks stored while the model was unavailable (or by a different model)."""
        if not self.embedder.ready:
            return 0
        model = self.embedder.model_name
        done = 0
        while True:
            with self._lock:
                if self._closed:  # the app shut down while the model was loading
                    break
                rows = self._conn.execute(
                    "SELECT id, text FROM chunks WHERE embedding IS NULL OR embedding_model IS NOT ? LIMIT ?",
                    (model, batch),
                ).fetchall()
            if not rows:
                break
            vectors = self.embedder.embed([r["text"] for r in rows])
            if vectors is None:
                break
            with self._lock:
                if self._closed:
                    break
                with self._conn:
                    self._conn.executemany(
                        "UPDATE chunks SET embedding = ?, embedding_model = ? WHERE id = ?",
                        [(vectors[i].tobytes(), model, r["id"]) for i, r in enumerate(rows)],
                    )
                self._snapshot = None
            done += len(rows)
        if done:
            log.info("Backfilled embeddings for %d chunks", done)
        return done

    # ---------- search ----------

    def _get_snapshot(self) -> _Snapshot:
        with self._lock:
            if self._snapshot is not None:
                return self._snapshot
            rows = self._conn.execute(
                """SELECT c.id, c.doc_id, d.name, c.page, c.text, c.embedding, c.embedding_model
                   FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY c.id"""
            ).fetchall()
            model = self.embedder.model_name
            vectors = [
                np.frombuffer(r["embedding"], dtype=np.float32)
                if r["embedding"] is not None and r["embedding_model"] == model
                else None
                for r in rows
            ]
            dim = next((v.shape[0] for v in vectors if v is not None), None)
            matrix = None
            has_vector = np.zeros(len(rows), dtype=bool)
            if dim:
                matrix = np.zeros((len(rows), dim), dtype=np.float32)
                for i, v in enumerate(vectors):
                    if v is not None and v.shape[0] == dim:
                        matrix[i] = v
                        has_vector[i] = True
            self._snapshot = _Snapshot(
                chunk_ids=[r["id"] for r in rows],
                doc_ids=[r["doc_id"] for r in rows],
                doc_names=[r["name"] for r in rows],
                pages=[r["page"] for r in rows],
                texts=[r["text"] for r in rows],
                bm25=BM25([tokenize(r["text"]) for r in rows]),
                matrix=matrix,
                has_vector=has_vector,
            )
            return self._snapshot

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Hybrid search: BM25 + dense cosine similarity, merged with Reciprocal Rank Fusion."""
        snap = self._get_snapshot()
        if not snap.chunk_ids or not query.strip():
            return []

        fused: dict[int, float] = {}

        lexical = snap.bm25.scores(tokenize(query))
        lexical_hits = [int(i) for i in np.argsort(-lexical)[:CANDIDATES] if lexical[i] > 0]
        for rank, i in enumerate(lexical_hits):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)

        similarity = None
        if snap.matrix is not None and snap.has_vector.any():
            query_vec = self.embedder.embed([query])
            if query_vec is not None and query_vec.shape[1] == snap.matrix.shape[1]:
                similarity = snap.matrix @ query_vec[0]
                similarity[~snap.has_vector] = -1.0
                dense_hits = [
                    int(i) for i in np.argsort(-similarity)[:CANDIDATES] if similarity[i] >= MIN_SIMILARITY
                ]
                for rank, i in enumerate(dense_hits):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
        best = ranked[0][1] if ranked else 1.0
        return [
            {
                "chunk_id": snap.chunk_ids[i],
                "doc_id": snap.doc_ids[i],
                "doc_name": snap.doc_names[i],
                "page": snap.pages[i],
                "text": snap.texts[i],
                "score": round(score / best, 4),
                "bm25": round(float(lexical[i]), 4),
                "similarity": round(float(similarity[i]), 4) if similarity is not None else None,
            }
            for i, score in ranked
        ]
