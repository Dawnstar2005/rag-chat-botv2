"""Local sentence embeddings via fastembed (ONNX Runtime, CPU, no API key needed).

The model is downloaded once into data/models. Loading happens in a background
thread so the server starts instantly; until it is ready, search falls back to
BM25 keyword search only.
"""

import logging
import os
import threading

import numpy as np

# Windows without Developer Mode cannot create symlinks; the HF cache still works, just noisily.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str, cache_dir: str, enabled: bool = True):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.status = "loading" if enabled else "disabled"
        self.error: str | None = None
        self._model = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        """Blocking load. Safe to call more than once."""
        if self.status == "disabled" or self.ready:
            return self.ready
        try:
            from fastembed import TextEmbedding

            log.info("Embedding model loading: %s", self.model_name)
            self._model = TextEmbedding(self.model_name, cache_dir=self.cache_dir)
            self.status = "ready"
            log.info("Embedding model ready")
        except Exception as exc:  # network down, disk full, unsupported model...
            self.status = "error"
            self.error = str(exc)
            log.warning("Embedding model failed to load, keyword search only: %s", exc)
        return self.ready

    def embed(self, texts: list[str]) -> np.ndarray | None:
        """Return L2-normalised float32 vectors (len(texts) x dim), or None if unavailable."""
        if not self.ready or not texts:
            return None
        with self._lock:
            vectors = np.asarray(list(self._model.embed(texts, batch_size=32)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
