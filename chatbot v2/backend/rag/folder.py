"""Index the files the user placed in the documents folder (scanned once at startup)."""

import logging
from pathlib import Path

from .loaders import SUPPORTED_EXTENSIONS, DocumentLoadError, load_document
from .store import DocumentStore

log = logging.getLogger(__name__)


def import_folder(store: DocumentStore, folder: Path, max_bytes: int) -> int:
    """Add every supported file under `folder` (subfolders included) that is not stored yet.
    Files already indexed are recognised by content, so a restart re-reads nothing heavy.
    Returns how many documents were added."""
    if not folder.is_dir():
        return 0
    added = 0
    for path in sorted(folder.rglob("*")):
        if store.closed:  # the app shut down mid-import
            break
        # "~$..." are Word's lock files for an open document; dot files are hidden system files.
        if path.name.startswith(("~$", ".")) or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            if not path.is_file():
                continue
            if path.stat().st_size > max_bytes:
                log.warning("Skipped %s: larger than %d MB", path, max_bytes // (1024 * 1024))
                continue
            data = path.read_bytes()
            if store.has_document(data):
                continue
            _, created = store.add_document(path.name, data, load_document(path.name, data))
        except (DocumentLoadError, OSError) as exc:
            log.warning("Skipped %s: %s", path, exc)
            continue
        except Exception:
            if store.closed:
                break
            log.exception("Failed to index %s", path)
            continue
        if created:
            added += 1
            log.info("Indexed %s", path)
    if added:
        log.info("Indexed %d new document(s) from %s", added, folder)
    return added
