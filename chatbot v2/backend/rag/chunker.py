"""Split text into overlapping chunks that respect paragraph and sentence boundaries."""

import re

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_SPACES = re.compile(r"[ \t\f\v ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def _hard_split(text: str, size: int) -> list[str]:
    """Split by words; slice words that are themselves longer than `size`."""
    pieces, current = [], ""
    for word in text.split():
        while len(word) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:size])
            word = word[size:]
        if current and len(current) + 1 + len(word) > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        pieces.append(current)
    return pieces


def _units(text: str, size: int) -> list[tuple[str, bool]]:
    """Return (unit_text, starts_paragraph) pairs, each unit no longer than `size`."""
    units: list[tuple[str, bool]] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.replace("\n", " ").strip()
        if not paragraph:
            continue
        if len(paragraph) <= size:
            units.append((paragraph, True))
            continue
        first = True
        for sentence in _SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            parts = [sentence] if len(sentence) <= size else _hard_split(sentence, size)
            for part in parts:
                units.append((part, first))
                first = False
    return units


def _join(units: list[tuple[str, bool]]) -> str:
    out = []
    for i, (text, para) in enumerate(units):
        if i:
            out.append("\n\n" if para else " ")
        out.append(text)
    return "".join(out)


def _length(units: list[tuple[str, bool]]) -> int:
    return sum(len(t) for t, _ in units) + 2 * max(len(units) - 1, 0)


def _overlap_tail(units: list[tuple[str, bool]], overlap: int) -> list[tuple[str, bool]]:
    """Trailing context to repeat at the start of the next chunk: whole units when they
    fit in `overlap`, otherwise the last words of the final unit (a whole paragraph is
    usually longer than `overlap`, which would leave no overlap at all)."""
    tail: list[tuple[str, bool]] = []
    for prev in reversed(units):
        if _length([prev] + tail) > overlap:
            break
        tail.insert(0, prev)
    if tail or not units:
        return tail
    suffix = ""
    for word in reversed(units[-1][0].split(" ")):
        candidate = f"{word} {suffix}" if suffix else word
        if len(candidate) > overlap:
            break
        suffix = candidate
    return [(suffix, False)] if suffix else []


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Pack paragraph/sentence units into chunks of at most ~chunk_size characters.

    Consecutive chunks share up to `overlap` characters of trailing context so an
    answer that straddles a boundary is still retrievable.
    """
    chunk_size = max(chunk_size, 100)
    overlap = max(0, min(overlap, chunk_size // 2))
    text = normalize(text)
    if not text:
        return []

    chunks: list[str] = []
    current: list[tuple[str, bool]] = []
    for unit in _units(text, chunk_size):
        if current and _length(current + [unit]) > chunk_size:
            chunks.append(_join(current))
            current = _overlap_tail(current, overlap)
            while current and _length(current + [unit]) > chunk_size:
                current.pop(0)
        current.append(unit)
    if current:
        chunks.append(_join(current))
    return chunks
