"""Okapi BM25 keyword search.

Mongolian is agglutinative ("цөл", "цөлд", "цөлийн", "цөлөөс"), so words are indexed by
a light stem: common case and plural endings are stripped. Long words also emit a
5-character prefix token, which catches stem changes the suffix list misses
("сургууль" / "сургуулийн"). Both improve recall noticeably.
"""

import math
import re
from collections import Counter, defaultdict

import numpy as np

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_PREFIX_LEN = 5
_MIN_STEM = 3

# Case endings (genitive, dative, accusative, ablative, instrumental, comitative,
# directional) and plurals, longest first.
_SUFFIXES = sorted(
    """ын ийн ины ны ний  д т ад эд од өд ид  ыг ийг  аас ээс оос өөс  аар ээр оор өөр
    тай тэй той  руу рүү луу лүү  ууд үүд иуд нууд нүүд""".split(),
    key=len,
    reverse=True,
)

# Question words: frequent in queries, rare in documents (so a stray match scores high),
# and some would stem onto content words ("хаанаас" -> "хаан").
_STOPWORDS = frozenset(
    "вэ бэ уу үү юу юуг хэн хэний хэнд хэзээ хаана хаанаас хаашаа ямар яаж яагаад "
    "хэрхэн хэр хэд хэдэн аль талаар тухай".split()
)


def stem(word: str) -> str:
    """Strip endings until none applies, then a final soft sign, so a bare word and its
    inflections meet ("говь", "говийн", "говьд" -> "гов"). Stems keep >= 3 letters."""
    while suffix := next((s for s in _SUFFIXES if word.endswith(s) and len(word) - len(s) >= _MIN_STEM), None):
        word = word[: -len(suffix)]
    if word.endswith("ь") and len(word) > _MIN_STEM:
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token.isdigit():
            tokens.append(token)
            continue
        if len(token) < 2 or token in _STOPWORDS:
            continue
        # The stem replaces the surface form: indexing both would let an exact match on a
        # generic word ("жилд") outrank stem matches on the topical one ("цөлд" ~ "цөлийн").
        tokens.append(stem(token))
        if len(token) > _PREFIX_LEN:
            tokens.append("~" + token[:_PREFIX_LEN])
    return tokens


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.size = len(documents)
        self.doc_len = np.array([len(d) for d in documents], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.size else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, tokens in enumerate(documents):
            for term, freq in Counter(tokens).items():
                self.postings[term].append((index, freq))
        self.idf = {
            term: math.log(1 + (self.size - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.size, dtype=np.float32)
        if not self.size or not self.avgdl:
            return scores
        for term in set(query_tokens):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for index, freq in postings:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[index] / self.avgdl)
                scores[index] += idf * freq * (self.k1 + 1) / (freq + norm)
        return scores
