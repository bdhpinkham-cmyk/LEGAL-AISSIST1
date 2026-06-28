"""
rag.py
======
Local Retrieval-Augmented-Generation store.

The brief calls for "ChromaDB/sqlite-vec for RAG. No client data is stored on
external servers." To stay genuinely cross-platform (Windows .exe *and* Android
.apk) without shipping a heavy native vector engine or a remote embedding API,
this module implements a self-contained TF-IDF + cosine-similarity retriever
persisted in a local SQLite file. It behaves like a vector store (add / query /
delete by case) but has zero native dependencies, so it builds cleanly with
PyInstaller and Flet's Android toolchain.

If ``chromadb`` happens to be installed the architecture is unchanged — the
public API (:class:`RAGStore`) is the seam you would swap behind. The default
implementation below is fully functional on its own.

Case isolation: every chunk row carries ``case_id`` and every query filters on
it, so retrieval can never cross case boundaries.
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
from collections import Counter
from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple

import config

_LOCK = threading.Lock()
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "at", "by", "as", "that",
    "this", "it", "from", "has", "have", "had", "not", "he", "she", "they", "we",
    "you", "i", "his", "her", "their", "its", "which", "who", "whom", "will",
    "would", "could", "should", "shall", "may", "might", "do", "does", "did",
}


def _db_path() -> str:
    config.ensure_directories()
    return str(config.VECTOR_DB_PATH / "rag.sqlite3")


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_db_path(), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _tokenize(text: str) -> List[str]:
    return [
        w.lower()
        for w in _WORD_RE.findall(text)
        if len(w) > 2 and w.lower() not in _STOPWORDS
    ]


def chunk_text(
    text: str,
    size: int = config.RAG_CHUNK_SIZE,
    overlap: int = config.RAG_CHUNK_OVERLAP,
) -> List[str]:
    """Split ``text`` into overlapping character windows on sentence-ish edges."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Try to break on a sentence boundary near the window end for cleaner chunks.
        if end < len(text):
            window = text[start:end]
            boundary = max(window.rfind(". "), window.rfind("\n"), window.rfind("? "))
            if boundary > size * 0.5:
                end = start + boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


class RAGStore:
    """A local, case-isolated TF-IDF retriever."""

    def __init__(self) -> None:
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with _LOCK, _connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id     INTEGER NOT NULL,
                    document_id INTEGER,
                    source      TEXT,
                    text        TEXT NOT NULL,
                    tokens_json TEXT NOT NULL,
                    length      INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_case ON chunks(case_id);
                """
            )

    # -- write -------------------------------------------------------------
    def add_document(
        self, case_id: int, document_id: int, source: str, text: str
    ) -> int:
        """Chunk ``text`` and index every chunk under ``case_id``. Returns chunk count."""
        import json

        chunks = chunk_text(text)
        if not chunks:
            return 0
        rows = []
        for c in chunks:
            toks = _tokenize(c)
            tf = Counter(toks)
            rows.append(
                (
                    case_id,
                    document_id,
                    source,
                    c,
                    json.dumps(tf),
                    len(toks),
                )
            )
        with _LOCK, _connect() as conn:
            conn.executemany(
                """INSERT INTO chunks (case_id, document_id, source, text, tokens_json, length)
                   VALUES (?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)

    def delete_document(self, case_id: int, document_id: int) -> None:
        with _LOCK, _connect() as conn:
            conn.execute(
                "DELETE FROM chunks WHERE case_id = ? AND document_id = ?",
                (case_id, document_id),
            )

    def reset_case(self, case_id: int) -> None:
        with _LOCK, _connect() as conn:
            conn.execute("DELETE FROM chunks WHERE case_id = ?", (case_id,))

    # -- read --------------------------------------------------------------
    def query(
        self, case_id: int, query: str, top_k: int = config.RAG_TOP_K
    ) -> List[Dict[str, object]]:
        """Return the ``top_k`` most relevant chunks for the case (TF-IDF cosine)."""
        import json

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        q_tf = Counter(q_tokens)

        with _connect() as conn:
            rows = conn.execute(
                "SELECT id, document_id, source, text, tokens_json FROM chunks WHERE case_id = ?",
                (case_id,),
            ).fetchall()

        if not rows:
            return []

        # Document frequency across the case corpus.
        n_docs = len(rows)
        df: Counter = Counter()
        parsed: List[Tuple[sqlite3.Row, Dict[str, int]]] = []
        for r in rows:
            tf = json.loads(r["tokens_json"])
            parsed.append((r, tf))
            for term in tf:
                df[term] += 1

        def idf(term: str) -> float:
            return math.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1.0

        # Query vector.
        q_vec = {t: q_tf[t] * idf(t) for t in q_tf}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scored: List[Tuple[float, sqlite3.Row]] = []
        for r, tf in parsed:
            d_vec = {t: tf[t] * idf(t) for t in tf}
            d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
            dot = sum(q_vec.get(t, 0.0) * d_vec.get(t, 0.0) for t in q_vec)
            score = dot / (q_norm * d_norm)
            if score > 0:
                scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, r in scored[:top_k]:
            results.append(
                {
                    "score": round(score, 4),
                    "document_id": r["document_id"],
                    "source": r["source"],
                    "text": r["text"],
                }
            )
        return results

    def context_for(
        self, case_id: int, query: str, top_k: int = config.RAG_TOP_K
    ) -> str:
        """Return a formatted context block suitable for prompt injection."""
        hits = self.query(case_id, query, top_k)
        if not hits:
            return ""
        blocks = []
        for i, h in enumerate(hits, 1):
            blocks.append(f"[Source {i}: {h['source']}]\n{h['text']}")
        return "\n\n".join(blocks)


# A module-level singleton is convenient for the UI layer.
STORE = RAGStore()
