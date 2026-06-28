"""
database.py
===========
Local SQLite persistence for the Pro Se Legal Intelligence app.

Design notes / pre-debugging decisions
--------------------------------------
* **Case isolation:** Every table that holds case data carries a ``case_id``
  foreign key, and *every* query in this module that reads or writes case data
  filters by ``case_id``. There is no global "documents" accessor that could
  leak one case's data into another. ``settings`` and ``portal_credentials``
  are the only intentionally global tables.

* **Thread safety:** Flet event handlers and the background worker threads used
  for LLM / audio / browser tasks run on different threads. ``sqlite3``
  connection objects are not safe to share across threads, so instead of
  holding one long-lived connection we open a fresh connection per operation
  inside a context manager. SQLite handles the file-level locking; we enable
  WAL mode for better concurrent read/write behaviour. A module-level lock
  serialises writers as a belt-and-braces guard against ``database is locked``.

* **Foreign keys + cascade:** ``PRAGMA foreign_keys = ON`` is set on every
  connection so deleting a case removes all of its rows.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

import config

_WRITE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------
@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Yield a configured SQLite connection, committing on success."""
    config.ensure_directories()
    conn = sqlite3.connect(str(config.DB_PATH), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    court           TEXT,
    case_number     TEXT,
    judge           TEXT,
    jurisdiction    TEXT,
    charges         TEXT,
    summary         TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id         INTEGER NOT NULL,
    filename        TEXT NOT NULL,
    path            TEXT,
    doc_type        TEXT,
    content         TEXT,
    metadata_json   TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS timeline_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id         INTEGER NOT NULL,
    event_date      TEXT,
    description     TEXT NOT NULL,
    actors          TEXT,
    source_doc      TEXT,
    document_id     INTEGER,
    inconsistency   INTEGER NOT NULL DEFAULT 0,
    inconsistency_note TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS deadlines (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id         INTEGER NOT NULL,
    title           TEXT NOT NULL,
    due_date        TEXT NOT NULL,
    rule            TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    notes           TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id         INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS documents_drafts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id         INTEGER NOT NULL,
    title           TEXT NOT NULL,
    body            TEXT NOT NULL,
    export_path     TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS settings (
    key             TEXT PRIMARY KEY,
    value           TEXT
);

CREATE TABLE IF NOT EXISTS portal_credentials (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    portal_name     TEXT NOT NULL,
    url             TEXT NOT NULL,
    username        TEXT,
    password        TEXT,
    created_at      TEXT NOT NULL
);
"""


def init_db() -> None:
    """Create all tables if they do not already exist."""
    with _WRITE_LOCK, _connect() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def create_case(
    name: str,
    court: str = "",
    case_number: str = "",
    judge: str = "",
    jurisdiction: str = "",
    charges: str = "",
    summary: str = "",
) -> int:
    with _WRITE_LOCK, _connect() as conn:
        cur = conn.execute(
            """INSERT INTO cases (name, court, case_number, judge, jurisdiction,
                                  charges, summary, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (name, court, case_number, judge, jurisdiction, charges, summary, _now()),
        )
        return int(cur.lastrowid)


def list_cases() -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM cases ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_case(case_id: int) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        return dict(row) if row else None


def update_case(case_id: int, **fields: Any) -> None:
    if not fields:
        return
    allowed = {
        "name",
        "court",
        "case_number",
        "judge",
        "jurisdiction",
        "charges",
        "summary",
    }
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    assignments = ", ".join(f"{k} = ?" for k in sets)
    with _WRITE_LOCK, _connect() as conn:
        conn.execute(
            f"UPDATE cases SET {assignments} WHERE id = ?",
            (*sets.values(), case_id),
        )


def delete_case(case_id: int) -> None:
    with _WRITE_LOCK, _connect() as conn:
        conn.execute("DELETE FROM cases WHERE id = ?", (case_id,))


# ---------------------------------------------------------------------------
# Documents / evidence
# ---------------------------------------------------------------------------
def add_document(
    case_id: int,
    filename: str,
    path: str,
    doc_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    with _WRITE_LOCK, _connect() as conn:
        cur = conn.execute(
            """INSERT INTO documents (case_id, filename, path, doc_type, content,
                                      metadata_json, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                case_id,
                filename,
                path,
                doc_type,
                content,
                json.dumps(metadata or {}),
                _now(),
            ),
        )
        return int(cur.lastrowid)


def list_documents(case_id: int) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE case_id = ? ORDER BY created_at DESC",
            (case_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["metadata"] = json.loads(d.get("metadata_json") or "{}")
            except json.JSONDecodeError:
                d["metadata"] = {}
            out.append(d)
        return out


def get_document(case_id: int, document_id: int) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ? AND case_id = ?",
            (document_id, case_id),
        ).fetchone()
        return dict(row) if row else None


def delete_document(case_id: int, document_id: int) -> None:
    with _WRITE_LOCK, _connect() as conn:
        conn.execute(
            "DELETE FROM documents WHERE id = ? AND case_id = ?",
            (document_id, case_id),
        )


# ---------------------------------------------------------------------------
# Timeline events
# ---------------------------------------------------------------------------
def add_timeline_event(
    case_id: int,
    event_date: str,
    description: str,
    actors: str = "",
    source_doc: str = "",
    document_id: Optional[int] = None,
    inconsistency: bool = False,
    inconsistency_note: str = "",
) -> int:
    with _WRITE_LOCK, _connect() as conn:
        cur = conn.execute(
            """INSERT INTO timeline_events
               (case_id, event_date, description, actors, source_doc, document_id,
                inconsistency, inconsistency_note, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                case_id,
                event_date,
                description,
                actors,
                source_doc,
                document_id,
                1 if inconsistency else 0,
                inconsistency_note,
                _now(),
            ),
        )
        return int(cur.lastrowid)


def list_timeline(case_id: int) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT * FROM timeline_events WHERE case_id = ?
               ORDER BY (event_date IS NULL), event_date ASC, id ASC""",
            (case_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def flag_inconsistency(case_id: int, event_id: int, note: str) -> None:
    with _WRITE_LOCK, _connect() as conn:
        conn.execute(
            """UPDATE timeline_events SET inconsistency = 1, inconsistency_note = ?
               WHERE id = ? AND case_id = ?""",
            (note, event_id, case_id),
        )


def clear_timeline(case_id: int) -> None:
    with _WRITE_LOCK, _connect() as conn:
        conn.execute("DELETE FROM timeline_events WHERE case_id = ?", (case_id,))


# ---------------------------------------------------------------------------
# Deadlines / tickler
# ---------------------------------------------------------------------------
def add_deadline(
    case_id: int,
    title: str,
    due_date: str,
    rule: str = "",
    status: str = "pending",
    notes: str = "",
) -> int:
    with _WRITE_LOCK, _connect() as conn:
        cur = conn.execute(
            """INSERT INTO deadlines (case_id, title, due_date, rule, status, notes, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (case_id, title, due_date, rule, status, notes, _now()),
        )
        return int(cur.lastrowid)


def list_deadlines(case_id: int) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM deadlines WHERE case_id = ? ORDER BY due_date ASC",
            (case_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_deadline_status(case_id: int, deadline_id: int, status: str) -> None:
    with _WRITE_LOCK, _connect() as conn:
        conn.execute(
            "UPDATE deadlines SET status = ? WHERE id = ? AND case_id = ?",
            (status, deadline_id, case_id),
        )


def delete_deadline(case_id: int, deadline_id: int) -> None:
    with _WRITE_LOCK, _connect() as conn:
        conn.execute(
            "DELETE FROM deadlines WHERE id = ? AND case_id = ?",
            (deadline_id, case_id),
        )


# ---------------------------------------------------------------------------
# Chat messages
# ---------------------------------------------------------------------------
def add_message(case_id: int, role: str, content: str) -> int:
    with _WRITE_LOCK, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO messages (case_id, role, content, created_at) VALUES (?,?,?,?)",
            (case_id, role, content, _now()),
        )
        return int(cur.lastrowid)


def list_messages(case_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT * FROM (
                   SELECT * FROM messages WHERE case_id = ? ORDER BY id DESC LIMIT ?
               ) ORDER BY id ASC""",
            (case_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Drafted documents
# ---------------------------------------------------------------------------
def add_draft(case_id: int, title: str, body: str, export_path: str = "") -> int:
    with _WRITE_LOCK, _connect() as conn:
        cur = conn.execute(
            """INSERT INTO documents_drafts (case_id, title, body, export_path, created_at)
               VALUES (?,?,?,?,?)""",
            (case_id, title, body, export_path, _now()),
        )
        return int(cur.lastrowid)


def list_drafts(case_id: int) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents_drafts WHERE case_id = ? ORDER BY created_at DESC",
            (case_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_draft_export(case_id: int, draft_id: int, export_path: str) -> None:
    with _WRITE_LOCK, _connect() as conn:
        conn.execute(
            "UPDATE documents_drafts SET export_path = ? WHERE id = ? AND case_id = ?",
            (export_path, draft_id, case_id),
        )


# ---------------------------------------------------------------------------
# Global settings (NOT case-scoped)
# ---------------------------------------------------------------------------
def set_setting(key: str, value: str) -> None:
    with _WRITE_LOCK, _connect() as conn:
        conn.execute(
            """INSERT INTO settings (key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (key, value),
        )


def get_setting(key: str, default: str = "") -> str:
    with _connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row and row["value"] is not None else default


def all_settings() -> Dict[str, str]:
    with _connect() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
        return {r["key"]: r["value"] for r in rows}


# ---------------------------------------------------------------------------
# Portal credentials (global; used by the Court Portal automation agent)
# ---------------------------------------------------------------------------
def add_portal_credential(
    portal_name: str, url: str, username: str, password: str
) -> int:
    with _WRITE_LOCK, _connect() as conn:
        cur = conn.execute(
            """INSERT INTO portal_credentials (portal_name, url, username, password, created_at)
               VALUES (?,?,?,?,?)""",
            (portal_name, url, username, password, _now()),
        )
        return int(cur.lastrowid)


def list_portal_credentials() -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM portal_credentials ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_portal_credential(cred_id: int) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM portal_credentials WHERE id = ?", (cred_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_portal_credential(cred_id: int) -> None:
    with _WRITE_LOCK, _connect() as conn:
        conn.execute("DELETE FROM portal_credentials WHERE id = ?", (cred_id,))
