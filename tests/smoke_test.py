"""
tests/smoke_test.py
===================
Dependency-light smoke test for the Pro Se Legal Intelligence core.

It exercises the logic that does NOT require the GUI runtime or any cloud SDK:
case isolation, the local RAG store, PII redaction, JSON extraction, timeline
ordering, tier routing, and the Gemini backend detection. Run by CI on every
push / PR. Exits non-zero on the first failed assertion.

Only standard library + ``requests`` is required (``requests`` is imported at
the top of ``ingestion``); no cloud SDKs are installed in CI.
"""

import os
import sys
import tempfile

# Point all app data at a throwaway dir BEFORE importing config (paths are
# resolved at import time).
os.environ["FLET_APP_STORAGE_DATA"] = tempfile.mkdtemp(prefix="proselegal-ci-")

# Make the project root importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import database as db  # noqa: E402
import gemini_client  # noqa: E402
from ingestion import redact_pii  # noqa: E402
from llm_router import LLMRouter, _extract_json  # noqa: E402
from rag import RAGStore  # noqa: E402


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(f"FAILED: {label}")
    print(f"  ok: {label}")


def main() -> None:
    print("Pro Se Legal Intelligence — smoke test")
    db.init_db()

    # --- Case isolation -----------------------------------------------------
    c1 = db.create_case("Case A", charges="theft")
    c2 = db.create_case("Case B", charges="trespass")
    d1 = db.add_document(c1, "a.txt", "/x", "document", "Alpha: a red car on Main St")
    db.add_document(c2, "b.txt", "/y", "document", "Beta: a blue truck downtown")
    check("each case sees only its own documents",
          len(db.list_documents(c1)) == 1 and len(db.list_documents(c2)) == 1)
    check("cross-case document read is blocked",
          db.get_document(c1, db.list_documents(c2)[0]["id"]) is None)

    # --- RAG retrieval + isolation -----------------------------------------
    store = RAGStore()
    store.add_document(c1, d1, "a.txt", "The defendant drove a red Honda near the bank on March 3rd.")
    store.add_document(c2, 999, "b.txt", "The witness saw a blue truck speeding downtown at night.")
    hits = store.query(c1, "what car did the defendant drive")
    check("RAG returns a relevant chunk", bool(hits) and "red" in hits[0]["text"].lower())
    check("RAG never leaks across cases",
          all("Honda" not in h["text"] for h in store.query(c2, "red Honda")))

    # --- PII redaction ------------------------------------------------------
    red, notes = redact_pii(
        "SSN 123-45-6789, call 555-123-4567, email a@b.com, minor Johnny", minor_names=["Johnny"]
    )
    check("redaction scrubs SSN/phone/email/minor",
          all(tok in red for tok in ("[REDACTED-SSN]", "[REDACTED-PHONE]",
                                      "[REDACTED-EMAIL]", "[REDACTED-MINOR]")))
    check("redaction reports notes", len(notes) >= 4)

    # --- JSON extraction robustness ----------------------------------------
    check("JSON in markdown fence parses", _extract_json('```json\n{"a":1}\n```') == {"a": 1})
    check("JSON embedded in prose parses", _extract_json("here: [1,2,3] done") == [1, 2, 3])

    # --- Timeline ordering + inconsistency flag ----------------------------
    db.add_timeline_event(c1, "2023-01-05", "later event")
    db.add_timeline_event(c1, "2023-01-01", "earlier event")
    tl = db.list_timeline(c1)
    check("timeline sorts by date", tl[0]["event_date"] == "2023-01-01")
    db.flag_inconsistency(c1, tl[0]["id"], "conflicts with b.txt")
    check("inconsistency flag persists", db.list_timeline(c1)[0]["inconsistency"] == 1)

    # --- Drafts + last-case persistence ------------------------------------
    db.add_draft(c1, "Motion to Suppress", "BODY", "/tmp/out.docx")
    check("draft saved with export path", db.list_drafts(c1)[0]["export_path"] == "/tmp/out.docx")
    db.set_setting("last_case_id", str(c1))
    check("last-case setting persists", db.get_setting("last_case_id") == str(c1))

    # --- Tier routing -------------------------------------------------------
    r = LLMRouter()
    check("extraction tier defaults to Gemini Flash", r.resolve("extraction")[:2] == ("gemini", "gemini-3.5-flash"))
    check("medium tier defaults to Sonnet 4.6", r.resolve("medium")[:2] == ("anthropic", "claude-sonnet-4-6"))
    check("heavy tier defaults to Opus 4.8", r.resolve("heavy")[:2] == ("anthropic", "claude-opus-4-8"))

    # --- Gemini backend detection ------------------------------------------
    check("gemini unconfigured by default", gemini_client.is_configured() is False)
    db.set_setting(config.SETTING_GEMINI_KEY, "fake-key")
    check("AI Studio: configured with an API key", gemini_client.is_configured() is True)
    db.set_setting(config.SETTING_GEMINI_KEY, "")
    db.set_setting(config.SETTING_GEMINI_BACKEND, "vertex")
    check("Vertex: needs a project", gemini_client.is_configured() is False)
    db.set_setting(config.SETTING_VERTEX_PROJECT, "my-proj")
    check("Vertex: configured with project (no key)", gemini_client.is_configured() is True)

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
