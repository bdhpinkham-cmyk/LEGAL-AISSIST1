# Pro Se Guardian AI

A personal-use tool that gives a self-represented (pro se) criminal defendant
grounded, cited legal information pulled from their own case documents. It is
a **legal information tool, not legal advice**, and using it does not create
an attorney-client relationship. It never invents deadlines and never phrases
output as "you should do X" — only "here is what applies and why, with
sources."

## Status

**Phase 0 (Foundation) — in progress.** This repo previously held a Python +
Flet desktop/mobile app ("Pro Se Legal Intelligence"). It's being rewritten
as a Flutter mobile app with a Supabase backend, per the current project
plan (see `CLAUDE.md`). The old app still runs and is kept at
`archive/flet-app/` for reference; it is no longer the active version.

## What's in this repo

- `mobile/` — the new Flutter app (iOS + Android). Current focus of Phase 0.
- `backend/` — the new FastAPI backend (built out starting Phase 2). Empty for now.
- `agents.py`, `ingestion.py`, `llm_router.py`, `gemini_client.py`, `rag.py`,
  `config.py`, `database.py`, `automation.py` — logic from the earlier
  desktop app, left at the repo root. Some of this (model routing, document
  ingestion, the agentic engines) may be reused inside `backend/` once that
  gets built — nothing here has been deleted, just not yet re-integrated.
- `archive/flet-app/` — the old Flet desktop UI and packaging scripts
  (`ui.py`, launcher scripts, PyInstaller spec). Kept so the previous working
  version isn't lost.
- `docs/` — design notes.

## Tech stack (current plan)

- **Mobile app:** Flutter (iOS + Android, one codebase)
- **Backend:** Python + FastAPI
- **Database / Auth / File storage:** Supabase (managed Postgres, pgvector for embeddings, built-in auth and storage)
- **LLMs:** Gemini (OCR, document understanding, embeddings) and Anthropic Claude (reasoning, drafting, citation enforcement, refusal logic)
- **Public court data:** Free Law Project's Juriscraper / CourtListener API, preferred over custom scrapers

## Setup

Setup instructions will be filled in as each Phase 0 task lands:

1. Flutter app setup — coming in Phase 0, Task 2 (Flutter scaffold)
2. Supabase project connection — coming in Phase 0, Task 3
3. Environment variables — this project uses a `.env` file for secrets
   (Supabase URL/anon key, API keys). `.env` is git-ignored; never commit it.

## Disclaimer

This software provides legal information and drafting assistance grounded in
the user's own case documents — it is not legal advice, and it does not
create an attorney-client relationship. Always verify any computed deadline
or drafted document against the controlling court rules and statutes before
relying on it or filing anything.
