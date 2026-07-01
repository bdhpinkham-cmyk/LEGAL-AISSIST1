# Project Brief: Pro Se Guardian AI

Read this whole file before doing anything. This is the persistent context for this project — treat it as ground truth for every decision.

## What This Is

A mobile app that gives self-represented criminal defendants grounded, cited legal information from their own case documents. Built by a solo, non-professional-developer founder who is learning by reviewing your work — explain what you're doing in plain English as you go, not just code comments.

**Current status:** Personal-use build, Phase 0 (foundation). No other users yet. Long-term goal is wider release, but nothing about that changes how Phase 0-2 get built.

## Non-Negotiable Rules (apply to every phase, every file)

1. **No generation without grounding.** Any feature that answers a legal question must cite its source (document + page/section, or statute/case citation) and must refuse to answer if it doesn't have grounded support. Never let an LLM call "fill in" an answer from general knowledge when the task is case-specific.
2. **No invented deadlines.** The app never calculates or suggests a legal deadline on its own. The user supplies the jurisdictional rule; the app only does date arithmetic on top of it.
3. **Court website access:**
   - Public docket pages requiring no login → automated polling is fine.
   - Login-gated portals (PACER, county e-filing) → user logs in manually, app captures the authenticated session, automates within that session only, and prompts the user again when it expires. Never store or replay login credentials programmatically.
   - Sites with CAPTCHAs / active bot-blocking → do not automate. Surface a manual-check prompt to the user instead.
4. **Tool, not lawyer.** Copy, prompts, and UI must never phrase output as "you should do X." Only "here is what applies and why, with sources."
5. **Keep the stack boring.** Don't introduce a new library/service/pattern without asking first, even if you think it's a better fit. Fewer moving parts is the priority given the founder's experience level.

## Tech Stack (decided, don't relitigate)

- **Mobile app:** Flutter (one codebase, iOS + Android)
- **Backend:** Python + FastAPI
- **Database + Auth + File storage:** Supabase (managed Postgres, includes pgvector for embeddings, includes auth, includes storage)
- **Vector search:** Supabase's built-in pgvector — no separate vector DB
- **LLM APIs:**
  - **Gemini** — OCR, image/document understanding, embeddings generation
  - **Anthropic (Claude)** — all reasoning/writing: grounded Q&A, citation enforcement, drafting, refusal logic
- **Court data (public):** Free Law Project's Juriscraper / CourtListener API where possible, before writing any custom scraper

## Communication Style for This Founder

- Explain plans in plain English before executing (use Plan Mode / ask for confirmation on anything beyond a trivial change).
- Avoid unexplained jargon. If you use a technical term, define it in one clause the first time.
- Prefer small, testable steps over large batches of changes.
- When something breaks, ask the founder to describe *what they observed*, not to diagnose the cause themselves.

## Phase Roadmap (build in this order, don't skip ahead)

- **Phase 0 — Foundation:** repo setup, Flutter shell, Supabase project + auth, empty case list screen.
- **Phase 1 — Ingestion:** photo/file upload → Gemini OCR → chunking → embeddings stored in Supabase.
- **Phase 2 — Grounding Engine (the core):** retrieval → Claude answer generation → mandatory citation check → confidence scoring → refusal-if-ungrounded. This phase gets the most scrutiny and testing of anything in the app.
- **Phase 3 — Deadlines & Case View:** manual jurisdictional rules in, date math out; timeline/checklist UI.
- **Phase 4 — Court Automation:** public docket polling first, then login-gated session-capture pattern.
- **Phase 5 — Polish:** plain-language summaries, glossary, drafting templates.

## Right Now: Phase 0 Tasks

Do these one at a time. Stop after each and let the founder test before continuing.

1. **Initialize the repo.**
   - `git init`
   - Create a `.gitignore` appropriate for Flutter + Python (exclude build artifacts, `.env` files, API keys, Supabase local config secrets)
   - Create a `README.md` with a one-paragraph project description (personal-use legal information tool, not legal advice) and setup instructions
   - Create the GitHub repo (private) via `gh repo create pro-se-guardian-ai --private --source=. --remote=origin` — ask the founder to confirm they're logged into `gh` first (`gh auth status`); if not, walk them through `gh auth login`
   - Initial commit and push

2. **Scaffold the Flutter app.**
   - Standard Flutter project structure
   - Basic navigation: login screen → home screen (empty case list) → "New Case" screen (placeholder)
   - No backend wiring yet — just confirm it runs in a simulator

3. **Connect Supabase.**
   - Ask the founder for their Supabase project URL and anon key (they'll need to create a free project at supabase.com first if they haven't)
   - Store these as environment variables, never hardcoded, and confirm `.env` is gitignored
   - Wire up email/password auth via Supabase Auth on the login screen

4. **Cases table.**
   - Create a `cases` table in Supabase: case name, jurisdiction, charges, created date, owned by the authenticated user
   - Enable row-level security so users only ever see their own rows
   - Wire the home screen to display real cases; make "New Case" actually insert a row

Confirm each numbered task works before moving to the next. When task 4 is done and the founder can create a case and see it appear in their list, Phase 0 is complete — stop and wait for direction on Phase 1.

## Repo Note (added during Task 1 rewrite)

This repo previously held a working Python + Flet desktop/mobile app ("Pro
Se Legal Intelligence", 7 agentic engines, already on GitHub as
`bdhpinkham-cmyk/LEGAL-AISSIST1`). Rather than starting a new repo, we're
reusing this one:

- The old Flet UI and desktop packaging scripts moved to `archive/flet-app/`.
- The old Python logic (`agents.py`, `ingestion.py`, `llm_router.py`,
  `gemini_client.py`, `rag.py`, `config.py`, `database.py`, `automation.py`)
  stayed at the repo root — it's a candidate for reuse inside `backend/`
  once that gets built in Phase 2, since the backend stack is Python either way.
- `mobile/` and `backend/` were added as the homes for the new work.
- The GitHub repo itself kept its existing name/URL; renaming it is a
  cosmetic change that can happen anytime in GitHub settings.
