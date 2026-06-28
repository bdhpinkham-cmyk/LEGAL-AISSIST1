# Pro Se Legal Intelligence

A production-ready, **fully local** legal intelligence application for a
self-represented (pro se) defendant, built with **Flet (Python)** so the same
codebase runs as a native **Windows `.exe`** and an **Android `.apk`**. It
replicates the agentic workflows of Thomson Reuters CoCounsel, scaled for a
local desktop/mobile environment using open APIs and cloud LLM routing.

All case data lives in a **local SQLite** database and a **local vector store** —
no client data is sent to any server except the cloud LLM / tool APIs you
explicitly configure.

---

## The 7 Core Agentic Engines

| # | Engine | Where |
|---|--------|-------|
| 1 | **Discovery & Timeline** — ingest PDFs / images / audio-video, extract facts to JSON, build an interactive timeline with an **Inconsistency Matrix** | `agents.py` · `ingestion.py` |
| 2 | **Deep Research & Judge Analytics** — iterative CourtListener loop with a hard step limit | `agents.py` |
| 3 | **OSINT Context (`@web`)** — Tavily/DuckDuckGo lookups with cited URLs, triggered from chat | `agents.py` |
| 4 | **Brief Builder & Defense Drafter** — synthesises timeline + research, auto-redacts PII, exports court-ready pleading paper (1-28 line numbers + caption) | `agents.py` · `ingestion.py` |
| 5 | **Procedural Guardrails** — statutory deadline tickler + cross-examination prep from timeline contradictions | `agents.py` |
| 6 | **Last-Mile Pro Se Toolkit** — Intake Interviewer, EXIF/GPS Metadata Extractor, Legalese Decoder | `agents.py` · `ingestion.py` |
| 7 | **Autonomous Court Portal Agent** — Playwright browser automation with a strict **CAPTCHA guardrail** | `automation.py` |

---

## Pre-Flight Report (bugs caught and fixed during pre-debugging)

These issues were traced and fixed *before* shipping the code:

1. **Cross-case data leakage.** An early design had a generic `get_document(id)`
   accessor. That would let one case read another's evidence. **Fix:** every
   case-data query takes and filters by `case_id` (`get_document(case_id, id)`,
   `list_timeline(case_id)`, etc.); the RAG store filters by `case_id` on every
   query. Verified by an automated isolation test.

2. **SQLite "database is locked" under threading.** Flet event handlers and the
   LLM/audio/browser worker threads run on different threads; a shared
   connection would crash. **Fix:** a fresh connection per operation inside a
   context manager, WAL journal mode, `PRAGMA foreign_keys=ON`, and a
   module-level write lock serialising writers.

3. **Frozen UI during heavy work.** Synchronous LLM, transcription, research and
   browser calls would block Flet's event loop. **Fix:** `AppUI.run_bg()` runs
   every slow operation on a daemon thread and streams progress back via
   `page.update()`; the UI never blocks.

4. **Runaway iterative agents.** The research agent could loop forever. **Fix:**
   `MAX_RESEARCH_ITERATIONS` hard cap, plus an explicit model "done?" decision
   each loop and a `MAX_AGENT_TOOL_STEPS` guard.

5. **Rate limits / transient API failures crashing the app.** **Fix:** every
   network call is wrapped in retry-with-exponential-backoff
   (`_with_retries`, `MAX_API_RETRIES`), classifying 429/5xx/timeout/connection
   errors as retryable and surfacing clear `LLMError`s otherwise.

6. **Model returning JSON wrapped in markdown / prose.** Naive `json.loads`
   would throw. **Fix:** `_extract_json` strips ``` fences and locates the first
   balanced JSON object/array. Verified by test.

7. **CAPTCHA-bypass risk in the portal agent.** **Fix:** the agent always
   launches a *visible* browser, detects bot-protection screens
   (`_is_blocked`), and on detection **pauses**, alerts the user
   ("Manual intervention required"), and blocks on a `threading.Event` until the
   user solves it and clicks **Resume** — it never attempts a bypass.

8. **Flet API version drift (`ft.icons`/`ft.Icons`, `ft.colors`/`ft.Colors`).**
   **Fix:** icons are passed as plain name strings and colours as hex strings,
   so the UI builds across Flet versions.

9. **Pleading line numbers breaking on reflow.** A hand-drawn number column
   alone would desync when text wraps. **Fix:** the DOCX export *also* enables
   Word's native section line-numbering (restart each page) in the section XML,
   so numbering stays correct in the real document.

10. **Heavy SDKs blocking import on minimal installs.** **Fix:** provider SDKs
    (`anthropic`/`openai`/`google-generativeai`) and parsers (`pypdf`/`Pillow`/
    `python-docx`/`playwright`) are imported lazily with clear "package not
    installed" messages, so the app runs with only the providers you use.

---

## Project layout

```
config.py        Paths, provider catalogue, limits (no third-party deps)
database.py      Local SQLite — case-isolated schema + CRUD
llm_router.py    Anthropic / OpenAI / Gemini routing with retry + JSON extraction
rag.py           Local case-isolated TF-IDF vector store (sqlite-backed)
ingestion.py     PDF/text/EXIF/audio extraction, PII redaction, DOCX pleading export
agents.py        The 7 agentic engines (except browser automation)
automation.py    Autonomous Court Portal Agent (Playwright) + CAPTCHA guardrail
ui.py            Flet dark-mode UI; all heavy work on background threads
main.py          Entry point
requirements.txt Exact dependencies
```

---

## Quick start (run locally)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the browser used by the Court Portal agent (one time)
playwright install chromium

# 4. Run the app
python main.py
```

On first launch, open **Settings**, choose a provider (default: Anthropic), and
paste your API key. Optionally add Tavily (web search) and Deepgram
(transcription) keys.

---

## Packaging Guide (beginner-friendly)

### Windows `.exe`

Flet bundles PyInstaller. From the project folder, with your venv active:

```bash
pip install -r requirements.txt
flet pack main.py --name "ProSeLegal" --product-name "Pro Se Legal Intelligence"
```

The standalone executable appears in **`dist/ProSeLegal.exe`**. Double-click to
run — no Python needed on the target machine.

> The browser automation feature needs Chromium present. For a self-contained
> build, run `playwright install chromium` on the build machine; for end users,
> have them run `playwright install chromium` once, or ship Chromium alongside.

If you prefer raw PyInstaller:

```bash
pip install pyinstaller
pyinstaller --noconfirm --windowed --name ProSeLegal main.py
```

### Android `.apk`

Flet builds Android packages with its `build` command (requires the Flutter SDK,
which `flet build` will guide you to install on first run):

```bash
pip install -r requirements.txt
flet build apk --project "ProSeLegal" --org com.prose.legal
```

The APK appears in **`build/apk/`**. Install on a device with:

```bash
adb install build/apk/app-release.apk
```

> **Mobile note:** Playwright browser automation (Engine 7) is a desktop
> capability and is not available inside the Android sandbox; all six other
> engines run natively on Android. The app degrades gracefully — the Court
> Portal tab reports that Playwright/Chromium is unavailable on mobile.

---

## Data location

All data is stored locally under a per-OS app directory (see
`config._platform_data_root()`): the SQLite workspace DB, the local vector
store, downloaded evidence, exported pleadings, and logs. Deleting a case
cascades to all of its evidence, timeline, chat, and vector chunks.

---

## Disclaimer

This software provides **legal information and drafting assistance, not legal
advice**, and does not create an attorney-client relationship. Always verify
computed deadlines and any drafted document against the controlling court rules
and statutes before filing.
