"""
agents.py
=========
The agentic engines that give the application its CoCounsel-style capabilities.

Each engine is a small, testable class that takes an :class:`LLMRouter` and
operates against the local database / RAG store. None of them touch the UI
directly — they accept an optional ``log`` callback so the caller (a background
worker thread in ui.py) can stream progress without the engines knowing about
Flet. This keeps the UI responsive: the heavy work runs off the event loop.

Engines implemented here
------------------------
1. DiscoveryTimelineEngine  — ingest evidence → facts JSON → timeline + inconsistencies
2. DeepResearchAgent        — iterative CourtListener loop with a hard step limit
3. OSINTAgent               — @web context lookup (Tavily / DuckDuckGo) with citations
4. BriefBuilder             — synthesise timeline + research into a drafted motion
5. ProceduralEngine         — statutory deadline calculator + cross-exam prep
6. IntakeInterviewer        — guided onboarding question flow
7. LegaleseDecoder          — plain-language (4th-grade) translation

The Court Portal browser-automation agent lives in ``automation.py``.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

import config
import database as db
from ingestion import (
    IngestionError,
    extract_document,
    extract_image_metadata,
    extract_text,
    redact_pii,
    transcribe_audio,
)
from llm_router import LLMError, LLMRouter
from rag import STORE

LogFn = Callable[[str], None]


def _noop(_: str) -> None:  # default log sink
    pass


# ===========================================================================
# 1. Discovery & Timeline Engine
# ===========================================================================
class DiscoveryTimelineEngine:
    """Ingest evidence and extract a strictly-typed timeline of facts."""

    def __init__(self, router: LLMRouter) -> None:
        self.router = router

    def ingest_file(
        self,
        case_id: int,
        path: str,
        filename: str,
        log: LogFn = _noop,
    ) -> Dict[str, Any]:
        """Ingest one evidence file end-to-end.

        Returns a summary dict: {document_id, char_count, metadata, facts_added,
        doc_type, gaps, units_expected, units_processed, case_fields_applied}.
        ``gaps`` lists any page range / audio segment that could not be
        extracted even after retries — nothing is dropped silently, it is
        surfaced here instead.
        """
        lower = filename.lower()
        metadata: Dict[str, str] = {}
        doc_type = "document"
        content = ""
        gaps: List[Dict[str, Any]] = []
        units_expected: Optional[int] = None
        units_processed: Optional[int] = None

        # The extraction tier (default Gemini 3.5 Flash) handles multimodal OCR.
        extraction = self.router.resolve(config.TIER_EXTRACTION)

        if lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif", ".tiff")):
            doc_type = "image"
            log("Extracting image metadata (EXIF / GPS)…")
            metadata = extract_image_metadata(path)
            log("Reading image (OCR + visual description)…")
            visual, gaps = extract_document(path, filename, extraction, log=log)
            content = self._describe_image_metadata(filename, metadata) + "\n\n" + visual
        elif lower.endswith(
            (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".mp4", ".mov", ".m4v", ".webm")
        ):
            doc_type = "media"
            log("Transcribing audio/video (this can take a while)…")
            content, gaps = transcribe_audio(
                path,
                deepgram_key=db.get_setting(config.SETTING_DEEPGRAM_KEY),
                openai_key=db.get_setting(config.SETTING_OPENAI_KEY),
                log=log,
            )
        else:
            doc_type = "pdf" if lower.endswith(".pdf") else "document"
            log("Parsing document (multimodal OCR for scanned/image pages)…")
            content, gaps = extract_document(path, filename, extraction, log=log)

        if not content.strip():
            raise IngestionError(
                f"No text could be extracted from {filename}. "
                "If it is a scanned PDF it may need OCR."
            )

        if gaps:
            log(f"Warning: {len(gaps)} gap(s) in extracted content for {filename}.")

        if doc_type == "pdf":
            try:
                from pypdf import PdfReader

                units_expected = len(PdfReader(path).pages)
                missing = sum(
                    (g["end_page"] - g["start_page"] + 1)
                    for g in gaps
                    if "start_page" in g and "end_page" in g
                )
                units_processed = max(0, units_expected - missing)
            except Exception:  # noqa: BLE001 — page count is informational only
                units_expected = units_processed = None

        document_id = db.add_document(
            case_id, filename, path, doc_type, content, metadata
        )

        log("Indexing into local RAG store…")
        STORE.add_document(case_id, document_id, filename, content)

        log("Extracting facts with the LLM…")
        facts = self._extract_facts(content, filename, log=log)
        for fact in facts:
            db.add_timeline_event(
                case_id,
                event_date=fact.get("date", ""),
                description=fact.get("fact", ""),
                actors=", ".join(fact.get("actors", []))
                if isinstance(fact.get("actors"), list)
                else str(fact.get("actors", "")),
                source_doc=filename,
                document_id=document_id,
            )

        log("Checking for case-identifying fields (court, case number, judge…)…")
        proposed_fields = self._extract_case_fields(content, filename, log=log)
        case_fields_applied = self._merge_case_fields(case_id, proposed_fields, log=log)

        return {
            "document_id": document_id,
            "char_count": len(content),
            "metadata": metadata,
            "facts_added": len(facts),
            "doc_type": doc_type,
            "gaps": gaps,
            "units_expected": units_expected,
            "units_processed": units_processed,
            "case_fields_applied": case_fields_applied,
        }

    def _describe_image_metadata(self, filename: str, metadata: Dict[str, str]) -> str:
        lines = [f"Image evidence: {filename}"]
        for k, v in metadata.items():
            lines.append(f"{k}: {v}")
        if "DateTimeOriginal" in metadata or "DateTime" in metadata:
            lines.append(
                "Authentication note: embedded capture timestamp present — "
                "usable to corroborate the timeline."
            )
        if "GPSLatitude" in metadata:
            lines.append(
                f"Geolocation present: {metadata.get('GPSLatitude')}, "
                f"{metadata.get('GPSLongitude')}"
            )
        return "\n".join(lines)

    def _extract_facts(self, content: str, source: str, log: LogFn = _noop) -> List[Dict[str, Any]]:
        """Sweep the ENTIRE document in windows so no facts are dropped on long files."""
        window = config.FACT_WINDOW_CHARS
        windows = [content[i : i + window] for i in range(0, len(content), window)]
        windows = windows[: config.FACT_MAX_WINDOWS]
        cleaned: List[Dict[str, Any]] = []
        seen: set = set()
        for idx, excerpt in enumerate(windows, 1):
            if len(windows) > 1:
                log(f"Extracting facts (section {idx}/{len(windows)})…")
            prompt = (
                "Extract every discrete, objective fact from the following evidence "
                "excerpt. Return a JSON array. Each element MUST have keys: "
                '"date" (ISO YYYY-MM-DD if a date is present, else ""), '
                '"fact" (a single factual statement), '
                '"actors" (array of people/entities involved). '
                "Do not invent facts. Only include what the text supports.\n\n"
                f"SOURCE: {source}\n\nEVIDENCE EXCERPT:\n{excerpt}"
            )
            try:
                data = self.router.complete_json(
                    prompt, tier=config.TIER_MEDIUM, max_tokens=4000
                )
            except LLMError:
                continue
            if isinstance(data, dict) and "facts" in data:
                data = data["facts"]
            if not isinstance(data, list):
                continue
            for item in data:
                if isinstance(item, dict) and item.get("fact"):
                    key = (item.get("date", ""), item.get("fact", "").strip().lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    cleaned.append(item)
        return cleaned

    def _extract_case_fields(self, content: str, source: str, log: LogFn = _noop) -> Dict[str, str]:
        """Sweep the document for case-identifying fields (court, case number, judge, …).

        Mirrors ``_extract_facts``'s windowed sweep so nothing past the first
        excerpt is missed. Never invents a value — only returns fields the
        model found explicitly stated in the text.
        """
        window = config.FACT_WINDOW_CHARS
        windows = [content[i : i + window] for i in range(0, len(content), window)]
        windows = windows[: config.FACT_MAX_WINDOWS]
        keys = config.CASE_FIELD_KEYS
        found: Dict[str, str] = {}
        for idx, excerpt in enumerate(windows, 1):
            if len(found) == len(keys):
                break
            if len(windows) > 1:
                log(f"Scanning for case info (section {idx}/{len(windows)})…")
            prompt = (
                "Identify any of the following case-identifying fields that are "
                f"explicitly stated in this evidence excerpt: {', '.join(keys)}. "
                "Return a JSON object using only the keys you are confident are "
                "present, each mapped to its exact value as written. Do not guess "
                "or invent a value; omit any key you cannot find verbatim.\n\n"
                f"SOURCE: {source}\n\nEVIDENCE EXCERPT:\n{excerpt}"
            )
            try:
                data = self.router.complete_json(prompt, tier=config.TIER_MEDIUM, max_tokens=500)
            except LLMError:
                continue
            if not isinstance(data, dict):
                continue
            for key in keys:
                value = data.get(key)
                if isinstance(value, str) and value.strip() and key not in found:
                    found[key] = value.strip()
        return found

    def _merge_case_fields(
        self, case_id: int, proposed: Dict[str, str], log: LogFn = _noop
    ) -> Dict[str, str]:
        """Apply ``proposed`` fields, but only into currently-empty case fields.

        A later document that disagrees with an already-filled field is logged
        as a conflict, never applied — this avoids a possibly-wrong OCR guess
        overwriting a value the user entered or a prior extraction confirmed.
        """
        if not proposed:
            return {}
        case = db.get_case(case_id) or {}
        to_write: Dict[str, str] = {}
        for key, value in proposed.items():
            if key not in config.CASE_FIELD_KEYS:
                continue
            current = str(case.get(key, "") or "").strip()
            if current:
                if current.lower() != value.strip().lower():
                    log(f"Field conflict: {key} already set to '{current}'; keeping existing.")
                continue
            to_write[key] = value
        if to_write:
            db.update_case(case_id, **to_write)
            log(f"Case info updated: {', '.join(to_write.keys())}.")
        return to_write

    def _should_skip_file(self, filename: str) -> bool:
        ext = os.path.splitext(filename.lower())[1]
        return ext in config.FOLDER_SKIP_EXTENSIONS

    def ingest_folder(
        self,
        case_id: int,
        folder_path: str,
        log: LogFn = _noop,
        on_field_update: Optional[Callable[[Dict[str, str]], None]] = None,
    ) -> Dict[str, Any]:
        """Walk a folder (recursively) and ingest every supported file.

        Reuses ``ingest_file`` for every file — no duplicated dispatch logic.
        A failure on one file is recorded and the run continues; it never
        aborts the whole folder. Returns a coverage-report dict.
        """
        discovered: List[Tuple[str, str]] = []
        for root, dirs, files in os.walk(folder_path):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
            for name in files:
                if name.startswith("."):
                    continue
                discovered.append((os.path.join(root, name), name))
        discovered.sort(key=lambda t: t[0])

        skipped: List[Dict[str, str]] = []
        to_process: List[Tuple[str, str]] = []
        for path, name in discovered:
            if self._should_skip_file(name):
                log(f"Skipping {name} (unsupported file type).")
                skipped.append({"filename": name, "reason": "unsupported file type"})
            else:
                to_process.append((path, name))

        if len(to_process) > config.FOLDER_MAX_FILES:
            log(
                f"Folder has {len(to_process)} eligible files; processing the first "
                f"{config.FOLDER_MAX_FILES} (FOLDER_MAX_FILES cap)."
            )
            for path, name in to_process[config.FOLDER_MAX_FILES:]:
                skipped.append({"filename": name, "reason": "exceeded FOLDER_MAX_FILES cap"})
            to_process = to_process[: config.FOLDER_MAX_FILES]

        files_succeeded = 0
        failed: List[Dict[str, str]] = []
        all_gaps: List[Dict[str, Any]] = []
        case_fields_applied: Dict[str, str] = {}
        pages_expected = 0
        pages_processed = 0
        have_page_counts = False

        log(f"Ingesting {len(to_process)} file(s) from {folder_path}…")
        for path, name in to_process:
            log(f"Ingesting {name}…")
            try:
                summary = self.ingest_file(case_id, path, name, log=log)
            except (IngestionError, LLMError) as exc:
                log(f"{name}: {exc}")
                failed.append({"filename": name, "reason": str(exc)})
                continue
            except Exception as exc:  # noqa: BLE001 — one bad file must not abort the run
                log(f"{name}: unexpected error: {exc}")
                failed.append({"filename": name, "reason": str(exc)})
                continue

            files_succeeded += 1
            all_gaps.extend({**g, "filename": name} for g in summary.get("gaps") or [])
            if summary.get("units_expected") is not None:
                have_page_counts = True
                pages_expected += summary["units_expected"]
                pages_processed += summary.get("units_processed") or 0
            applied = summary.get("case_fields_applied") or {}
            if applied:
                case_fields_applied.update(applied)
                if on_field_update:
                    on_field_update(applied)

        log(
            f"Folder ingestion complete: {files_succeeded}/{len(to_process)} files, "
            f"{len(skipped)} skipped, {len(failed)} failed."
        )
        return {
            "files_total": len(to_process),
            "files_succeeded": files_succeeded,
            "files_failed": failed,
            "files_skipped": skipped,
            "pages_expected": pages_expected if have_page_counts else None,
            "pages_processed": pages_processed if have_page_counts else None,
            "gaps": all_gaps,
            "case_fields_applied": case_fields_applied,
        }

    def analyze_inconsistencies(self, case_id: int, log: LogFn = _noop) -> int:
        """Compare all timeline facts and flag contradictions. Returns flag count."""
        events = db.list_timeline(case_id)
        if len(events) < 2:
            return 0
        log("Building the Inconsistency Matrix…")
        compact = [
            {
                "id": e["id"],
                "date": e["event_date"],
                "fact": e["description"],
                "source": e["source_doc"],
            }
            for e in events
        ]
        prompt = (
            "You are auditing a fact timeline for contradictions across sources "
            "(conflicting dates, mutually exclusive claims, impossible sequences). "
            "Return a JSON array of contradictions. Each element MUST have: "
            '"event_id" (the id of the contradicted event), and '
            '"note" (a one-sentence explanation of the conflict, naming the other source). '
            "If there are no contradictions return an empty array.\n\n"
            f"TIMELINE:\n{json.dumps(compact, indent=2)}"
        )
        try:
            data = self.router.complete_json(prompt, max_tokens=3000)
        except LLMError:
            return 0
        if isinstance(data, dict) and "contradictions" in data:
            data = data["contradictions"]
        if not isinstance(data, list):
            return 0
        flagged = 0
        for item in data:
            if isinstance(item, dict) and item.get("event_id") is not None:
                try:
                    db.flag_inconsistency(
                        case_id, int(item["event_id"]), str(item.get("note", ""))
                    )
                    flagged += 1
                except (ValueError, TypeError):
                    continue
        log(f"Flagged {flagged} potential inconsistencies.")
        return flagged


# ===========================================================================
# 2. Deep Research & Judge Analytics Agent (iterative CourtListener loop)
# ===========================================================================
class DeepResearchAgent:
    """Iterative legal-research agent over the free CourtListener / RECAP API."""

    SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"

    def __init__(self, router: LLMRouter) -> None:
        self.router = router

    def _search(self, query: str) -> List[Dict[str, Any]]:
        last_exc: Optional[Exception] = None
        for attempt in range(config.MAX_API_RETRIES):
            try:
                resp = requests.get(
                    self.SEARCH_URL,
                    params={"q": query, "order_by": "score desc"},
                    timeout=config.HTTP_TIMEOUT_SECONDS,
                    headers={"User-Agent": f"{config.APP_NAME}/{config.APP_VERSION}"},
                )
                if resp.status_code == 429:
                    raise RuntimeError("rate limit")
                resp.raise_for_status()
                return resp.json().get("results", [])[:8]
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < config.MAX_API_RETRIES - 1:
                    time.sleep(config.RETRY_BASE_DELAY * (2 ** attempt))
        raise LLMError(f"CourtListener search failed: {last_exc}")

    def run(
        self,
        question: str,
        judge: str = "",
        log: LogFn = _noop,
    ) -> Dict[str, Any]:
        """Run a bounded research loop. Returns {answer, citations, transcript}."""
        transcript: List[str] = []
        collected: List[Dict[str, Any]] = []

        # Seed the first query from the user's question.
        query = question
        if judge:
            log(f"Including judge analytics for: {judge}")

        for iteration in range(1, config.MAX_RESEARCH_ITERATIONS + 1):
            log(f"Research iteration {iteration}/{config.MAX_RESEARCH_ITERATIONS}: '{query}'")
            transcript.append(f"Query {iteration}: {query}")
            try:
                results = self._search(query)
            except LLMError as exc:
                transcript.append(str(exc))
                break

            for r in results:
                collected.append(
                    {
                        "caption": r.get("caseName") or r.get("caseNameShort") or "",
                        "court": r.get("court") or "",
                        "date": r.get("dateFiled") or "",
                        "url": "https://www.courtlistener.com" + (r.get("absolute_url") or ""),
                        "snippet": (r.get("snippet") or "")[:400],
                    }
                )

            # Ask the model whether it has enough, and if not, to refine the query.
            decision = self._decide_next(question, judge, collected, iteration)
            transcript.append(f"Decision: {decision.get('reasoning', '')}")
            if decision.get("done") or iteration == config.MAX_RESEARCH_ITERATIONS:
                break
            next_q = decision.get("next_query", "").strip()
            if not next_q or next_q == query:
                break
            query = next_q

        log("Synthesising final research memo…")
        answer = self._synthesize(question, judge, collected)
        # De-duplicate citations by URL.
        seen = set()
        citations = []
        for c in collected:
            if c["url"] and c["url"] not in seen:
                seen.add(c["url"])
                citations.append(c)
        return {"answer": answer, "citations": citations, "transcript": transcript}

    def _decide_next(
        self,
        question: str,
        judge: str,
        collected: List[Dict[str, Any]],
        iteration: int,
    ) -> Dict[str, Any]:
        compact = [
            {"caption": c["caption"], "court": c["court"], "snippet": c["snippet"]}
            for c in collected[-12:]
        ]
        prompt = (
            "You are a legal research agent deciding whether you have enough "
            "precedent to answer the question, or whether to refine your search. "
            f"This is iteration {iteration} of {config.MAX_RESEARCH_ITERATIONS} (hard limit).\n\n"
            f"QUESTION: {question}\n"
            f"ASSIGNED JUDGE (analyse history if relevant): {judge or 'N/A'}\n\n"
            f"RESULTS SO FAR:\n{json.dumps(compact, indent=2)}\n\n"
            'Return JSON: {"done": bool, "reasoning": "...", "next_query": "a refined '
            'search string if not done"}.'
        )
        try:
            data = self.router.complete_json(prompt, max_tokens=1200)
            if isinstance(data, dict):
                return data
        except LLMError:
            pass
        return {"done": True, "reasoning": "Could not refine; stopping.", "next_query": ""}

    def _synthesize(
        self, question: str, judge: str, collected: List[Dict[str, Any]]
    ) -> str:
        compact = [
            {
                "caption": c["caption"],
                "court": c["court"],
                "date": c["date"],
                "snippet": c["snippet"],
                "url": c["url"],
            }
            for c in collected[:20]
        ]
        judge_clause = (
            f"\nAlso provide a short 'Judge Analytics' section on Judge {judge}: "
            "note any patterns visible in the retrieved cases relevant to this issue."
            if judge
            else ""
        )
        prompt = (
            "Write a concise research memo answering the question using ONLY the "
            "retrieved authorities below. Cite case captions inline. Be candid "
            "about gaps. Do not fabricate citations." + judge_clause + "\n\n"
            f"QUESTION: {question}\n\nAUTHORITIES:\n{json.dumps(compact, indent=2)}"
        )
        try:
            return self.router.complete(prompt, max_tokens=4000)
        except LLMError as exc:
            return f"(Synthesis failed: {exc})"


# ===========================================================================
# 3. OSINT Context Agent (@web)
# ===========================================================================
class OSINTAgent:
    """Non-legal context lookups via Tavily (preferred) or DuckDuckGo."""

    def __init__(self, router: LLMRouter) -> None:
        self.router = router

    def run(self, query: str, log: LogFn = _noop) -> Dict[str, Any]:
        tavily = db.get_setting(config.SETTING_TAVILY_KEY)
        if tavily:
            log("Searching the web via Tavily…")
            results = self._tavily(query, tavily)
        else:
            log("Searching the web via DuckDuckGo…")
            results = self._duckduckgo(query)

        if not results:
            return {"answer": "No web results were found.", "citations": []}

        log("Summarising findings with citations…")
        compact = [{"title": r["title"], "snippet": r["snippet"], "url": r["url"]} for r in results]
        prompt = (
            "Summarise what these web results say about the query. Cite each claim "
            "with the source URL in parentheses. Be factual and note uncertainty.\n\n"
            f"QUERY: {query}\n\nRESULTS:\n{json.dumps(compact, indent=2)}"
        )
        try:
            answer = self.router.complete(prompt, max_tokens=2500)
        except LLMError as exc:
            answer = f"(Summary failed: {exc})"
        return {"answer": answer, "citations": results}

    def _tavily(self, query: str, key: str) -> List[Dict[str, str]]:
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": key, "query": query, "max_results": 6},
                timeout=config.HTTP_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("content", "")[:400],
                    "url": r.get("url", ""),
                }
                for r in resp.json().get("results", [])
            ]
        except Exception:  # noqa: BLE001
            return []

    def _duckduckgo(self, query: str) -> List[Dict[str, str]]:
        try:
            resp = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (compatible; ProSeLegal/1.0)"},
                timeout=config.HTTP_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            html = resp.text
            results: List[Dict[str, str]] = []
            # Parse result blocks with a tolerant regex (no bs4 dependency).
            pattern = re.compile(
                r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
                r'(?:class="result__snippet"[^>]*>(.*?)</a>)?',
                re.DOTALL,
            )
            for m in pattern.finditer(html):
                url = self._clean_ddg_url(m.group(1))
                title = _strip_tags(m.group(2) or "")
                snippet = _strip_tags(m.group(3) or "")
                if url and title:
                    results.append({"title": title, "snippet": snippet[:400], "url": url})
                if len(results) >= 6:
                    break
            return results
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _clean_ddg_url(href: str) -> str:
        m = re.search(r"uddg=([^&]+)", href)
        if m:
            from urllib.parse import unquote

            return unquote(m.group(1))
        return href


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


# ===========================================================================
# 4. Brief Builder & Defense Drafter
# ===========================================================================
class BriefBuilder:
    """Synthesise timeline + research into a structured legal motion."""

    def __init__(self, router: LLMRouter) -> None:
        self.router = router

    def draft(
        self,
        case_id: int,
        motion_type: str,
        instructions: str = "",
        research_memo: str = "",
        log: LogFn = _noop,
    ) -> str:
        case = db.get_case(case_id) or {}
        events = db.list_timeline(case_id)
        facts_block = "\n".join(
            f"- ({e['event_date'] or 'n.d.'}) {e['description']}"
            + (f"  [!] {e['inconsistency_note']}" if e["inconsistency"] else "")
            for e in events
        )
        log("Drafting motion with the LLM…")
        prompt = (
            "Draft a clear, well-structured legal motion for a self-represented "
            "(pro se) defendant. Use numbered sections: Introduction, Statement of "
            "Facts, Legal Argument (with headings), and a Conclusion / Prayer for "
            "Relief. Write in plain but professional legal prose. Do NOT invent "
            "citations; only use authorities supplied in the research memo.\n\n"
            f"MOTION TYPE: {motion_type}\n"
            f"CASE: {case.get('name','')} (No. {case.get('case_number','')})\n"
            f"COURT: {case.get('court','')}\n"
            f"CHARGES: {case.get('charges','')}\n\n"
            f"USER INSTRUCTIONS: {instructions or '(none)'}\n\n"
            f"ESTABLISHED FACTS / TIMELINE:\n{facts_block or '(none on file)'}\n\n"
            f"RESEARCH MEMO:\n{research_memo or '(none provided)'}"
        )
        # Heaviest synthesis task in the app — route to the heavy tier (Opus 4.8).
        return self.router.complete(prompt, tier=config.TIER_HEAVY, max_tokens=8000)


# ===========================================================================
# 5. Procedural Guardrails (deadlines + cross-exam prep)
# ===========================================================================
class ProceduralEngine:
    """Statutory deadline tickler and cross-examination question generator."""

    def __init__(self, router: LLMRouter) -> None:
        self.router = router

    def compute_deadlines(
        self,
        case_id: int,
        trigger_event: str,
        trigger_date: str,
        jurisdiction: str = "",
        log: LogFn = _noop,
    ) -> List[Dict[str, Any]]:
        """Ask the model to enumerate statutory deadlines, then store them."""
        log("Calculating statutory deadlines…")
        prompt = (
            "You are a procedural-deadline calculator. Given a triggering event and "
            "date, list the key statutory/court deadlines a pro se defendant must "
            "track. Return a JSON array; each element MUST have: "
            '"title", "days_from_trigger" (integer), "rule" (the rule/statute name '
            "if commonly known, else a short description). Be conservative and add a "
            "note in the rule field reminding the user to verify against local rules.\n\n"
            f"JURISDICTION: {jurisdiction or 'general US'}\n"
            f"TRIGGER EVENT: {trigger_event}\n"
            f"TRIGGER DATE: {trigger_date}"
        )
        try:
            data = self.router.complete_json(prompt, max_tokens=2500)
        except LLMError as exc:
            raise LLMError(f"Deadline calculation failed: {exc}")
        if isinstance(data, dict) and "deadlines" in data:
            data = data["deadlines"]
        if not isinstance(data, list):
            return []

        try:
            base = datetime.fromisoformat(trigger_date)
        except ValueError:
            base = datetime.utcnow()

        created: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            try:
                days = int(item.get("days_from_trigger", 0))
            except (ValueError, TypeError):
                days = 0
            due = (base + timedelta(days=days)).date().isoformat()
            db.add_deadline(
                case_id,
                title=str(item["title"]),
                due_date=due,
                rule=str(item.get("rule", "")),
                notes=f"Computed +{days} days from {trigger_event}",
            )
            created.append({"title": item["title"], "due_date": due, "rule": item.get("rule", "")})
        log(f"Created {len(created)} deadlines.")
        return created

    def cross_exam_questions(self, case_id: int, witness: str, log: LogFn = _noop) -> str:
        """Generate cross-examination questions targeting timeline contradictions."""
        events = db.list_timeline(case_id)
        contradictions = [e for e in events if e["inconsistency"]]
        context = contradictions or events
        compact = [
            {
                "date": e["event_date"],
                "fact": e["description"],
                "source": e["source_doc"],
                "conflict": e["inconsistency_note"],
            }
            for e in context[:40]
        ]
        log("Generating cross-examination questions…")
        prompt = (
            "Generate a focused set of cross-examination questions for the named "
            "witness. Prioritise questions that expose the contradictions in the "
            "timeline. Group questions by topic, keep each question short, leading, "
            "and answerable yes/no where possible. Add a brief note on the purpose "
            "of each group.\n\n"
            f"WITNESS: {witness}\n\nTIMELINE / CONTRADICTIONS:\n{json.dumps(compact, indent=2)}"
        )
        return self.router.complete(prompt, max_tokens=4000)


# ===========================================================================
# 6. Pro Se Toolkit — Intake Interviewer + Legalese Decoder
# ===========================================================================
class IntakeInterviewer:
    """Guided onboarding flow that establishes baseline case facts."""

    QUESTIONS = [
        ("name", "What is your full name (as it appears on court documents)?"),
        ("charges", "What are you being charged with, or what is the dispute about?"),
        ("court", "Which court is handling this (name and county/state)?"),
        ("case_number", "What is the case number, if you have one?"),
        ("judge", "Who is the assigned judge, if known?"),
        ("jurisdiction", "What state/jurisdiction's law applies?"),
        ("arrest_or_filing", "When did the key event happen (arrest / filing / incident date)?"),
        ("narrative", "In a few sentences, what happened from your point of view?"),
    ]

    def __init__(self, router: LLMRouter) -> None:
        self.router = router

    def summarize(self, answers: Dict[str, str], log: LogFn = _noop) -> str:
        log("Summarising intake into a baseline case file…")
        prompt = (
            "Summarise this pro se intake interview into a neutral baseline case "
            "summary (2-3 short paragraphs). Then list 3-5 immediate next steps the "
            "defendant should consider. Do not give a verdict or guarantees.\n\n"
            f"INTAKE ANSWERS:\n{json.dumps(answers, indent=2)}"
        )
        try:
            return self.router.complete(prompt, max_tokens=2000)
        except LLMError as exc:
            return f"(Could not summarise intake: {exc})"


class LegaleseDecoder:
    """Translate complex legal text to a 4th-grade reading level."""

    def __init__(self, router: LLMRouter) -> None:
        self.router = router

    def decode(self, text: str) -> str:
        prompt = (
            "Rewrite the following legal text so a 4th-grade reader can understand "
            "it. Use short sentences and everyday words. Keep the meaning accurate. "
            "If a legal term must be kept, define it in parentheses.\n\n"
            f"TEXT:\n{text}"
        )
        return self.router.complete(prompt, max_tokens=2000)


# ===========================================================================
# Chat orchestration (RAG + @web routing) used by the chat UI
# ===========================================================================
class ChatEngine:
    """Routes a chat turn to RAG-grounded answering or the OSINT @web agent."""

    def __init__(self, router: LLMRouter) -> None:
        self.router = router
        self.osint = OSINTAgent(router)

    def handle(self, case_id: int, user_text: str, log: LogFn = _noop) -> Dict[str, Any]:
        """Return {answer, citations}. Detects a leading/embedded @web command."""
        if "@web" in user_text.lower():
            query = re.sub(r"@web", "", user_text, flags=re.IGNORECASE).strip()
            result = self.osint.run(query or user_text, log=log)
            return {"answer": result["answer"], "citations": result.get("citations", [])}

        log("Retrieving relevant case context…")
        context = STORE.context_for(case_id, user_text)
        history = db.list_messages(case_id, limit=12)
        history_block = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        case = db.get_case(case_id) or {}
        prompt = (
            "Answer the defendant's question using the case context below. If the "
            "context is insufficient, say so and suggest what evidence would help. "
            "Be practical and plain-spoken. This is legal information, not legal "
            "advice.\n\n"
            f"CASE: {case.get('name','')} — {case.get('charges','')}\n\n"
            f"CASE CONTEXT (from evidence):\n{context or '(no indexed evidence yet)'}\n\n"
            f"RECENT CONVERSATION:\n{history_block or '(none)'}\n\n"
            f"QUESTION: {user_text}"
        )
        answer = self.router.complete(prompt, max_tokens=3000)
        return {"answer": answer, "citations": []}
