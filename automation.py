"""
automation.py
=============
Autonomous Court Portal Agent (browser automation via Playwright).

What it does
------------
Given locally-stored portal credentials and a search query, the agent opens a
real browser, logs in, runs the search, opens dockets, downloads PDFs/minutes,
and injects them into the local RAG database for the active case.

Critical security guardrail
---------------------------
The agent must **never** attempt to bypass CAPTCHAs, bot-protection screens, or
ToS blockers. :meth:`_is_blocked` detects those screens. When one is hit the
agent:

  1. stops automated actions,
  2. raises a visible browser window (it always launches non-headless),
  3. calls ``alert("Manual intervention required …")`` so the Chat UI tells the
     user, and
  4. blocks on ``wait_for_user()`` until the user solves the challenge in the
     real browser and clicks "Resume" — then it continues its workflow.

Threading
---------
This module uses Playwright's **sync** API and is intended to be driven from a
dedicated worker thread (see ui.py). It does not touch Flet. Communication with
the UI is entirely through the injected ``log`` / ``alert`` / ``wait_for_user``
callbacks, so the UI thread never blocks.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional

import config
import database as db
from ingestion import extract_text
from rag import STORE

LogFn = Callable[[str], None]
AlertFn = Callable[[str], None]
WaitFn = Callable[[], bool]  # returns True to resume, False to abort


def _noop(_: str) -> None:
    pass


# Phrases that indicate a bot-protection / CAPTCHA / ToS wall.
_BLOCK_MARKERS = (
    "captcha",
    "recaptcha",
    "hcaptcha",
    "are you human",
    "verify you are human",
    "i'm not a robot",
    "unusual traffic",
    "access denied",
    "cloudflare",
    "checking your browser",
    "please verify",
    "security check",
    "rate limit",
    "blocked",
)


class CaptchaInterrupt(Exception):
    """Internal signal that a bot-protection screen was encountered."""


class CourtPortalAgent:
    """Drives a real browser to fetch public docket documents."""

    def __init__(self) -> None:
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    # ------------------------------------------------------------------
    def run(
        self,
        case_id: int,
        credential: Dict[str, str],
        query: str,
        log: LogFn = _noop,
        alert: AlertFn = _noop,
        wait_for_user: WaitFn = lambda: True,
    ) -> Dict[str, object]:
        """Execute the portal workflow. Returns {downloaded, ingested, notes}."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Playwright is not installed. Run: pip install playwright && "
                "playwright install chromium"
            ) from exc

        config.ensure_directories()
        downloaded: List[str] = []
        notes: List[str] = []

        with sync_playwright() as p:
            # Always launch headed so the user can solve any challenge that appears.
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()
            try:
                self._guarded(
                    page,
                    lambda: page.goto(credential["url"], timeout=60000),
                    log,
                    alert,
                    wait_for_user,
                    "Opening portal",
                )

                log("Attempting login…")
                self._login(page, credential, log, alert, wait_for_user)

                log(f"Searching for: {query}")
                self._search(page, query, log, alert, wait_for_user)

                log("Collecting document links…")
                pdf_links = self._collect_pdf_links(page)
                notes.append(f"Found {len(pdf_links)} candidate documents.")

                for i, link in enumerate(pdf_links[:15], 1):
                    if self._abort:
                        notes.append("Aborted by user.")
                        break
                    log(f"Downloading document {i}/{min(len(pdf_links),15)}…")
                    saved = self._download(context, link)
                    if saved:
                        downloaded.append(saved)

            except CaptchaInterrupt:
                notes.append("Stopped at a bot-protection screen (handled).")
            finally:
                # Give the user a beat to see the final state, then close.
                time.sleep(1.0)
                context.close()
                browser.close()

        log("Ingesting downloaded documents into the case RAG store…")
        ingested = 0
        for path in downloaded:
            try:
                text = extract_text(path)
                if not text.strip():
                    continue
                doc_id = db.add_document(
                    case_id, os.path.basename(path), path, "portal", text, {"source": "court_portal"}
                )
                STORE.add_document(case_id, doc_id, os.path.basename(path), text)
                ingested += 1
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Ingest failed for {os.path.basename(path)}: {exc}")

        return {"downloaded": len(downloaded), "ingested": ingested, "notes": notes}

    # ------------------------------------------------------------------
    # Guardrail-wrapped helpers
    # ------------------------------------------------------------------
    def _guarded(
        self,
        page,
        action: Callable[[], object],
        log: LogFn,
        alert: AlertFn,
        wait_for_user: WaitFn,
        description: str,
    ) -> None:
        """Run ``action`` then check for a block screen; pause for the user if hit."""
        action()
        page.wait_for_timeout(1500)
        attempts = 0
        while self._is_blocked(page):
            attempts += 1
            if attempts > 5:
                raise CaptchaInterrupt()
            alert(
                "Manual intervention required: a CAPTCHA or security check appeared "
                f"during '{description}'. Please solve it in the open browser window, "
                "then click Resume."
            )
            log("Paused — waiting for the user to clear the security check…")
            resume = wait_for_user()
            if not resume:
                self._abort = True
                raise CaptchaInterrupt()
            page.wait_for_timeout(1500)

    def _is_blocked(self, page) -> bool:
        """Heuristically detect a CAPTCHA / bot-protection / ToS wall."""
        try:
            content = (page.content() or "").lower()
        except Exception:  # noqa: BLE001
            return False
        if any(marker in content for marker in _BLOCK_MARKERS):
            return True
        # reCAPTCHA / hCaptcha iframes.
        try:
            for frame in page.frames:
                url = (frame.url or "").lower()
                if "recaptcha" in url or "hcaptcha" in url or "captcha" in url:
                    return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def _login(
        self,
        page,
        credential: Dict[str, str],
        log: LogFn,
        alert: AlertFn,
        wait_for_user: WaitFn,
    ) -> None:
        username = credential.get("username", "")
        password = credential.get("password", "")
        if not username and not password:
            log("No credentials supplied; proceeding as a public/guest search.")
            return

        # Heuristic field discovery — portals vary widely.
        user_selectors = [
            "input[type=email]",
            "input[name*='user' i]",
            "input[id*='user' i]",
            "input[name*='login' i]",
            "input[name*='email' i]",
        ]
        pass_selectors = ["input[type=password]", "input[name*='pass' i]"]

        filled_user = self._fill_first(page, user_selectors, username)
        filled_pass = self._fill_first(page, pass_selectors, password)

        if not (filled_user or filled_pass):
            alert(
                "Manual intervention required: could not locate the login fields "
                "automatically. Please log in manually in the browser window, then "
                "click Resume."
            )
            if not wait_for_user():
                self._abort = True
                raise CaptchaInterrupt()
            return

        # Submit.
        for sel in ["button[type=submit]", "input[type=submit]", "button:has-text('Log')",
                    "button:has-text('Sign')"]:
            try:
                if page.locator(sel).count():
                    page.locator(sel).first.click(timeout=5000)
                    break
            except Exception:  # noqa: BLE001
                continue
        else:
            try:
                page.keyboard.press("Enter")
            except Exception:  # noqa: BLE001
                pass

        page.wait_for_timeout(2500)
        self._guarded(page, lambda: None, log, alert, wait_for_user, "login")

    def _search(
        self,
        page,
        query: str,
        log: LogFn,
        alert: AlertFn,
        wait_for_user: WaitFn,
    ) -> None:
        search_selectors = [
            "input[type=search]",
            "input[name*='search' i]",
            "input[id*='search' i]",
            "input[name*='case' i]",
            "input[placeholder*='search' i]",
            "input[placeholder*='case' i]",
        ]
        if self._fill_first(page, search_selectors, query):
            try:
                page.keyboard.press("Enter")
            except Exception:  # noqa: BLE001
                pass
            page.wait_for_timeout(2500)
        else:
            alert(
                "Manual intervention required: could not find the search box "
                "automatically. Please run the search manually in the browser, then "
                "click Resume."
            )
            if not wait_for_user():
                self._abort = True
                raise CaptchaInterrupt()
        self._guarded(page, lambda: None, log, alert, wait_for_user, "search")

    def _fill_first(self, page, selectors: List[str], value: str) -> bool:
        if not value:
            return False
        for sel in selectors:
            try:
                loc = page.locator(sel)
                if loc.count():
                    loc.first.fill(value, timeout=5000)
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    def _collect_pdf_links(self, page) -> List[str]:
        links: List[str] = []
        try:
            anchors = page.locator("a")
            count = min(anchors.count(), 300)
            for i in range(count):
                try:
                    href = anchors.nth(i).get_attribute("href")
                except Exception:  # noqa: BLE001
                    continue
                if not href:
                    continue
                low = href.lower()
                if ".pdf" in low or "document" in low or "minute" in low or "docket" in low:
                    absolute = self._absolute(page, href)
                    if absolute and absolute not in links:
                        links.append(absolute)
        except Exception:  # noqa: BLE001
            pass
        return links

    @staticmethod
    def _absolute(page, href: str) -> str:
        if href.startswith("http"):
            return href
        try:
            from urllib.parse import urljoin

            return urljoin(page.url, href)
        except Exception:  # noqa: BLE001
            return ""

    def _download(self, context, url: str) -> Optional[str]:
        try:
            # Use the browser context's request API to inherit the auth session.
            resp = context.request.get(url, timeout=60000)
            if not resp.ok:
                return None
            body = resp.body()
            if not body:
                return None
            name = url.split("/")[-1].split("?")[0] or f"portal_{int(time.time())}.pdf"
            if not name.lower().endswith(".pdf"):
                name += ".pdf"
            safe = "".join(c for c in name if c.isalnum() or c in "._-") or "portal.pdf"
            dest = str(config.EVIDENCE_DIR / f"{int(time.time()*1000)}_{safe}")
            with open(dest, "wb") as fh:
                fh.write(body)
            return dest
        except Exception:  # noqa: BLE001
            return None
