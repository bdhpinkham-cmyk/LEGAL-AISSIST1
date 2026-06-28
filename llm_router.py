"""
llm_router.py
=============
A single entry point for talking to cloud LLMs (Anthropic, OpenAI, Gemini).

Why a router?
-------------
Local desktop/mobile hardware can't run a frontier model, so heavy reasoning is
delegated to a cloud provider chosen by the user. The router hides the
per-provider SDK differences behind two methods:

    router.complete(prompt, system=..., max_tokens=...)   -> str
    router.complete_json(prompt, system=..., schema_hint=) -> dict | list

Pre-debugging decisions
-----------------------
* **Rate limits & transient errors:** Every network call is wrapped in
  :func:`_with_retries`, which retries on rate-limit / 5xx / connection errors
  using exponential backoff (config.MAX_API_RETRIES attempts). Non-retryable
  client errors (bad key, bad request) are surfaced immediately as
  :class:`LLMError` so the UI can show a clear message instead of hanging.

* **No hard SDK import at module load:** Provider SDKs are imported lazily
  inside the call path. A user who only configured Anthropic never needs the
  OpenAI/Gemini packages installed for the app to import.

* **Robust JSON extraction:** Models sometimes wrap JSON in markdown fences or
  add prose. :meth:`complete_json` strips fences and locates the first valid
  JSON object/array, raising a clear error if none is found.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

import config
import database as db


class LLMError(Exception):
    """Raised for unrecoverable LLM problems (bad key, exhausted retries…)."""


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
def _with_retries(fn, *, what: str):
    """Call ``fn`` with exponential backoff on transient failures."""
    last_exc: Optional[Exception] = None
    for attempt in range(config.MAX_API_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we classify below
            message = str(exc).lower()
            retryable = any(
                token in message
                for token in (
                    "rate limit",
                    "ratelimit",
                    "429",
                    "overloaded",
                    "timeout",
                    "timed out",
                    "connection",
                    "temporarily",
                    "500",
                    "502",
                    "503",
                    "529",
                )
            )
            last_exc = exc
            if not retryable or attempt == config.MAX_API_RETRIES - 1:
                break
            delay = config.RETRY_BASE_DELAY * (2 ** attempt)
            time.sleep(delay)
    raise LLMError(f"{what} failed after retries: {last_exc}")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class LLMRouter:
    """Routes completion requests to the user-selected provider."""

    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        """Re-read provider / model / key selection from the settings table."""
        self.provider = db.get_setting(
            config.SETTING_ACTIVE_PROVIDER, config.DEFAULT_PROVIDER
        )
        if self.provider not in config.PROVIDERS:
            self.provider = config.DEFAULT_PROVIDER
        default_model = config.PROVIDERS[self.provider]["default_model"]
        self.model = db.get_setting(config.SETTING_ACTIVE_MODEL, default_model)
        self.keys = {
            "anthropic": db.get_setting(config.SETTING_ANTHROPIC_KEY),
            "openai": db.get_setting(config.SETTING_OPENAI_KEY),
            "gemini": db.get_setting(config.SETTING_GEMINI_KEY),
        }

    # -- public API --------------------------------------------------------
    def is_configured(self) -> bool:
        return bool(self.keys.get(self.provider))

    def active_label(self) -> str:
        return f"{config.PROVIDERS[self.provider]['label']} · {self.model}"

    def complete(
        self,
        prompt: str,
        system: str = "You are a meticulous legal-analysis assistant.",
        max_tokens: int = 4000,
    ) -> str:
        """Return the model's text response for ``prompt``."""
        if not self.is_configured():
            raise LLMError(
                f"No API key configured for {config.PROVIDERS[self.provider]['label']}. "
                "Add one in Settings."
            )
        if self.provider == "anthropic":
            return self._complete_anthropic(prompt, system, max_tokens)
        if self.provider == "openai":
            return self._complete_openai(prompt, system, max_tokens)
        if self.provider == "gemini":
            return self._complete_gemini(prompt, system, max_tokens)
        raise LLMError(f"Unknown provider: {self.provider}")

    def complete_json(
        self,
        prompt: str,
        system: str = "You are a meticulous legal-analysis assistant. "
        "Reply with valid JSON only — no markdown, no commentary.",
        max_tokens: int = 4000,
    ) -> Any:
        """Return parsed JSON from the model. Raises LLMError on parse failure."""
        raw = self.complete(prompt, system=system, max_tokens=max_tokens)
        return _extract_json(raw)

    # -- provider implementations -----------------------------------------
    def _complete_anthropic(self, prompt: str, system: str, max_tokens: int) -> str:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The 'anthropic' package is not installed.") from exc

        client = anthropic.Anthropic(api_key=self.keys["anthropic"])

        def _call() -> str:
            resp = client.messages.create(
                model=self.model,
                max_tokens=min(max_tokens, 8000),
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            return "\n".join(parts).strip()

        return _with_retries(_call, what="Anthropic completion")

    def _complete_openai(self, prompt: str, system: str, max_tokens: int) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The 'openai' package is not installed.") from exc

        client = OpenAI(api_key=self.keys["openai"])

        def _call() -> str:
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=min(max_tokens, 8000),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return (resp.choices[0].message.content or "").strip()

        return _with_retries(_call, what="OpenAI completion")

    def _complete_gemini(self, prompt: str, system: str, max_tokens: int) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The 'google-generativeai' package is not installed.") from exc

        genai.configure(api_key=self.keys["gemini"])
        model = genai.GenerativeModel(
            model_name=self.model, system_instruction=system
        )

        def _call() -> str:
            resp = model.generate_content(
                prompt,
                generation_config={"max_output_tokens": min(max_tokens, 8000)},
            )
            return (resp.text or "").strip()

        return _with_retries(_call, what="Gemini completion")


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------
def _extract_json(raw: str) -> Any:
    """Best-effort extraction of a JSON object/array from a model response."""
    text = raw.strip()

    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to locating the first balanced JSON object or array.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise LLMError("Model did not return valid JSON.")
