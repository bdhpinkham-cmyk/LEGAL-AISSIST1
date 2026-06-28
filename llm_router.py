"""
llm_router.py
=============
Task-tiered entry point for talking to cloud LLMs (Anthropic, OpenAI, Gemini).

Why tiers?
----------
Different tasks deserve different models. The router maps each call to one of
three tiers (configurable in Settings):

    extraction  -> Gemini 3.5 Flash  (fast, cheap, huge-context multimodal
                                       parsing / OCR of PDFs and images)
    medium      -> Sonnet 4.6 / Opus 4.6 / Gemini 3.1 Pro (everyday reasoning)
    heavy       -> Opus 4.8          (the most demanding synthesis only)

Callers select a tier:

    router.complete(prompt, tier="medium")          -> str
    router.complete_json(prompt, tier="medium")     -> dict | list
    router.complete(prompt, tier="heavy")           -> str  (brief drafting)
    router.resolve("extraction")                     -> (provider, model, key)

Multimodal document parsing itself lives in ``ingestion`` (it needs file I/O),
but it asks this router which provider/model/key to use for the extraction tier.

Pre-debugging decisions
-----------------------
* **Rate limits & transient errors:** every network call goes through
  :func:`_with_retries` (exponential backoff, ``config.MAX_API_RETRIES``).
* **Lazy SDK imports:** a user who only configured one provider never needs the
  others installed.
* **Robust JSON extraction:** models sometimes wrap JSON in fences / prose;
  :func:`_extract_json` strips fences and finds the first balanced JSON value.
* **Clear missing-key errors:** if a tier resolves to a provider without a key,
  the error names the tier and provider so the UI can guide the user.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Optional, Tuple

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
        except Exception as exc:  # noqa: BLE001 - classified below
            message = str(exc).lower()
            retryable = any(
                token in message
                for token in (
                    "rate limit", "ratelimit", "429", "overloaded", "timeout",
                    "timed out", "connection", "temporarily", "500", "502",
                    "503", "529", "resource has been exhausted", "unavailable",
                )
            )
            last_exc = exc
            if not retryable or attempt == config.MAX_API_RETRIES - 1:
                break
            time.sleep(config.RETRY_BASE_DELAY * (2 ** attempt))
    raise LLMError(f"{what} failed after retries: {last_exc}")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class LLMRouter:
    """Routes completion requests to the model configured for each task tier."""

    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        """Re-read tier routing and API keys from the settings table."""
        self.keys = {
            "anthropic": db.get_setting(config.SETTING_ANTHROPIC_KEY),
            "openai": db.get_setting(config.SETTING_OPENAI_KEY),
            "gemini": db.get_setting(config.SETTING_GEMINI_KEY),
        }
        self.tiers: Dict[str, Tuple[str, str]] = {}
        for tier, (def_provider, def_model) in config.TIER_DEFAULTS.items():
            pkey, mkey = config.tier_setting_keys(tier)
            provider = db.get_setting(pkey, def_provider)
            if provider not in config.PROVIDERS:
                provider = def_provider
            model = db.get_setting(mkey, def_model)
            self.tiers[tier] = (provider, model)

    # -- introspection -----------------------------------------------------
    def resolve(self, tier: str) -> Tuple[str, str, str]:
        """Return (provider, model, api_key) for a tier; "" key if unset."""
        provider, model = self.tiers.get(tier, config.TIER_DEFAULTS[config.TIER_MEDIUM])
        return provider, model, self.keys.get(provider, "")

    def is_configured(self, tier: str) -> bool:
        _, _, key = self.resolve(tier)
        return bool(key)

    def summary(self) -> str:
        e = self.tiers[config.TIER_EXTRACTION][1]
        m = self.tiers[config.TIER_MEDIUM][1]
        h = self.tiers[config.TIER_HEAVY][1]
        return f"Extract: {e}  ·  Medium: {m}  ·  Heavy: {h}"

    # -- public API --------------------------------------------------------
    def complete(
        self,
        prompt: str,
        tier: str = config.TIER_MEDIUM,
        system: str = "You are a meticulous legal-analysis assistant.",
        max_tokens: int = 4000,
    ) -> str:
        provider, model, key = self.resolve(tier)
        if not key:
            raise LLMError(
                f"No API key configured for the '{tier}' tier "
                f"({config.PROVIDERS[provider]['label']}). Add one in Settings."
            )
        return self._complete_with(provider, model, key, prompt, system, max_tokens)

    def complete_json(
        self,
        prompt: str,
        tier: str = config.TIER_MEDIUM,
        system: str = "You are a meticulous legal-analysis assistant. "
        "Reply with valid JSON only — no markdown, no commentary.",
        max_tokens: int = 4000,
    ) -> Any:
        raw = self.complete(prompt, tier=tier, system=system, max_tokens=max_tokens)
        return _extract_json(raw)

    # -- provider dispatch -------------------------------------------------
    def _complete_with(
        self,
        provider: str,
        model: str,
        key: str,
        prompt: str,
        system: str,
        max_tokens: int,
    ) -> str:
        if provider == "anthropic":
            return self._anthropic(model, key, prompt, system, max_tokens)
        if provider == "openai":
            return self._openai(model, key, prompt, system, max_tokens)
        if provider == "gemini":
            return self._gemini(model, key, prompt, system, max_tokens)
        raise LLMError(f"Unknown provider: {provider}")

    def _anthropic(self, model, key, prompt, system, max_tokens) -> str:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The 'anthropic' package is not installed.") from exc
        client = anthropic.Anthropic(api_key=key)

        def _call() -> str:
            resp = client.messages.create(
                model=model,
                max_tokens=min(max_tokens, 8000),
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            return "\n".join(parts).strip()

        return _with_retries(_call, what=f"Anthropic ({model})")

    def _openai(self, model, key, prompt, system, max_tokens) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The 'openai' package is not installed.") from exc
        client = OpenAI(api_key=key)

        def _call() -> str:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=min(max_tokens, 8000),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return (resp.choices[0].message.content or "").strip()

        return _with_retries(_call, what=f"OpenAI ({model})")

    def _gemini(self, model, key, prompt, system, max_tokens) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The 'google-generativeai' package is not installed.") from exc
        genai.configure(api_key=key)
        gmodel = genai.GenerativeModel(model_name=model, system_instruction=system)

        def _call() -> str:
            resp = gmodel.generate_content(
                prompt,
                generation_config={"max_output_tokens": min(max_tokens, 8192)},
            )
            return (getattr(resp, "text", "") or "").strip()

        return _with_retries(_call, what=f"Gemini ({model})")


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------
def _extract_json(raw: str) -> Any:
    """Best-effort extraction of a JSON object/array from a model response."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
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
