"""
gemini_client.py
================
Backend-agnostic Gemini access supporting **two** authentication paths:

* **AI Studio** — a Google AI API key (Google AI Pro plan).
* **Vertex AI** — a GCP project + region using Application Default Credentials
  (ADC). No API key; run ``gcloud auth application-default login`` (or attach a
  service account) before use.

Both paths are exposed through the same two functions:

    generate_text(model, prompt, system=, max_tokens=)        -> str
    generate_from_file(model, path, mime, prompt, max_tokens=) -> str   # multimodal

Implementation notes
--------------------
* Prefers the unified ``google-genai`` SDK (``from google import genai``), which
  supports both AI Studio (``Client(api_key=...)``) and Vertex
  (``Client(vertexai=True, project=, location=)``). For multimodal input it
  sends file bytes **inline** (``types.Part.from_bytes``), which works on both
  backends — the AI-Studio-only Files API is avoided so Vertex behaves the same.
  Page-batched PDFs (see ingestion.py) keep each inline payload small.
* Falls back to the legacy ``google-generativeai`` package for the API-key path
  if ``google-genai`` is not installed (Vertex then requires the new SDK).
* Transient failures are retried with exponential backoff.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import config
import database as db


class GeminiError(Exception):
    """User-facing Gemini configuration / call failure."""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def _settings() -> dict:
    return {
        "backend": db.get_setting(config.SETTING_GEMINI_BACKEND, "api_key"),
        "api_key": db.get_setting(config.SETTING_GEMINI_KEY),
        "project": db.get_setting(config.SETTING_VERTEX_PROJECT),
        "location": db.get_setting(config.SETTING_VERTEX_LOCATION, "us-central1"),
    }


def is_configured() -> bool:
    s = _settings()
    if s["backend"] == "vertex":
        return bool(s["project"])
    return bool(s["api_key"])


def backend_label() -> str:
    s = _settings()
    if s["backend"] == "vertex":
        return f"Vertex AI ({s['project'] or 'no project'} / {s['location']})"
    return "Google AI Studio (API key)"


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------
def _new_client() -> Tuple[str, object]:
    """Return ("genai" | "legacy", client). Raises GeminiError if unconfigured."""
    s = _settings()
    try:
        from google import genai  # unified SDK
    except ImportError:
        genai = None

    if genai is not None:
        if s["backend"] == "vertex":
            if not s["project"]:
                raise GeminiError("Vertex AI is selected but no GCP project is set.")
            return "genai", genai.Client(
                vertexai=True,
                project=s["project"],
                location=s["location"] or "us-central1",
            )
        if not s["api_key"]:
            raise GeminiError("Gemini API key is not configured.")
        return "genai", genai.Client(api_key=s["api_key"])

    # Fallback: legacy google-generativeai (API-key path only).
    if s["backend"] == "vertex":
        raise GeminiError(
            "The Vertex AI backend requires the 'google-genai' package. "
            "Install it (pip install google-genai) or switch to the API-key backend."
        )
    try:
        import google.generativeai as legacy
    except ImportError as exc:  # pragma: no cover
        raise GeminiError(
            "Neither 'google-genai' nor 'google-generativeai' is installed."
        ) from exc
    if not s["api_key"]:
        raise GeminiError("Gemini API key is not configured.")
    legacy.configure(api_key=s["api_key"])
    return "legacy", legacy


def _retry(fn, what: str):
    last_exc: Optional[Exception] = None
    for attempt in range(config.MAX_API_RETRIES):
        try:
            return fn()
        except GeminiError:
            raise
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            retryable = any(
                t in msg
                for t in (
                    "429", "rate", "exhausted", "unavailable", "timeout",
                    "timed out", "connection", "500", "502", "503", "internal",
                )
            )
            last_exc = exc
            if not retryable or attempt == config.MAX_API_RETRIES - 1:
                break
            time.sleep(config.RETRY_BASE_DELAY * (2 ** attempt))
    raise GeminiError(f"{what} failed: {last_exc}")


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------
def generate_text(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 8192,
) -> str:
    kind, client = _new_client()

    def _call() -> str:
        if kind == "genai":
            from google.genai import types

            cfg = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                system_instruction=system,
            )
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            return (getattr(resp, "text", "") or "").strip()
        gmodel = client.GenerativeModel(model_name=model, system_instruction=system)
        resp = gmodel.generate_content(
            prompt, generation_config={"max_output_tokens": max_tokens}
        )
        return (getattr(resp, "text", "") or "").strip()

    return _retry(_call, f"Gemini text ({model})")


# ---------------------------------------------------------------------------
# Multimodal (file) generation
# ---------------------------------------------------------------------------
def generate_from_file(
    model: str,
    path: str,
    mime: str,
    prompt: str,
    max_tokens: int = 8192,
) -> str:
    kind, client = _new_client()

    if kind == "genai":
        from google.genai import types

        with open(path, "rb") as fh:
            data = fh.read()

        def _call() -> str:
            part = types.Part.from_bytes(data=data, mime_type=mime)
            cfg = types.GenerateContentConfig(max_output_tokens=max_tokens)
            resp = client.models.generate_content(
                model=model, contents=[part, prompt], config=cfg
            )
            return (getattr(resp, "text", "") or "").strip()

        return _retry(_call, f"Gemini multimodal ({model})")

    # Legacy API-key path: upload + poll + delete.
    legacy = client

    def _call_legacy() -> str:
        file = legacy.upload_file(path=path, mime_type=mime)
        waited = 0.0
        while getattr(file.state, "name", "ACTIVE") == "PROCESSING":
            if waited >= config.GEMINI_UPLOAD_POLL_MAX:
                raise GeminiError("Gemini file processing timed out.")
            time.sleep(config.GEMINI_UPLOAD_POLL_SECONDS)
            waited += config.GEMINI_UPLOAD_POLL_SECONDS
            file = legacy.get_file(file.name)
        if getattr(file.state, "name", "ACTIVE") == "FAILED":
            raise GeminiError("Gemini could not process the uploaded file.")
        try:
            gmodel = legacy.GenerativeModel(model_name=model)
            resp = gmodel.generate_content(
                [file, prompt], generation_config={"max_output_tokens": max_tokens}
            )
            return (getattr(resp, "text", "") or "").strip()
        finally:
            try:
                legacy.delete_file(file.name)
            except Exception:  # noqa: BLE001
                pass

    return _retry(_call_legacy, f"Gemini multimodal ({model})")
