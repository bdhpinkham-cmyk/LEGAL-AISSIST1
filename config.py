"""
config.py
=========
Central configuration, filesystem paths and constants for the
Pro Se Legal Intelligence application.

All persistent data lives under a single application-data directory that is
created on first run. Nothing is written outside of this directory, which keeps
client data local to the device (a hard requirement of the project).

This module deliberately has *no* third-party dependencies so that it can be
imported from anywhere (including PyInstaller hooks) without side effects.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "ProSeLegalIntelligence"
APP_TITLE = "Pro Se Legal Intelligence"
APP_VERSION = "1.0.0"


def _platform_data_root() -> Path:
    """Return a writable, platform-appropriate base directory.

    * Windows -> %APPDATA%\\ProSeLegalIntelligence
    * macOS   -> ~/Library/Application Support/ProSeLegalIntelligence
    * Linux   -> $XDG_DATA_HOME or ~/.local/share/ProSeLegalIntelligence
    * Android -> Flet exposes a writable app storage dir via FLET_APP_STORAGE_DATA
    """
    # Flet on mobile sets this env var to the per-app writable storage path.
    flet_storage = os.environ.get("FLET_APP_STORAGE_DATA")
    if flet_storage:
        return Path(flet_storage) / APP_NAME

    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return Path(base) / APP_NAME

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME

    # Linux / other POSIX
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else (Path.home() / ".local" / "share")
    return base / APP_NAME


# ---------------------------------------------------------------------------
# Resolved paths (created eagerly so the rest of the app can assume they exist)
# ---------------------------------------------------------------------------
DATA_ROOT: Path = _platform_data_root()
DB_PATH: Path = DATA_ROOT / "workspaces.db"
VECTOR_DB_PATH: Path = DATA_ROOT / "chroma"
EVIDENCE_DIR: Path = DATA_ROOT / "evidence"
EXPORT_DIR: Path = DATA_ROOT / "exports"
BROWSER_PROFILE_DIR: Path = DATA_ROOT / "browser_profile"
LOG_DIR: Path = DATA_ROOT / "logs"


def ensure_directories() -> None:
    """Create every directory the application relies on. Safe to call repeatedly."""
    for directory in (
        DATA_ROOT,
        VECTOR_DB_PATH,
        EVIDENCE_DIR,
        EXPORT_DIR,
        BROWSER_PROFILE_DIR,
        LOG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LLM provider catalogue
# ---------------------------------------------------------------------------
# The default model strings below are the current, correct public API model IDs
# for each provider as of this build. They can be overridden per-provider in the
# Settings screen.
PROVIDERS = {
    "anthropic": {
        "label": "Anthropic (Claude)",
        "default_model": "claude-opus-4-8",
        "models": [
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ],
        "settings_key": "anthropic_api_key",
    },
    "openai": {
        "label": "OpenAI (GPT)",
        "default_model": "gpt-4o",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "settings_key": "openai_api_key",
    },
    "gemini": {
        "label": "Google (Gemini)",
        "default_model": "gemini-1.5-pro",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash"],
        "settings_key": "gemini_api_key",
    },
}

DEFAULT_PROVIDER = "anthropic"

# Settings keys (stored in the global ``settings`` table).
SETTING_ACTIVE_PROVIDER = "active_provider"
SETTING_ACTIVE_MODEL = "active_model"
SETTING_TAVILY_KEY = "tavily_api_key"
SETTING_DEEPGRAM_KEY = "deepgram_api_key"
SETTING_OPENAI_KEY = "openai_api_key"
SETTING_ANTHROPIC_KEY = "anthropic_api_key"
SETTING_GEMINI_KEY = "gemini_api_key"

# Hard limits to keep iterative agents from running away.
MAX_RESEARCH_ITERATIONS = 5
MAX_AGENT_TOOL_STEPS = 8

# Networking / retry behaviour.
HTTP_TIMEOUT_SECONDS = 60
MAX_API_RETRIES = 4
RETRY_BASE_DELAY = 2.0  # seconds; exponential backoff base

# RAG chunking parameters.
RAG_CHUNK_SIZE = 1200  # characters
RAG_CHUNK_OVERLAP = 200
RAG_TOP_K = 6
