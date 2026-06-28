"""
main.py
=======
Entry point for the Pro Se Legal Intelligence desktop/mobile application.

Run locally:        python main.py
Build Windows .exe: see README.md → Packaging Guide
Build Android .apk:  flet build apk
"""

from __future__ import annotations

import logging

import flet as ft

import config
import database as db
from ui import AppUI


def _configure_logging() -> None:
    config.ensure_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_DIR / "app.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main(page: ft.Page) -> None:
    """Flet application target."""
    config.ensure_directories()
    db.init_db()
    AppUI(page)


def run() -> None:
    _configure_logging()
    logging.getLogger(__name__).info("Starting %s v%s", config.APP_TITLE, config.APP_VERSION)
    # ft.app picks the desktop window on Windows/macOS/Linux and the native
    # view on Android when built with `flet build apk`.
    ft.app(target=main)


if __name__ == "__main__":
    run()
