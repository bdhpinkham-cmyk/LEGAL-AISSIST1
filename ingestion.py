"""
ingestion.py
============
File ingestion + document production helpers used by the agentic engines.

Responsibilities
----------------
* Text extraction from PDFs and plain-text files.
* EXIF / metadata extraction from images (the "Metadata Extractor" toolkit item)
  to help authenticate the timeline.
* Audio / video transcription via the Deepgram or OpenAI Whisper REST APIs.
* PII auto-redaction (SSNs, phone numbers, emails, and configurable minor names).
* Court-ready DOCX export onto numbered pleading paper (lines 1-28 + caption).

Every function degrades gracefully: if an optional dependency is missing it
raises a clear, user-facing message rather than crashing the UI thread.
"""

from __future__ import annotations

import io
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests

import config


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def extract_text(path: str) -> str:
    """Extract text from a supported file (PDF / TXT / MD). Returns ''."""
    lower = path.lower()
    if lower.endswith(".pdf"):
        return _extract_pdf(path)
    if lower.endswith((".txt", ".md", ".csv", ".log")):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError as exc:
            raise IngestionError(f"Could not read {os.path.basename(path)}: {exc}")
    # Unknown text type: attempt a best-effort decode.
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _extract_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise IngestionError("The 'pypdf' package is required to read PDFs.") from exc
    try:
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"Failed to parse PDF: {exc}") from exc


class IngestionError(Exception):
    """User-facing ingestion failure."""


# ---------------------------------------------------------------------------
# Image metadata (EXIF) extraction
# ---------------------------------------------------------------------------
def extract_image_metadata(path: str) -> Dict[str, str]:
    """Return a flat dict of useful EXIF fields (timestamp, GPS, camera)."""
    try:
        from PIL import Image, ExifTags
    except ImportError as exc:  # pragma: no cover
        raise IngestionError("The 'Pillow' package is required to read image metadata.") from exc

    meta: Dict[str, str] = {}
    try:
        with Image.open(path) as img:
            meta["format"] = img.format or ""
            meta["size"] = f"{img.width}x{img.height}"
            exif = img.getexif()
            if not exif:
                return meta
            tag_map = {v: k for k, v in ExifTags.TAGS.items()}
            for tag_id, value in exif.items():
                name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if name in ("DateTime", "DateTimeOriginal", "Make", "Model", "Software"):
                    meta[name] = str(value)
            # GPS sub-IFD
            gps_ifd_id = tag_map.get("GPSInfo")
            if gps_ifd_id and gps_ifd_id in exif:
                gps = exif.get_ifd(gps_ifd_id)
                coords = _parse_gps(gps)
                if coords:
                    meta["GPSLatitude"], meta["GPSLongitude"] = coords
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)
    return meta


def _parse_gps(gps: Dict) -> Optional[Tuple[str, str]]:
    try:
        from PIL import ExifTags

        gps_tags = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}

        def _to_deg(values, ref) -> float:
            d, m, s = [float(x) for x in values]
            deg = d + m / 60.0 + s / 3600.0
            if ref in ("S", "W"):
                deg = -deg
            return deg

        lat = gps_tags.get("GPSLatitude")
        lat_ref = gps_tags.get("GPSLatitudeRef")
        lon = gps_tags.get("GPSLongitude")
        lon_ref = gps_tags.get("GPSLongitudeRef")
        if lat and lon and lat_ref and lon_ref:
            return (
                f"{_to_deg(lat, lat_ref):.6f}",
                f"{_to_deg(lon, lon_ref):.6f}",
            )
    except Exception:  # noqa: BLE001
        return None
    return None


# ---------------------------------------------------------------------------
# Audio / video transcription
# ---------------------------------------------------------------------------
def transcribe_audio(
    path: str, deepgram_key: str = "", openai_key: str = ""
) -> str:
    """Transcribe an audio/video file. Prefers Deepgram, falls back to Whisper."""
    if deepgram_key:
        return _transcribe_deepgram(path, deepgram_key)
    if openai_key:
        return _transcribe_whisper(path, openai_key)
    raise IngestionError(
        "Transcription requires a Deepgram or OpenAI API key (set one in Settings)."
    )


def _transcribe_deepgram(path: str, key: str) -> str:
    url = "https://api.deepgram.com/v1/listen?smart_format=true&punctuate=true"
    headers = {"Authorization": f"Token {key}", "Content-Type": "application/octet-stream"}
    last_exc: Optional[Exception] = None
    for attempt in range(config.MAX_API_RETRIES):
        try:
            with open(path, "rb") as fh:
                resp = requests.post(
                    url, headers=headers, data=fh, timeout=300
                )
            if resp.status_code == 429:
                raise RuntimeError("rate limit")
            resp.raise_for_status()
            data = resp.json()
            return (
                data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < config.MAX_API_RETRIES - 1:
                time.sleep(config.RETRY_BASE_DELAY * (2 ** attempt))
    raise IngestionError(f"Deepgram transcription failed: {last_exc}")


def _transcribe_whisper(path: str, key: str) -> str:
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {key}"}
    last_exc: Optional[Exception] = None
    for attempt in range(config.MAX_API_RETRIES):
        try:
            with open(path, "rb") as fh:
                files = {"file": (os.path.basename(path), fh)}
                data = {"model": "whisper-1"}
                resp = requests.post(
                    url, headers=headers, files=files, data=data, timeout=300
                )
            if resp.status_code == 429:
                raise RuntimeError("rate limit")
            resp.raise_for_status()
            return resp.json().get("text", "").strip()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < config.MAX_API_RETRIES - 1:
                time.sleep(config.RETRY_BASE_DELAY * (2 ** attempt))
    raise IngestionError(f"Whisper transcription failed: {last_exc}")


# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CC_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


def redact_pii(text: str, minor_names: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    """Scrub PII from ``text``. Returns (redacted_text, list_of_redaction_notes)."""
    notes: List[str] = []
    redacted = text

    def _sub(pattern: re.Pattern, label: str, value: str) -> None:
        nonlocal redacted
        count = len(pattern.findall(redacted))
        if count:
            redacted = pattern.sub(value, redacted)
            notes.append(f"Redacted {count} {label}")

    _sub(_SSN_RE, "SSN(s)", "[REDACTED-SSN]")
    _sub(_CC_RE, "possible card number(s)", "[REDACTED-NUMBER]")
    _sub(_PHONE_RE, "phone number(s)", "[REDACTED-PHONE]")
    _sub(_EMAIL_RE, "email address(es)", "[REDACTED-EMAIL]")

    for name in minor_names or []:
        name = name.strip()
        if not name:
            continue
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        count = len(pattern.findall(redacted))
        if count:
            redacted = pattern.sub("[REDACTED-MINOR]", redacted)
            notes.append(f"Redacted {count} reference(s) to a protected name")

    return redacted, notes


# ---------------------------------------------------------------------------
# Court-ready DOCX export (numbered pleading paper)
# ---------------------------------------------------------------------------
def export_pleading_docx(
    out_path: str,
    *,
    party_name: str,
    court: str,
    case_number: str,
    judge: str,
    title: str,
    body: str,
    plaintiff: str = "THE PEOPLE / PLAINTIFF",
    defendant: str = "",
) -> str:
    """Render a legal pleading to DOCX with a caption and 1-28 line numbers.

    Line numbering on pleading paper is produced by enabling Word's built-in
    line-number field in the section properties (restarts each page), which is
    how real pleading templates do it — this keeps numbering correct regardless
    of how the text reflows. A numbered left column is also drawn for on-screen
    fidelity.
    """
    try:
        from docx import Document
        from docx.enum.section import WD_SECTION
        from docx.shared import Pt, Inches
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
    except ImportError as exc:  # pragma: no cover
        raise IngestionError("The 'python-docx' package is required for DOCX export.") from exc

    doc = Document()
    section = doc.sections[0]
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(0.5)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)

    # Enable Word line numbering (continuous, restart each page) on the section.
    sectPr = section._sectPr
    ln = OxmlElement("w:lnNumType")
    ln.set(qn("w:countBy"), "1")
    ln.set(qn("w:start"), "1")
    ln.set(qn("w:restart"), "newPage")
    ln.set(qn("w:distance"), "360")
    sectPr.append(ln)

    base = doc.styles["Normal"]
    base.font.name = "Times New Roman"
    base.font.size = Pt(12)

    # Party / attorney block.
    head = doc.add_paragraph()
    head.add_run(f"{party_name}\nDefendant In Pro Per\n").bold = False

    # Court caption.
    court_p = doc.add_paragraph()
    run = court_p.add_run(court.upper() if court else "SUPERIOR COURT OF THE STATE")
    run.bold = True
    court_p.alignment = 1  # centre

    # Caption table: parties vs case number.
    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    left = table.cell(0, 0)
    right = table.cell(0, 1)
    left.text = (
        f"{plaintiff},\n        Plaintiff,\n   vs.\n"
        f"{defendant or party_name},\n        Defendant."
    )
    right.text = (
        f"Case No.: {case_number or '__________'}\n\n"
        f"{title.upper()}\n\n"
        f"Judge: {judge or '__________'}"
    )

    doc.add_paragraph("")

    # Body with an explicit numbered column for on-screen pleading fidelity.
    line_no = 1
    for raw_line in body.split("\n"):
        # Wrap long lines so the numbered column stays readable.
        for segment in _wrap(raw_line, 95) or [""]:
            p = doc.add_paragraph()
            num_run = p.add_run(f"{line_no:>2}  ")
            num_run.font.size = Pt(9)
            p.add_run(segment)
            line_no += 1
            if line_no > 28:
                line_no = 1

    config.ensure_directories()
    doc.save(out_path)
    return out_path


def _wrap(text: str, width: int) -> List[str]:
    if not text:
        return [""]
    words = text.split(" ")
    lines: List[str] = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 > width and current:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}".strip()
    if current:
        lines.append(current)
    return lines
