"""
ui.py
=====
Flet user interface for the Pro Se Legal Intelligence application.

Architecture / pre-debugging decisions
--------------------------------------
* **Never block the UI thread.** Every potentially-slow operation (LLM calls,
  transcription, research loops, browser automation) runs on a daemon worker
  thread via :meth:`AppUI.run_bg`. Workers communicate back by mutating controls
  and calling ``page.update()`` — which Flet supports from background threads.

* **Version-robust Flet usage.** Icons are passed as plain name strings
  (e.g. ``"gavel"``) and colours as hex strings, so the module works across the
  ``ft.icons``/``ft.Icons`` and ``ft.colors``/``ft.Colors`` renames.

* **Case isolation in the UI.** ``self.case_id`` is the single source of truth
  for the active case; every data call passes it through to the database layer.

* **Court-portal pause/resume.** The automation agent blocks on a
  ``threading.Event`` when it hits a CAPTCHA; the UI surfaces a banner and
  Resume/Abort buttons that set the event, so manual intervention works without
  freezing anything.
"""

from __future__ import annotations

import os
import threading
from datetime import date, datetime
from typing import Callable, Dict, List, Optional

import flet as ft

import config
import database as db
from agents import (
    BriefBuilder,
    ChatEngine,
    DeepResearchAgent,
    DiscoveryTimelineEngine,
    IntakeInterviewer,
    LegaleseDecoder,
    ProceduralEngine,
)
from automation import CourtPortalAgent
from ingestion import IngestionError, export_pleading_docx, redact_pii
from llm_router import LLMError, LLMRouter

# --- palette ---------------------------------------------------------------
BG = "#0E1116"
SURFACE = "#161B22"
SURFACE_2 = "#1F2630"
ACCENT = "#3B82F6"
ACCENT_2 = "#22D3EE"
DANGER = "#EF4444"
WARN = "#F59E0B"
OK = "#22C55E"
TEXT = "#E6EDF3"
MUTED = "#8B949E"


class AppUI:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self.router = LLMRouter()
        self.case_id: Optional[int] = None

        # Engines (constructed lazily-ish; they only hold the router).
        self.discovery = DiscoveryTimelineEngine(self.router)
        self.research = DeepResearchAgent(self.router)
        self.brief = BriefBuilder(self.router)
        self.procedural = ProceduralEngine(self.router)
        self.intake = IntakeInterviewer(self.router)
        self.decoder = LegaleseDecoder(self.router)
        self.chat = ChatEngine(self.router)
        self.portal_agent = CourtPortalAgent()

        # Court-portal pause/resume coordination.
        self._portal_event = threading.Event()
        self._portal_decision = True

        # File pickers.
        self.evidence_picker = ft.FilePicker(on_result=self._on_evidence_picked)
        self.page.overlay.append(self.evidence_picker)

        self.content = ft.Column(expand=True, scroll=ft.ScrollMode.AUTO, spacing=12)
        self._build_shell()
        self.show_dashboard()

    # ======================================================================
    # Shell / navigation
    # ======================================================================
    def _build_shell(self) -> None:
        self.page.title = config.APP_TITLE
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.bgcolor = BG
        self.page.padding = 0

        nav_items = [
            ("dashboard", "Dashboard", self.show_dashboard),
            ("question_answer", "Intake", self.show_intake),
            ("timeline", "Discovery & Timeline", self.show_discovery),
            ("travel_explore", "Deep Research", self.show_research),
            ("chat", "Chat (@web)", self.show_chat),
            ("description", "Brief Builder", self.show_brief),
            ("alarm", "Procedural", self.show_procedural),
            ("dns", "Court Portal", self.show_portal),
            ("settings", "Settings", self.show_settings),
        ]
        self.nav_rail = ft.NavigationRail(
            selected_index=0,
            label_type=ft.NavigationRailLabelType.ALL,
            bgcolor=SURFACE,
            min_width=72,
            min_extended_width=200,
            extended=True,
            destinations=[
                ft.NavigationRailDestination(icon=icon, label=label)
                for icon, label, _ in nav_items
            ],
            on_change=lambda e: nav_items[e.control.selected_index][2](),
        )

        self.case_banner = ft.Text(
            "No case selected", color=MUTED, size=13, italic=True
        )
        self.provider_chip = ft.Text(self._provider_label(), color=ACCENT_2, size=12)

        header = ft.Container(
            content=ft.Row(
                [
                    ft.Text(config.APP_TITLE, size=20, weight=ft.FontWeight.BOLD, color=TEXT),
                    ft.Container(expand=True),
                    self.case_banner,
                    ft.Container(width=16),
                    self.provider_chip,
                ],
                alignment=ft.MainAxisAlignment.START,
            ),
            padding=16,
            bgcolor=SURFACE,
        )

        body = ft.Row(
            [
                self.nav_rail,
                ft.VerticalDivider(width=1, color=SURFACE_2),
                ft.Container(
                    content=self.content,
                    expand=True,
                    padding=20,
                ),
            ],
            expand=True,
        )

        self.page.add(ft.Column([header, body], expand=True, spacing=0))

    def _provider_label(self) -> str:
        try:
            return self.router.summary()
        except Exception:  # noqa: BLE001
            return "No models configured"

    # ======================================================================
    # Generic helpers
    # ======================================================================
    def run_bg(self, work: Callable[[], object], on_done: Optional[Callable] = None) -> None:
        """Run ``work`` on a daemon thread; call ``on_done(result, error)`` after."""

        def runner() -> None:
            try:
                result = work()
                if on_done:
                    on_done(result, None)
            except Exception as exc:  # noqa: BLE001
                if on_done:
                    on_done(None, exc)
                else:
                    self.snack(f"Error: {exc}", DANGER)

        threading.Thread(target=runner, daemon=True).start()

    def snack(self, message: str, color: str = SURFACE_2) -> None:
        self.page.snack_bar = ft.SnackBar(
            content=ft.Text(message, color=TEXT), bgcolor=color
        )
        self.page.snack_bar.open = True
        self.page.update()

    def _set_content(self, *controls: ft.Control) -> None:
        self.content.controls = list(controls)
        self.page.update()

    def _section_title(self, text: str, subtitle: str = "") -> ft.Control:
        items = [ft.Text(text, size=22, weight=ft.FontWeight.BOLD, color=TEXT)]
        if subtitle:
            items.append(ft.Text(subtitle, size=13, color=MUTED))
        return ft.Column(items, spacing=2)

    def _card(self, *controls: ft.Control) -> ft.Container:
        return ft.Container(
            content=ft.Column(list(controls), spacing=10),
            bgcolor=SURFACE,
            border_radius=12,
            padding=18,
        )

    def _require_case(self) -> bool:
        if self.case_id is None:
            self._set_content(
                self._section_title("No case selected"),
                ft.Text(
                    "Create or select a case on the Dashboard first.", color=MUTED
                ),
                ft.FilledButton("Go to Dashboard", on_click=lambda e: self.show_dashboard()),
            )
            return False
        return True

    def _new_log(self, height: int = 220) -> ft.ListView:
        return ft.ListView(expand=False, height=height, spacing=2, auto_scroll=True)

    def _log_to(self, log_view: ft.ListView) -> Callable[[str], None]:
        def _log(msg: str) -> None:
            log_view.controls.append(
                ft.Text(f"• {msg}", size=12, color=MUTED, selectable=True)
            )
            self.page.update()

        return _log

    # ======================================================================
    # Dashboard + case management
    # ======================================================================
    def show_dashboard(self) -> None:
        cases = db.list_cases()
        options = [ft.dropdown.Option(str(c["id"]), c["name"]) for c in cases]
        self.case_dropdown = ft.Dropdown(
            label="Active case",
            options=options,
            value=str(self.case_id) if self.case_id else None,
            on_change=self._on_case_selected,
            width=360,
            color=TEXT,
        )

        # New-case form.
        self.nc_name = ft.TextField(label="Case name", width=360, color=TEXT)
        self.nc_court = ft.TextField(label="Court", width=360, color=TEXT)
        self.nc_number = ft.TextField(label="Case number", width=360, color=TEXT)
        self.nc_judge = ft.TextField(label="Judge", width=360, color=TEXT)
        self.nc_jur = ft.TextField(label="Jurisdiction (e.g. California)", width=360, color=TEXT)
        self.nc_charges = ft.TextField(label="Charges / dispute", width=360, color=TEXT)

        deadlines_summary = self._deadline_summary()

        self._set_content(
            self._section_title(
                "Dashboard", "Manage workspaces and review upcoming deadlines."
            ),
            ft.Row(
                [
                    self._card(
                        ft.Text("Select a case", size=16, weight=ft.FontWeight.BOLD, color=TEXT),
                        self.case_dropdown,
                        ft.Row(
                            [
                                ft.OutlinedButton(
                                    "Delete case",
                                    icon="delete",
                                    on_click=self._delete_case,
                                ),
                            ]
                        ),
                    ),
                    self._card(
                        ft.Text("Create a new case", size=16, weight=ft.FontWeight.BOLD, color=TEXT),
                        self.nc_name,
                        self.nc_court,
                        self.nc_number,
                        self.nc_judge,
                        self.nc_jur,
                        self.nc_charges,
                        ft.FilledButton("Create case", icon="add", on_click=self._create_case),
                    ),
                ],
                wrap=True,
                spacing=16,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            self._card(
                ft.Text("Deadline Tickler", size=16, weight=ft.FontWeight.BOLD, color=TEXT),
                deadlines_summary,
            ),
        )

    def _deadline_summary(self) -> ft.Control:
        if self.case_id is None:
            return ft.Text("Select a case to see its deadlines.", color=MUTED)
        deadlines = db.list_deadlines(self.case_id)
        if not deadlines:
            return ft.Text("No deadlines yet. Add them in the Procedural tab.", color=MUTED)
        rows = []
        today = date.today()
        for d in deadlines:
            try:
                due = datetime.fromisoformat(d["due_date"]).date()
                days = (due - today).days
            except ValueError:
                days = None
            colour = OK
            if days is not None and days < 0:
                colour = DANGER
            elif days is not None and days <= 7:
                colour = WARN
            label = f"{d['due_date']}  ·  {d['title']}"
            if days is not None:
                label += f"  ({days} days)" if days >= 0 else f"  (OVERDUE {abs(days)}d)"
            rows.append(ft.Text(label, color=colour, size=13))
        return ft.Column(rows, spacing=4)

    def _on_case_selected(self, e: ft.ControlEvent) -> None:
        if e.control.value:
            self.case_id = int(e.control.value)
            case = db.get_case(self.case_id)
            self.case_banner.value = f"Case: {case['name']}" if case else "No case selected"
            self.page.update()
            self.show_dashboard()

    def _create_case(self, e: ft.ControlEvent) -> None:
        if not self.nc_name.value.strip():
            self.snack("Case name is required.", DANGER)
            return
        cid = db.create_case(
            name=self.nc_name.value.strip(),
            court=self.nc_court.value.strip(),
            case_number=self.nc_number.value.strip(),
            judge=self.nc_judge.value.strip(),
            jurisdiction=self.nc_jur.value.strip(),
            charges=self.nc_charges.value.strip(),
        )
        self.case_id = cid
        self.case_banner.value = f"Case: {self.nc_name.value.strip()}"
        self.snack("Case created.", OK)
        self.show_dashboard()

    def _delete_case(self, e: ft.ControlEvent) -> None:
        if self.case_id is None:
            self.snack("No case selected.", DANGER)
            return

        def confirm(_: ft.ControlEvent) -> None:
            db.delete_case(self.case_id)
            from rag import STORE

            STORE.reset_case(self.case_id)
            self.case_id = None
            self.case_banner.value = "No case selected"
            dlg.open = False
            self.page.update()
            self.snack("Case deleted.", OK)
            self.show_dashboard()

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Delete this case?"),
            content=ft.Text("This permanently removes all evidence, timeline, and chat for the case."),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: self._close_dialog(dlg)),
                ft.FilledButton("Delete", on_click=confirm),
            ],
        )
        self.page.dialog = dlg
        dlg.open = True
        self.page.update()

    def _close_dialog(self, dlg: ft.AlertDialog) -> None:
        dlg.open = False
        self.page.update()

    # ======================================================================
    # Intake interviewer
    # ======================================================================
    def show_intake(self) -> None:
        self.intake_fields: Dict[str, ft.TextField] = {}
        rows = []
        for key, question in self.intake.QUESTIONS:
            field = ft.TextField(
                label=question,
                color=TEXT,
                multiline=key == "narrative",
                min_lines=3 if key == "narrative" else 1,
            )
            self.intake_fields[key] = field
            rows.append(field)

        self.intake_output = ft.Text("", color=TEXT, selectable=True)

        self._set_content(
            self._section_title(
                "Intake Interviewer",
                "Answer these questions to establish a baseline case file.",
            ),
            self._card(*rows,
                       ft.FilledButton("Build baseline case", icon="auto_awesome",
                                       on_click=self._submit_intake)),
            self._card(ft.Text("Baseline summary", weight=ft.FontWeight.BOLD, color=TEXT),
                       self.intake_output),
        )

    def _submit_intake(self, e: ft.ControlEvent) -> None:
        answers = {k: f.value.strip() for k, f in self.intake_fields.items()}
        if not answers.get("name"):
            self.snack("At least your name is required.", DANGER)
            return

        # Create or update the case from intake.
        if self.case_id is None:
            self.case_id = db.create_case(
                name=answers.get("name") or "Untitled case",
                court=answers.get("court", ""),
                case_number=answers.get("case_number", ""),
                judge=answers.get("judge", ""),
                jurisdiction=answers.get("jurisdiction", ""),
                charges=answers.get("charges", ""),
            )
            self.case_banner.value = f"Case: {answers.get('name')}"
        else:
            db.update_case(
                self.case_id,
                court=answers.get("court", ""),
                case_number=answers.get("case_number", ""),
                judge=answers.get("judge", ""),
                jurisdiction=answers.get("jurisdiction", ""),
                charges=answers.get("charges", ""),
            )

        self.intake_output.value = "Working…"
        self.page.update()

        def work() -> str:
            summary = self.intake.summarize(answers)
            db.update_case(self.case_id, summary=summary)
            return summary

        def done(result, error) -> None:
            self.intake_output.value = result if result else f"Error: {error}"
            self.page.update()
            if not error:
                self.snack("Baseline case built.", OK)

        self.run_bg(work, done)

    # ======================================================================
    # Discovery & Timeline
    # ======================================================================
    def show_discovery(self) -> None:
        if not self._require_case():
            return
        self.discovery_log = self._new_log(160)
        self.timeline_container = ft.Column(spacing=0)
        self._render_timeline()

        self._set_content(
            self._section_title(
                "Discovery & Timeline Engine",
                "Ingest evidence (PDF, image, audio/video) and build an interactive timeline.",
            ),
            self._card(
                ft.Row(
                    [
                        ft.FilledButton(
                            "Add evidence file(s)",
                            icon="upload_file",
                            on_click=lambda e: self.evidence_picker.pick_files(
                                allow_multiple=True
                            ),
                        ),
                        ft.OutlinedButton(
                            "Run Inconsistency Matrix",
                            icon="rule",
                            on_click=self._run_inconsistency,
                        ),
                        ft.OutlinedButton(
                            "Clear timeline",
                            icon="delete_sweep",
                            on_click=self._clear_timeline,
                        ),
                    ],
                    wrap=True,
                ),
                self.discovery_log,
            ),
            self._card(
                ft.Text("Timeline", weight=ft.FontWeight.BOLD, color=TEXT),
                ft.Text("Rows in red are flagged contradictions.", size=12, color=MUTED),
                self.timeline_container,
            ),
        )

    def _render_timeline(self) -> None:
        events = db.list_timeline(self.case_id) if self.case_id else []
        if not events:
            self.timeline_container.controls = [
                ft.Text("No facts yet. Add evidence to populate the timeline.", color=MUTED)
            ]
            self.page.update()
            return
        header = ft.Row(
            [
                ft.Container(ft.Text("Date", color=MUTED, weight=ft.FontWeight.BOLD), width=110),
                ft.Container(ft.Text("Fact", color=MUTED, weight=ft.FontWeight.BOLD), expand=True),
                ft.Container(ft.Text("Source", color=MUTED, weight=ft.FontWeight.BOLD), width=160),
            ]
        )
        rows = [header, ft.Divider(color=SURFACE_2)]
        for e in events:
            colour = DANGER if e["inconsistency"] else TEXT
            fact_text = e["description"]
            if e["inconsistency"] and e["inconsistency_note"]:
                fact_text += f"\n⚠ {e['inconsistency_note']}"
            rows.append(
                ft.Row(
                    [
                        ft.Container(ft.Text(e["event_date"] or "n.d.", color=colour, size=12), width=110),
                        ft.Container(ft.Text(fact_text, color=colour, size=12, selectable=True), expand=True),
                        ft.Container(ft.Text(e["source_doc"] or "", color=MUTED, size=11), width=160),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.START,
                )
            )
            rows.append(ft.Divider(color=SURFACE_2, height=1))
        self.timeline_container.controls = rows
        self.page.update()

    def _on_evidence_picked(self, e: ft.FilePickerResultEvent) -> None:
        if not e.files or self.case_id is None:
            return
        log = self._log_to(self.discovery_log)
        files = [(f.path, f.name) for f in e.files if f.path]

        def work() -> int:
            count = 0
            for path, name in files:
                log(f"Ingesting {name}…")
                try:
                    summary = self.discovery.ingest_file(self.case_id, path, name, log=log)
                    log(
                        f"{name}: {summary['facts_added']} facts, "
                        f"{summary['char_count']} chars indexed."
                    )
                    count += 1
                except (IngestionError, LLMError) as exc:
                    log(f"{name}: {exc}")
            return count

        def done(result, error) -> None:
            if error:
                self.snack(f"Ingestion error: {error}", DANGER)
            else:
                self.snack(f"Ingested {result} file(s).", OK)
            self._render_timeline()

        self.run_bg(work, done)

    def _run_inconsistency(self, e: ft.ControlEvent) -> None:
        if not self._require_case():
            return
        log = self._log_to(self.discovery_log)

        def work() -> int:
            return self.discovery.analyze_inconsistencies(self.case_id, log=log)

        def done(result, error) -> None:
            if error:
                self.snack(f"Analysis error: {error}", DANGER)
            else:
                self.snack(f"Flagged {result} inconsistencies.", WARN if result else OK)
            self._render_timeline()

        self.run_bg(work, done)

    def _clear_timeline(self, e: ft.ControlEvent) -> None:
        if self.case_id is None:
            return
        db.clear_timeline(self.case_id)
        self._render_timeline()
        self.snack("Timeline cleared.", OK)

    # ======================================================================
    # Deep Research
    # ======================================================================
    def show_research(self) -> None:
        if not self._require_case():
            return
        case = db.get_case(self.case_id) or {}
        self.research_query = ft.TextField(
            label="Research question",
            color=TEXT,
            multiline=True,
            min_lines=2,
        )
        self.research_judge = ft.TextField(
            label="Assigned judge (optional)", value=case.get("judge", ""), color=TEXT
        )
        self.research_log = self._new_log(160)
        self.research_output = ft.Text("", color=TEXT, selectable=True)
        self.research_citations = ft.Column(spacing=4)
        self._last_research_memo = ""

        self._set_content(
            self._section_title(
                "Deep Research & Judge Analytics",
                f"Iterative CourtListener search (hard limit: {config.MAX_RESEARCH_ITERATIONS} loops).",
            ),
            self._card(
                self.research_query,
                self.research_judge,
                ft.FilledButton("Run research", icon="travel_explore", on_click=self._run_research),
                self.research_log,
            ),
            self._card(
                ft.Text("Research memo", weight=ft.FontWeight.BOLD, color=TEXT),
                self.research_output,
            ),
            self._card(
                ft.Text("Citations", weight=ft.FontWeight.BOLD, color=TEXT),
                self.research_citations,
            ),
        )

    def _run_research(self, e: ft.ControlEvent) -> None:
        question = self.research_query.value.strip()
        if not question:
            self.snack("Enter a research question.", DANGER)
            return
        log = self._log_to(self.research_log)
        self.research_output.value = "Researching…"
        self.page.update()

        def work() -> Dict:
            return self.research.run(
                question, judge=self.research_judge.value.strip(), log=log
            )

        def done(result, error) -> None:
            if error:
                self.research_output.value = f"Error: {error}"
                self.page.update()
                return
            self._last_research_memo = result["answer"]
            self.research_output.value = result["answer"]
            self.research_citations.controls = [
                ft.Row(
                    [
                        ft.Icon("link", color=ACCENT_2, size=14),
                        ft.TextButton(
                            c["caption"] or c["url"],
                            url=c["url"],
                            tooltip=c["url"],
                        ),
                    ]
                )
                for c in result["citations"]
            ] or [ft.Text("No authorities retrieved.", color=MUTED)]
            self.page.update()
            self.snack("Research complete.", OK)

        self.run_bg(work, done)

    # ======================================================================
    # Chat
    # ======================================================================
    def show_chat(self) -> None:
        if not self._require_case():
            return
        self.chat_list = ft.ListView(expand=True, spacing=10, auto_scroll=True, height=420)
        self._load_chat_history()
        self.chat_input = ft.TextField(
            hint_text="Ask about your case…  (prefix with @web for an internet lookup)",
            color=TEXT,
            expand=True,
            on_submit=self._send_chat,
        )
        self.decode_input = ft.TextField(
            label="Paste complex legal text to decode (Legalese Decoder)",
            color=TEXT,
            multiline=True,
            min_lines=2,
        )
        self.decode_output = ft.Text("", color=TEXT, selectable=True)

        self._set_content(
            self._section_title(
                "Chat", "RAG-grounded answers over your evidence. Use @web for OSINT context."
            ),
            self._card(
                self.chat_list,
                ft.Row(
                    [
                        self.chat_input,
                        ft.IconButton(icon="send", icon_color=ACCENT, on_click=self._send_chat),
                    ]
                ),
            ),
            self._card(
                ft.Text("Legalese Decoder", weight=ft.FontWeight.BOLD, color=TEXT),
                self.decode_input,
                ft.FilledButton("Translate to 4th-grade level", icon="translate",
                                on_click=self._decode_legalese),
                self.decode_output,
            ),
        )

    def _load_chat_history(self) -> None:
        self.chat_list.controls = []
        for m in db.list_messages(self.case_id):
            self.chat_list.controls.append(self._chat_bubble(m["role"], m["content"]))
        self.page.update()

    def _chat_bubble(self, role: str, content: str) -> ft.Control:
        is_user = role == "user"
        return ft.Row(
            [
                ft.Container(
                    content=ft.Text(content, color=TEXT, selectable=True),
                    bgcolor=ACCENT if is_user else SURFACE_2,
                    padding=12,
                    border_radius=12,
                    width=620,
                )
            ],
            alignment=ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START,
        )

    def _send_chat(self, e: ft.ControlEvent) -> None:
        text = self.chat_input.value.strip()
        if not text:
            return
        self.chat_input.value = ""
        db.add_message(self.case_id, "user", text)
        self.chat_list.controls.append(self._chat_bubble("user", text))
        thinking = self._chat_bubble("assistant", "…")
        self.chat_list.controls.append(thinking)
        self.page.update()

        def work() -> Dict:
            return self.chat.handle(self.case_id, text)

        def done(result, error) -> None:
            if error:
                answer = f"Error: {error}"
            else:
                answer = result["answer"]
                cites = result.get("citations") or []
                if cites:
                    answer += "\n\nSources:\n" + "\n".join(
                        f"- {c.get('title','')}: {c.get('url','')}" for c in cites
                    )
            db.add_message(self.case_id, "assistant", answer)
            self.chat_list.controls[-1] = self._chat_bubble("assistant", answer)
            self.page.update()

        self.run_bg(work, done)

    def _decode_legalese(self, e: ft.ControlEvent) -> None:
        text = self.decode_input.value.strip()
        if not text:
            self.snack("Paste some text to decode.", DANGER)
            return
        self.decode_output.value = "Translating…"
        self.page.update()

        def work() -> str:
            return self.decoder.decode(text)

        def done(result, error) -> None:
            self.decode_output.value = result if result else f"Error: {error}"
            self.page.update()

        self.run_bg(work, done)

    # ======================================================================
    # Brief Builder
    # ======================================================================
    def show_brief(self) -> None:
        if not self._require_case():
            return
        self.brief_type = ft.TextField(
            label="Motion type (e.g. Motion to Suppress Evidence)", color=TEXT
        )
        self.brief_instructions = ft.TextField(
            label="Specific instructions / arguments to include",
            color=TEXT,
            multiline=True,
            min_lines=2,
        )
        self.brief_use_research = ft.Checkbox(
            label="Include the last research memo", value=True
        )
        self.brief_redact = ft.Checkbox(label="Auto-redact PII before export", value=True)
        self.brief_minor_names = ft.TextField(
            label="Protected names to redact (comma-separated)", color=TEXT
        )
        self.brief_output = ft.TextField(
            label="Drafted motion (editable)",
            color=TEXT,
            multiline=True,
            min_lines=12,
        )

        self._set_content(
            self._section_title(
                "Brief Builder & Defense Drafter",
                "Synthesise the timeline + research into a court-ready pleading.",
            ),
            self._card(
                self.brief_type,
                self.brief_instructions,
                self.brief_use_research,
                ft.FilledButton("Draft motion", icon="description", on_click=self._draft_brief),
            ),
            self._card(
                self.brief_output,
                self.brief_redact,
                self.brief_minor_names,
                ft.Row(
                    [
                        ft.FilledButton(
                            "Export to pleading paper (.docx)",
                            icon="picture_as_pdf",
                            on_click=self._export_brief,
                        ),
                    ]
                ),
            ),
        )

    def _draft_brief(self, e: ft.ControlEvent) -> None:
        motion_type = self.brief_type.value.strip()
        if not motion_type:
            self.snack("Enter a motion type.", DANGER)
            return
        memo = getattr(self, "_last_research_memo", "") if self.brief_use_research.value else ""
        self.brief_output.value = "Drafting…"
        self.page.update()

        def work() -> str:
            return self.brief.draft(
                self.case_id,
                motion_type,
                instructions=self.brief_instructions.value.strip(),
                research_memo=memo,
            )

        def done(result, error) -> None:
            self.brief_output.value = result if result else f"Error: {error}"
            self.page.update()
            if not error:
                self.snack("Draft ready. Review before filing.", OK)

        self.run_bg(work, done)

    def _export_brief(self, e: ft.ControlEvent) -> None:
        body = self.brief_output.value.strip()
        if not body or body == "Drafting…":
            self.snack("Draft a motion first.", DANGER)
            return
        case = db.get_case(self.case_id) or {}
        notes: List[str] = []
        if self.brief_redact.value:
            minors = [n.strip() for n in (self.brief_minor_names.value or "").split(",")]
            body, notes = redact_pii(body, minor_names=minors)

        title = self.brief_type.value.strip() or "Motion"
        filename = f"{title.replace(' ', '_')}_{int(datetime.utcnow().timestamp())}.docx"
        out_path = str(config.EXPORT_DIR / filename)

        def work() -> str:
            return export_pleading_docx(
                out_path,
                party_name=case.get("name", ""),
                court=case.get("court", ""),
                case_number=case.get("case_number", ""),
                judge=case.get("judge", ""),
                title=title,
                body=body,
                defendant=case.get("name", ""),
            )

        def done(result, error) -> None:
            if error:
                self.snack(f"Export failed: {error}", DANGER)
                return
            draft_id = db.add_draft(self.case_id, title, body, result)
            msg = f"Exported to {result}"
            if notes:
                msg += "  (" + "; ".join(notes) + ")"
            self.snack(msg, OK)

        self.run_bg(work, done)

    # ======================================================================
    # Procedural (deadlines + cross-exam)
    # ======================================================================
    def show_procedural(self) -> None:
        if not self._require_case():
            return
        case = db.get_case(self.case_id) or {}
        self.dl_event = ft.TextField(label="Trigger event (e.g. Arraignment, Service of Complaint)", color=TEXT)
        self.dl_date = ft.TextField(
            label="Trigger date (YYYY-MM-DD)",
            value=date.today().isoformat(),
            color=TEXT,
        )
        self.dl_jur = ft.TextField(
            label="Jurisdiction", value=case.get("jurisdiction", ""), color=TEXT
        )
        self.deadline_list = ft.Column(spacing=4)
        self._render_deadlines()

        self.witness_field = ft.TextField(label="Witness name", color=TEXT)
        self.crossexam_output = ft.Text("", color=TEXT, selectable=True)

        self._set_content(
            self._section_title(
                "Procedural Guardrails",
                "Statutory deadline tickler and cross-examination prep.",
            ),
            self._card(
                ft.Text("Deadline calculator", weight=ft.FontWeight.BOLD, color=TEXT),
                self.dl_event,
                self.dl_date,
                self.dl_jur,
                ft.FilledButton("Compute deadlines", icon="event", on_click=self._compute_deadlines),
                ft.Text(
                    "Always verify computed dates against your local court rules.",
                    size=12, color=WARN,
                ),
            ),
            self._card(
                ft.Text("Tracked deadlines", weight=ft.FontWeight.BOLD, color=TEXT),
                self.deadline_list,
            ),
            self._card(
                ft.Text("Cross-Examination Prep Engine", weight=ft.FontWeight.BOLD, color=TEXT),
                self.witness_field,
                ft.FilledButton(
                    "Generate questions from contradictions",
                    icon="quiz",
                    on_click=self._cross_exam,
                ),
                self.crossexam_output,
            ),
        )

    def _render_deadlines(self) -> None:
        deadlines = db.list_deadlines(self.case_id)
        if not deadlines:
            self.deadline_list.controls = [ft.Text("No deadlines tracked yet.", color=MUTED)]
            self.page.update()
            return
        controls = []
        today = date.today()
        for d in deadlines:
            try:
                days = (datetime.fromisoformat(d["due_date"]).date() - today).days
            except ValueError:
                days = None
            colour = OK
            if days is not None and days < 0:
                colour = DANGER
            elif days is not None and days <= 7:
                colour = WARN
            controls.append(
                ft.Row(
                    [
                        ft.Text(d["due_date"], color=colour, width=110, size=12),
                        ft.Text(d["title"], color=TEXT, expand=True, size=12),
                        ft.Text(d["rule"] or "", color=MUTED, width=240, size=11),
                        ft.IconButton(
                            icon="check_circle",
                            icon_color=OK if d["status"] == "done" else MUTED,
                            tooltip="Mark done",
                            on_click=lambda e, did=d["id"]: self._mark_deadline(did),
                        ),
                        ft.IconButton(
                            icon="delete",
                            icon_color=MUTED,
                            on_click=lambda e, did=d["id"]: self._delete_deadline(did),
                        ),
                    ]
                )
            )
        self.deadline_list.controls = controls
        self.page.update()

    def _compute_deadlines(self, e: ft.ControlEvent) -> None:
        event = self.dl_event.value.strip()
        if not event:
            self.snack("Enter a trigger event.", DANGER)
            return

        def work() -> List:
            return self.procedural.compute_deadlines(
                self.case_id,
                event,
                self.dl_date.value.strip(),
                jurisdiction=self.dl_jur.value.strip(),
            )

        def done(result, error) -> None:
            if error:
                self.snack(f"Error: {error}", DANGER)
            else:
                self.snack(f"Added {len(result)} deadlines.", OK)
            self._render_deadlines()

        self.run_bg(work, done)

    def _mark_deadline(self, deadline_id: int) -> None:
        db.update_deadline_status(self.case_id, deadline_id, "done")
        self._render_deadlines()

    def _delete_deadline(self, deadline_id: int) -> None:
        db.delete_deadline(self.case_id, deadline_id)
        self._render_deadlines()

    def _cross_exam(self, e: ft.ControlEvent) -> None:
        witness = self.witness_field.value.strip()
        if not witness:
            self.snack("Enter a witness name.", DANGER)
            return
        self.crossexam_output.value = "Generating…"
        self.page.update()

        def work() -> str:
            return self.procedural.cross_exam_questions(self.case_id, witness)

        def done(result, error) -> None:
            self.crossexam_output.value = result if result else f"Error: {error}"
            self.page.update()

        self.run_bg(work, done)

    # ======================================================================
    # Court Portal automation
    # ======================================================================
    def show_portal(self) -> None:
        if not self._require_case():
            return
        creds = db.list_portal_credentials()
        self.portal_cred_dropdown = ft.Dropdown(
            label="Saved portal",
            options=[ft.dropdown.Option(str(c["id"]), f"{c['portal_name']} ({c['url']})") for c in creds],
            width=480,
            color=TEXT,
        )
        # Add-credential form.
        self.pc_name = ft.TextField(label="Portal name", color=TEXT, width=300)
        self.pc_url = ft.TextField(label="Login / search URL", color=TEXT, width=480)
        self.pc_user = ft.TextField(label="Username (optional)", color=TEXT, width=300)
        self.pc_pass = ft.TextField(label="Password (optional)", color=TEXT, width=300, password=True)

        self.portal_query = ft.TextField(label="Search query (case number / name)", color=TEXT, width=480)
        self.portal_log = self._new_log(220)

        self.portal_banner = ft.Container(visible=False, bgcolor=WARN, padding=12, border_radius=8)
        self.portal_resume_btn = ft.FilledButton(
            "I solved it — Resume", icon="play_arrow", on_click=self._portal_resume, visible=False
        )
        self.portal_abort_btn = ft.OutlinedButton(
            "Abort", icon="stop", on_click=self._portal_abort, visible=False
        )

        self._set_content(
            self._section_title(
                "Autonomous Court Portal Agent",
                "Logs into public case-search databases, downloads documents, and indexes them. "
                "It will pause and ask you to solve any CAPTCHA — it never bypasses one.",
            ),
            self._card(
                ft.Text("Saved credentials are stored locally only.", size=12, color=MUTED),
                self.portal_cred_dropdown,
                ft.Divider(color=SURFACE_2),
                ft.Text("Add a portal", weight=ft.FontWeight.BOLD, color=TEXT),
                ft.Row([self.pc_name, self.pc_user, self.pc_pass], wrap=True),
                self.pc_url,
                ft.OutlinedButton("Save portal", icon="save", on_click=self._save_portal_cred),
            ),
            self._card(
                self.portal_query,
                ft.Row(
                    [
                        ft.FilledButton("Start agent", icon="smart_toy", on_click=self._start_portal),
                        self.portal_resume_btn,
                        self.portal_abort_btn,
                    ]
                ),
                self.portal_banner,
                self.portal_log,
            ),
        )

    def _save_portal_cred(self, e: ft.ControlEvent) -> None:
        if not self.pc_name.value.strip() or not self.pc_url.value.strip():
            self.snack("Portal name and URL are required.", DANGER)
            return
        db.add_portal_credential(
            self.pc_name.value.strip(),
            self.pc_url.value.strip(),
            self.pc_user.value.strip(),
            self.pc_pass.value.strip(),
        )
        self.snack("Portal saved.", OK)
        self.show_portal()

    def _portal_alert(self, message: str) -> None:
        self.portal_banner.content = ft.Text(message, color="#1A1200")
        self.portal_banner.visible = True
        self.portal_resume_btn.visible = True
        self.portal_abort_btn.visible = True
        self.page.update()

    def _portal_wait_for_user(self) -> bool:
        self._portal_event.clear()
        self._portal_event.wait()  # blocks the worker thread, not the UI
        return self._portal_decision

    def _portal_resume(self, e: ft.ControlEvent) -> None:
        self._portal_decision = True
        self.portal_banner.visible = False
        self.portal_resume_btn.visible = False
        self.portal_abort_btn.visible = False
        self.page.update()
        self._portal_event.set()

    def _portal_abort(self, e: ft.ControlEvent) -> None:
        self._portal_decision = False
        self.portal_agent.abort()
        self.portal_banner.visible = False
        self.portal_resume_btn.visible = False
        self.portal_abort_btn.visible = False
        self.page.update()
        self._portal_event.set()

    def _start_portal(self, e: ft.ControlEvent) -> None:
        if not self.portal_cred_dropdown.value:
            self.snack("Select a saved portal.", DANGER)
            return
        cred = db.get_portal_credential(int(self.portal_cred_dropdown.value))
        if not cred:
            self.snack("Credential not found.", DANGER)
            return
        query = self.portal_query.value.strip() or (db.get_case(self.case_id) or {}).get("case_number", "")
        log = self._log_to(self.portal_log)
        self.portal_agent = CourtPortalAgent()  # fresh agent per run

        def work() -> Dict:
            return self.portal_agent.run(
                self.case_id,
                cred,
                query,
                log=log,
                alert=self._portal_alert,
                wait_for_user=self._portal_wait_for_user,
            )

        def done(result, error) -> None:
            if error:
                self.snack(f"Portal agent error: {error}", DANGER)
                return
            self.snack(
                f"Downloaded {result['downloaded']}, indexed {result['ingested']} document(s).",
                OK,
            )

        log("Launching browser agent…")
        self.run_bg(work, done)

    # ======================================================================
    # Settings
    # ======================================================================
    def show_settings(self) -> None:
        # Build a provider+model selector pair for each task tier.
        self.tier_controls: Dict[str, Dict[str, ft.Dropdown]] = {}
        tier_cards: List[ft.Control] = []
        for tier in (config.TIER_EXTRACTION, config.TIER_MEDIUM, config.TIER_HEAVY):
            def_provider, def_model = config.TIER_DEFAULTS[tier]
            pkey, mkey = config.tier_setting_keys(tier)
            provider = db.get_setting(pkey, def_provider)
            if provider not in config.PROVIDERS:
                provider = def_provider
            model = db.get_setting(mkey, def_model)

            model_dd = ft.Dropdown(
                label="Model",
                value=model,
                options=[ft.dropdown.Option(m) for m in config.PROVIDERS[provider]["models"]],
                width=320,
                color=TEXT,
            )
            provider_dd = ft.Dropdown(
                label="Provider",
                value=provider,
                options=[ft.dropdown.Option(k, v["label"]) for k, v in config.PROVIDERS.items()],
                width=240,
                color=TEXT,
                on_change=lambda e, t=tier: self._on_tier_provider_change(t),
            )
            self.tier_controls[tier] = {"provider": provider_dd, "model": model_dd}
            tier_cards.append(
                self._card(
                    ft.Text(config.TIER_LABELS[tier], weight=ft.FontWeight.BOLD, color=TEXT),
                    ft.Row([provider_dd, model_dd], wrap=True),
                )
            )

        self.set_anthropic = ft.TextField(
            label="Anthropic API key", value=db.get_setting(config.SETTING_ANTHROPIC_KEY),
            password=True, can_reveal_password=True, color=TEXT, width=480,
        )
        self.set_openai = ft.TextField(
            label="OpenAI API key", value=db.get_setting(config.SETTING_OPENAI_KEY),
            password=True, can_reveal_password=True, color=TEXT, width=480,
        )
        self.set_gemini = ft.TextField(
            label="Gemini API key (AI Studio backend)", value=db.get_setting(config.SETTING_GEMINI_KEY),
            password=True, can_reveal_password=True, color=TEXT, width=480,
        )
        self.set_gemini_backend = ft.Dropdown(
            label="Gemini backend",
            value=db.get_setting(config.SETTING_GEMINI_BACKEND, "api_key"),
            options=[
                ft.dropdown.Option("api_key", "Google AI Studio (API key)"),
                ft.dropdown.Option("vertex", "Vertex AI (project + ADC)"),
            ],
            width=320,
            color=TEXT,
        )
        self.set_vertex_project = ft.TextField(
            label="Vertex GCP project ID", value=db.get_setting(config.SETTING_VERTEX_PROJECT),
            color=TEXT, width=320,
        )
        self.set_vertex_location = ft.TextField(
            label="Vertex region", value=db.get_setting(config.SETTING_VERTEX_LOCATION, "us-central1"),
            color=TEXT, width=200,
        )
        self.set_tavily = ft.TextField(
            label="Tavily API key (web search; optional)", value=db.get_setting(config.SETTING_TAVILY_KEY),
            password=True, can_reveal_password=True, color=TEXT, width=480,
        )
        self.set_deepgram = ft.TextField(
            label="Deepgram API key (transcription; optional)",
            value=db.get_setting(config.SETTING_DEEPGRAM_KEY),
            password=True, can_reveal_password=True, color=TEXT, width=480,
        )

        self._set_content(
            self._section_title(
                "Settings",
                "Route each task to the model it does best. Keys are stored locally only.",
            ),
            self._card(
                ft.Text("Model routing by task tier", weight=ft.FontWeight.BOLD, color=TEXT),
                ft.Text(
                    "Defaults: extraction → Gemini 3.5 Flash · medium → Sonnet 4.6 "
                    "(Opus 4.6 / Gemini 3.1 Pro also good here) · heavy → Opus 4.8.",
                    size=12, color=MUTED,
                ),
                *tier_cards,
            ),
            self._card(
                ft.Text("Cloud LLM keys", weight=ft.FontWeight.BOLD, color=TEXT),
                self.set_anthropic,
                self.set_openai,
                self.set_gemini,
                ft.Divider(color=SURFACE_2),
                ft.Text("Gemini backend", weight=ft.FontWeight.BOLD, color=TEXT),
                ft.Text(
                    "Vertex AI uses Application Default Credentials — run "
                    "'gcloud auth application-default login' (or attach a service "
                    "account) and fill in the project/region. No API key needed.",
                    size=12, color=MUTED,
                ),
                self.set_gemini_backend,
                ft.Row([self.set_vertex_project, self.set_vertex_location], wrap=True),
            ),
            self._card(
                ft.Text("Tool keys", weight=ft.FontWeight.BOLD, color=TEXT),
                self.set_tavily,
                self.set_deepgram,
            ),
            ft.FilledButton("Save settings", icon="save", on_click=self._save_settings),
        )

    def _on_tier_provider_change(self, tier: str) -> None:
        controls = self.tier_controls[tier]
        provider = controls["provider"].value
        models = config.PROVIDERS[provider]["models"]
        controls["model"].options = [ft.dropdown.Option(m) for m in models]
        controls["model"].value = config.PROVIDERS[provider]["default_model"]
        self.page.update()

    def _save_settings(self, e: ft.ControlEvent) -> None:
        for tier, controls in self.tier_controls.items():
            pkey, mkey = config.tier_setting_keys(tier)
            db.set_setting(pkey, controls["provider"].value)
            db.set_setting(mkey, controls["model"].value)
        db.set_setting(config.SETTING_ANTHROPIC_KEY, self.set_anthropic.value.strip())
        db.set_setting(config.SETTING_OPENAI_KEY, self.set_openai.value.strip())
        db.set_setting(config.SETTING_GEMINI_KEY, self.set_gemini.value.strip())
        db.set_setting(config.SETTING_GEMINI_BACKEND, self.set_gemini_backend.value)
        db.set_setting(config.SETTING_VERTEX_PROJECT, self.set_vertex_project.value.strip())
        db.set_setting(config.SETTING_VERTEX_LOCATION, self.set_vertex_location.value.strip() or "us-central1")
        db.set_setting(config.SETTING_TAVILY_KEY, self.set_tavily.value.strip())
        db.set_setting(config.SETTING_DEEPGRAM_KEY, self.set_deepgram.value.strip())
        self.router.reload()
        self.provider_chip.value = self._provider_label()
        self.page.update()
        self.snack("Settings saved.", OK)
