# CourtListener Operational Protocols — Case Law & Judge Insights

**Scope:** How *Pro Se Legal Intelligence* (`ProSeLegalIntelligence` v1.0.0) gathers case law and
judge insights from CourtListener with fiduciary-grade grounding: **zero fabricated citations,
every assertion traceable to a verifiable source.**

This document is wired to the existing code:
- `agents.py` → `DeepResearchAgent` (lines 434–575): `SEARCH_URL` (l.437), `_search` (l.442),
  `run` (l.462), `_decide_next` (l.518), `_synthesize` (l.547).
- `config.py`: `MAX_RESEARCH_ITERATIONS = 5` (l.169), `MAX_API_RETRIES = 4` (l.174),
  `RETRY_BASE_DELAY = 2.0` (l.175), `HTTP_TIMEOUT_SECONDS = 60` (l.173),
  tiers `TIER_EXTRACTION/MEDIUM/HEAVY` (l.131–139), `CASE_FIELD_KEYS` (l.214).
- `llm_router.py`: `LLMRouter.complete` (l.125), `complete_json` (l.140), `resolve` (l.105),
  `_with_retries` (l.56). Medium tier = Claude Sonnet 4.6; Heavy tier = Claude Opus 4.8;
  Extraction tier = Gemini 3.5 Flash.

---

## 0. What the existing `DeepResearchAgent` does today (baseline)

`_search` (agents.py l.442) issues an **anonymous** `GET` to
`https://www.courtlistener.com/api/rest/v4/search/` with `params={"q": query, "order_by": "score desc"}`
and a `User-Agent` header only. It takes `resp.json().get("results", [])[:8]`.

`run` reads exactly these result fields (l.486–495):
`caseName` / `caseNameShort`, `court`, `dateFiled`, `absolute_url` (prefixed with the host),
and `snippet` (truncated to 400 chars). Citations are de-duplicated by URL (l.510–516).

`_decide_next` (l.518, **medium** tier) returns `{done, reasoning, next_query}`; the loop stops on
`done`, on hitting `MAX_RESEARCH_ITERATIONS`, or when the query stops changing.
`_synthesize` (l.547, **medium** tier via `complete`) writes the memo from up to 20 collected
items using only `caption/court/date/snippet/url`.

**Two structural defects this protocol must fix:**
1. **No auth.** As of the May 2026 changes, CourtListener v4 endpoints **require a token; anonymous
   requests now receive `401 Unauthorized`** (and rate limits were cut sharply). The current
   anonymous `_search` will break or be throttled to near-zero. A token is now mandatory.
2. **Snippet-only grounding.** The agent only ever sees the 400-char search `snippet`, never full
   opinion text. It cannot pin-cite, cannot tell holding from dicta, and never verifies that a cited
   case still exists as authority. This is the core grounding gap.

---

## 1. Endpoint Reference Table (v4)

Base: `https://www.courtlistener.com/api/rest/v4/`. All require `Authorization: Token <key>`.

| Purpose | Endpoint | Key request params | Key response fields used |
|---|---|---|---|
| Unified search | `search/` | `q`, `type` (`o`,`r`,`rd`,`d`,`p`,`oa`), `court`, `filed_after`/`filed_before`, `precedential_status`, `order_by`, `cursor` | `caseName`, `court`, `dateFiled`, `absolute_url`, `snippet`, `citation`, `cluster_id`, `status` |
| Case-law cluster (the "decision") | `clusters/{id}/` | `fields`, `omit` | `case_name`, `date_filed`, `precedential_status`, `citations`, `sub_opinions` (→ opinion URLs), `docket` |
| Full opinion text | `opinions/{id}/` | `fields`, `omit` | `html_with_citations` (**preferred** full text), `plain_text` (fallback), `author_id`, `cluster`, `type` |
| Citation network (forward/back) | `opinions-cited/` | filter by `citing_opinion`, `cited_opinion` | `id`, `depth`, `citing_opinion_id`, `cited_opinion_id` |
| **Citation verify / hallucination guard** | `citation-lookup/` | `POST` `text=` (≤ ~64,000 chars / ~50 pp; ≤ 250 cites/request) | per-cite `status` (200 found / 404 not found / 429 beyond cap), `normalized_citations`, `clusters[]` |
| Dockets (PACER/RECAP linkage) | `dockets/{id}/` | | `court`, `case_name`, `date_filed`, `clusters` |
| Judge / person | `people/` , `people/{id}/` | `name_last`, `fjc_id` | bio, links to positions/educations/political-affiliations |
| Judicial positions | `positions/` | filter by `person` | `position_type`, `court`, `date_start`, `date_termination`, `appointer` |
| Political affiliation | `political-affiliations/` | filter by `person` | party, source, `date_start`, `date_end` |
| Education | `educations/` | filter by `person` | school, degree, year |
| Financial disclosures | `financial-disclosures/` (+ `investments`, `gifts`, `debts`, `agreements`, etc.) | filter by `person` | disclosure documents and itemized holdings |

Link chain for case law: **search result → `cluster_id` → `clusters/{id}/` → `sub_opinions` →
`opinions/{id}/` (full text)**. Link a judge to authored opinions via the opinion `author_id`
(equals a `people` id) or `search/?type=o&author=<person_id>`.

Use `fields=` / `omit=` on `clusters/` and `opinions/` to trim payloads (e.g.
`omit=plain_text` when you only need `html_with_citations`).

---

## 2. Auth, Rate Limits & Pagination (constraints that shape "efficient")

- **Token auth (mandatory).** Header: `Authorization: Token <your-token>`. Store under a new
  setting (e.g. `SETTING_COURTLISTENER_KEY = "courtlistener_api_key"`) alongside the other keys in
  `config.py` (l.148–152). Without it, v4 returns **401**.
- **Rate limits (post-May-2026 defaults).** New accounts default to roughly **5 req/min, 50 req/hr,
  125 req/day**. Accounts that historically made ≥1,000 requests are grandfathered at the old
  **5,000 req/hr**. **Design for the low tier.** The current `[:8]` results × 5 iterations + planned
  full-text pulls can blow the 50/hr ceiling fast.
- **Citation-lookup throttle:** ≤ **60 valid citations/min** and ≤ **250 citations/request**;
  citations past 250 come back with `status: 429` (identified, not looked up).
- **Pagination:** v4 uses **cursor-based** pagination (`cursor` param + `next`/`previous` in the
  response). Do not assume offset paging. For this app, prefer narrowing the query over deep paging.
- **Retries:** route every HTTP call through the same backoff discipline as `_with_retries`
  (`MAX_API_RETRIES = 4`, `RETRY_BASE_DELAY = 2.0`, exp backoff), treating `401` as **fatal/not
  retryable** (missing key) and `429`/5xx as retryable.

---

## 3. Is there a CourtListener MCP server? (Yes.)

**Yes — an official one exists.** Free Law Project shipped an official MCP server (announced
2026‑05‑12) hosted at **`https://mcp.courtlistener.com/`**, listed in Anthropic's connector
directory, usable from Claude apps/Cursor/etc. It is a thin proxy that forwards your CourtListener
token per call (OAuth 2.0 with Dynamic Client Registration for the hosted server; or self-host via
`pip install "courtlistener-api-client[mcp]"` → `courtlistener-mcp`).

Official MCP tools: `search`, `get_endpoint_item`, `get_more_results`, `get_counts`,
`call_endpoint`, `get_endpoint_schema`, `get_choices`, `extract_citations`, `analyze_citations`,
`resume_citation_analysis`, plus alert/subscription tools.

**Design decision for this app:** *Pro Se Legal Intelligence* is a self-contained Python
desktop/mobile app that runs its own LLM router and does direct `requests` calls — it is **not** an
MCP client. We therefore **design around the REST API the app already uses** (Section 1), which the
MCP server itself proxies, so we lose nothing. (Optional future: the `analyze_citations` /
`extract_citations` MCP tools mirror the `citation-lookup` REST endpoint we adopt below.)

---

## 4. Protocol Set A — Most-Efficient Retrieval

Goal: maximum coverage, minimum API calls, within ~50 req/hr.

**A1. Authenticate once.** Read the token via `db.get_setting(config.SETTING_COURTLISTENER_KEY)`.
If empty, surface a clear "CourtListener token required" message (mirroring `LLMRouter.complete`'s
missing-key error style, l.133) and stop — do **not** fall back to anonymous (it now 401s).

**A2. Build the seed query (medium tier).** Before iterating, have the **medium** tier convert the
user's natural-language question into a focused boolean query plus structured filters
(`court`, `filed_after`, `precedential_status=Published`), returned as JSON. This front-loads
precision instead of burning iterations on it. Always set `type=o` for case-law research.

**A3. Search, capturing IDs.** Call `search/?type=o&q=...&order_by="score desc"` with filters and
the token. **Capture `cluster_id` and `citation` in addition to today's fields** (caption/court/
date/url/snippet). Keep `[:8]` per page; do **not** auto-page — refine instead (A5).

**A4. Pull full text only for top candidates.** After the *final* search iteration, rank collected
results (medium tier) and fetch full text for only the **top 3–5** clusters:
`clusters/{cluster_id}/` → first/lead `sub_opinions` → `opinions/{id}/?fields=html_with_citations,
plain_text,author_id,type`. Prefer `html_with_citations`; fall back to `plain_text`. This bounds the
expensive calls: ≤ ~5 cluster + ≤ ~5 opinion fetches per research run.

**A5. Refine vs. stop (medium tier, reuse `_decide_next`).** Keep the bounded loop
(`MAX_RESEARCH_ITERATIONS = 5`). Stop when `done` is true, the query stops changing, or coverage is
adequate. Add a hard **API-call budget** counter (e.g. 20 calls/run) so the loop also stops on
budget exhaustion — protects the 50/hr ceiling.

**A6. Expand via citation network (cheap, high-value).** For each relied-upon opinion, optionally
call `opinions-cited/` to pull its **backward citations** (authorities it relies on) and
**forward citations** (later opinions citing it). Forward citations are reused in Protocol B as the
"subsequent treatment" signal. Cap expansion to the top 1–2 anchor opinions.

**A7. Cache within and across runs.** Cache by stable id: `cluster/{id}`, `opinion/{id}`,
`citation-lookup(normalized cite)`. Opinions are immutable, so cache them for the case's lifetime in
the local store (`DATA_ROOT`). This collapses repeated runs on the same issue to near-zero new
calls and is the single biggest lever against the new low rate limits.

**A8. Tier mapping.** Query construction, ranking, refine decisions, treatment classification →
**medium** (Sonnet 4.6). Final memo synthesis stays **medium** for routine research; escalate to
**heavy** (Opus 4.8) only when the memo feeds `BriefBuilder.draft` (already heavy, agents.py l.714).
Extraction tier is not used here (no OCR).

---

## 5. Protocol Set B — Fiduciary-Grade Grounding

Every rule below prevents hallucinated or misleading law. Each rule marks whether it is enforced
**[CODE]** (programmatic, deterministic) or **[LLM]** (model judgment under instruction). Prefer
[CODE] wherever a fact is checkable.

**B1. Cite only what was actually retrieved. [CODE]** A citation may appear in the memo only if it
maps to a `cluster_id`/`opinion id` in the run's `collected` set with a real CourtListener
`absolute_url`. After synthesis, **parse the memo and drop/flag any case name or reporter cite not
present in `collected`** (regex-extract citations, intersect with retrieved set). Anything unmatched
is removed and listed under "Unverified — excluded."

**B2. Verify every citation with `citation-lookup`. [CODE]** Before returning, POST the full memo
text to `citation-lookup/` (`text=`, ≤250 cites). For each returned citation:
`status 200` → verified (attach the matched `clusters[]` id/url); `status 404` → **citation does not
exist in CourtListener → strip it from the memo and log it**; `status 429` → over the per-request
cap, re-submit the remainder. This is the hallucination guardrail and it is deterministic.

**B3. Require pin-point support text. [LLM, fed CODE-supplied text]** No assertion about a case may
be written unless it is backed by a quoted span from that opinion's **full text** (`html_with_
citations`/`plain_text` fetched in A4). The synthesis prompt must require, per cited proposition, a
short verbatim quote + the opinion URL. Propositions without retrievable support text are recast as
"not established by retrieved authority."

**B4. Distinguish holding vs. dicta. [LLM]** When stating what a case "holds," the medium tier must
classify the supporting span as **holding** (necessary to the judgment) or **dicta** (incidental)
and label it in the memo. If it cannot tell from the retrieved text, it says so rather than
asserting a holding.

**B5. Mandatory subsequent-treatment ("good law") check. [CODE gather + LLM read].** Before *relying*
on any case, gather its **forward citations** via `opinions-cited/?cited_opinion={id}` (A6). The
medium tier then scans the citing opinions' snippets/text for negative-treatment language
(*overruled, abrogated, reversed, superseded, declined to follow, criticized*). The memo must state
the treatment status or explicitly note **"no negative treatment detected in retrieved citing
opinions — not a conclusive good-law determination"** (see B-LIMIT).

**B6. Jurisdiction-applicability check. [LLM, CODE-assisted].** Compare each authority's `court`
against the case's `jurisdiction`/`court` (from `CASE_FIELD_KEYS`, config l.214). Label each cite as
**binding**, **persuasive**, or **out-of-jurisdiction**, and never present persuasive authority as
controlling.

**B7. Honest "insufficient authority." [LLM, hard rule].** If retrieval yields no on-point,
verified, good-law authority, the memo must say **"No sufficient authority was found in
CourtListener for this proposition"** — never fill the gap with a guessed or remembered case. This
extends the existing "Be candid about gaps. Do not fabricate citations." instruction
(agents.py l.567–569) into an enforced output state.

**B8. Judge insights are facts, not predictions. [CODE gather + LLM].** Judge analytics pull only
from `people/`, `positions/`, `political-affiliations/`, `educations/`, `financial-disclosures/`,
and the judge's authored opinions (`author_id` / `search/?type=o&author=`). Report tenure,
appointer, and observed patterns **with citations to the underlying opinions**; never state
predicted rulings as fact. Replaces today's snippet-only judge clause (agents.py l.560–565).

**B9. Provenance on every citation object. [CODE].** Each citation in the returned `citations` list
must carry: `cluster_id`, `opinion_id`, `absolute_url`, `precedential_status`, `treatment_status`
(B5), and `applicability` (B6). A citation missing `cluster_id`/`opinion_id` is downgraded to
"unverified" and excluded from the relied-upon set.

### B-LIMIT — The single most important honesty disclosure

**CourtListener has NO Shepard's / KeyCite equivalent.** The `citation-lookup` API verifies only
that *a citation exists and resolves to a case in the database* — it explicitly does **not** perform
"good law" analysis, Shepardizing, or treatment classification (overruled/criticized/superseded).
The only treatment signal available is the **raw citation network** (`opinions-cited/`): you can see
*that* later cases cite an opinion, and read those opinions, but CourtListener does **not** label the
*nature* of the treatment. Therefore the app's "good law" check in B5 is a **best-effort,
text-derived heuristic, not an authoritative citator result.** Every memo must carry a standing
disclaimer to that effect, and the app must never represent a case as "still good law" with
citator-grade certainty. (This is legal *information*, not advice — consistent with the app's
existing chat disclaimer, agents.py l.882–883.)

---

## 6. Concrete changes recommended for `DeepResearchAgent`

1. **Add the token (config + router).** New `config.SETTING_COURTLISTENER_KEY`; in `_search`
   (agents.py l.442) add `headers["Authorization"] = f"Token {key}"`. Treat `401` as a fatal,
   non-retryable `LLMError` (distinct from the `429` path at l.452).
2. **Capture IDs in `run` (l.486–495).** Add `cluster_id` and `citation` to each collected dict so
   B1/B9 can verify and link to full text.
3. **Add `_fetch_full_text(cluster_id)`** — `clusters/{id}/` → `sub_opinions` → `opinions/{id}/
   ?fields=html_with_citations,plain_text,author_id,type`; cache by id (A4/A7). Call only for the
   top 3–5 ranked clusters.
4. **Add `_verify_citations(memo_text)`** — POST to `citation-lookup/`; strip 404s, attach 200
   matches, resubmit past-250 remainder (B2). Deterministic, runs after `_synthesize`.
5. **Add `_treatment(opinion_id)`** — `opinions-cited/?cited_opinion={id}`; collect forward
   citations; medium-tier scan for negative-treatment language (B5). Annotate each citation with
   `treatment_status` + the B-LIMIT caveat.
6. **Add `_judge_profile(name)`** — query `people/` → `positions/` / `political-affiliations/` /
   `educations/` / `financial-disclosures/` + authored opinions; feed structured facts into the
   judge section (B8), replacing the snippet-only clause at l.560–565.
7. **Harden `_synthesize` (l.547).** Pass **full opinion text** (not just snippets) for the top
   candidates; require per-proposition verbatim quote + URL (B3), holding/dicta labels (B4),
   binding/persuasive labels (B6), and an enforced "insufficient authority" state (B7). Keep memo
   on **medium** tier; escalate to **heavy** only when feeding `BriefBuilder`.
8. **Add an API-call budget counter** to `run` (A5) so the loop also stops on budget exhaustion,
   protecting the post-May-2026 ~50 req/hr ceiling.
9. **Persistent opinion/cite cache** under `DATA_ROOT` (A7) — opinions are immutable; cache for the
   case lifetime to make repeat runs nearly free.

---

## 7. Sources

- REST API overview / v4 (auth, rate limits, cursor paging): https://www.courtlistener.com/help/api/rest/ , https://wiki.free.law/c/courtlistener/help/api/rest/v4/overview
- Rate-limit change (May 2026; 5/min·50/hr·125/day default, 5,000/hr grandfathered): https://github.com/freelawproject/courtlistener/issues/7200 , https://free.law/2026/05/07/api-included-in-memberships/
- Search API (`type`, fields, filters, refinement): https://www.courtlistener.com/help/api/rest/search/ , https://www.courtlistener.com/help/api/rest/v3/search/
- Case Law APIs (clusters, opinions, `html_with_citations`/`plain_text`, opinions-cited network): https://www.courtlistener.com/help/api/rest/case-law/
- Citation-lookup / verification API (POST `text`, ≤250/req, 60 valid/min, 200/404/429, clusters[]): https://www.courtlistener.com/help/api/rest/v4/citation-lookup/ , https://free.law/2024/04/16/citation-lookup-api/
- Judge / people / positions / political-affiliations / educations / financial-disclosures: https://www.courtlistener.com/help/api/rest/judges/ , https://www.courtlistener.com/help/api/rest/v3/judges/ , https://www.courtlistener.com/help/api/rest/v3/financial-disclosures/
- Official CourtListener MCP server (hosted at mcp.courtlistener.com; OAuth; tool list): https://free.law/2026/05/12/courtlistener-is-now-available-inside-claude/ , https://wiki.free.law/c/courtlistener/help/api/mcp/model-context-protocol-mcp-server-for-agentic-access , https://github.com/freelawproject/courtlistener-api-client/blob/main/MCP_README.md
