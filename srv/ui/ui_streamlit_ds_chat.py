# ui/ui_streamlit_ds_chat.py - ChatGPT-Style UI for T21 Research Assistant
import os
import base64
import json
import textwrap
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import re
import smtplib
import html as html_lib
import urllib.parse
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from dotenv import load_dotenv
_OllamaClient = None
try:
    from ollama import Client as _OllamaClient
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

# Load .env (works both locally and inside Docker where /app/.env is mounted)
_ENV_PATH = Path(__file__).parent.parent / ".env"  # srv/.env → /app/.env in Docker
load_dotenv(_ENV_PATH, override=False)  # override=False: Docker-injected vars take priority

# -------------------- Config --------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8698")
DEFAULT_TOPK = int(os.getenv("DEFAULT_TOPK", "6"))
DEFAULT_INCLUDE_HITS = os.getenv("DEFAULT_INCLUDE_HITS", "false").lower() == "true"

# Single source of truth for the app version + copyright line shown in the sidebar
# footer and the fixed bottom bar -- previously duplicated independently in both
# places, which let them drift (fixed as part of the v7 rebuild).
APP_VERSION_LABEL = "Version 7.0"
COPYRIGHT_LABEL = "&copy; 2026 Costello Lab"

# -------------------- LLM Judge Config --------------------
_JUDGE_OLLAMA_BASE_URL = os.getenv("OLLAMA_REMOTE_BASE_URL", "https://ollama.com")
_JUDGE_OLLAMA_API_KEY  = os.getenv("OLLAMA_API_KEY", "")
_JUDGE_MODEL           = "gpt-oss:120b-cloud"   # scoring
_IMPROVE_MODEL         = os.getenv("NEMOTRON_CHAT_MODEL", "nemotron-3-super")  # rewrite

# -------------------- Logo --------------------
_LOGO_PATH = Path(__file__).parent / "images" / "Logo_T21_research_Assistant.png"
_LOGO_B64 = ""
if _LOGO_PATH.exists():
    _LOGO_B64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()

_COSTELLO_LOGO_PATH = Path(__file__).parent / "images" / "Costello_Lab_logo.png"
_COSTELLO_LOGO_B64 = ""
if _COSTELLO_LOGO_PATH.exists():
    _COSTELLO_LOGO_B64 = base64.b64encode(_COSTELLO_LOGO_PATH.read_bytes()).decode()

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="T21 Research Assistant",
    page_icon=str(_LOGO_PATH) if _LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Session State Initialization --------------------
if "conversations" not in st.session_state:
    # List of conversations: [{id, title, messages: [{role, content, data}]}]
    st.session_state.conversations = []
if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# -------------------- LLM Judge Helper --------------------
_JUDGE_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert biomedical judge evaluating an AI-generated answer about Down syndrome (Trisomy 21).
Score the response independently on each of the three criteria below using a scale of 1–5.
Do NOT compare responses against each other — evaluate each criterion on its own merits.

Scoring scale:
  1 = Very Poor   2 = Poor   3 = Fair   4 = Good   5 = Excellent

Criteria:
  1. Accuracy        — Is the information factually correct relative to the ground truth?
  2. Informativeness — Does it provide relevant, useful, and specific information?
  3. Completeness    — Is the answer comprehensive and complete based on current knowledge?

Return ONLY a valid JSON object (no markdown fences, no extra text):
{
  "accuracy": <1-5>,
  "informativeness": <1-5>,
  "completeness": <1-5>,
  "justification": "<2-3 sentence explanation of the scores>"
}
""")


def _llm_judge_score(question: str, summary_text: str) -> dict | None:
    """Call the LLM judge and return scores dict, or None on failure."""
    if not _OLLAMA_AVAILABLE:
        return None
    if not _JUDGE_OLLAMA_API_KEY:
        return None
    user_prompt = f"QUESTION:\n{question}\n\nRESPONSE TO EVALUATE:\n{summary_text}"
    try:
        client = _OllamaClient(
            host=_JUDGE_OLLAMA_BASE_URL,
            headers={"Authorization": f"Bearer {_JUDGE_OLLAMA_API_KEY}"}
        )
        resp = client.chat(
            model=_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": 0.0},
        )
        raw = resp["message"]["content"]
        raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        parsed = json.loads(raw)
        required = {"accuracy", "informativeness", "completeness", "justification"}
        if not required.issubset(parsed.keys()):
            return None
        for key in ("accuracy", "informativeness", "completeness"):
            v = int(parsed[key])
            if v < 1 or v > 5:
                return None
            parsed[key] = v
        return parsed
    except Exception:
        return None


def _llm_deep_dive_summary(
    question: str,
    original_summary: str,
    scores: dict,
    raw_evidence: list | None = None,
) -> str | None:
    """Generate a detailed, unlimited-length Deep Dive synthesis with full citations."""
    if not _OLLAMA_AVAILABLE or not _JUDGE_OLLAMA_API_KEY:
        return None

    judge_notes = scores.get("justification", "").strip()

    # Build a compact citation catalogue from retrieved evidence so the LLM
    # can cite beyond the handful of papers in the original summary.
    _cite_catalogue = ""
    if raw_evidence:
        _seen_ev: dict = {}
        _catalogue_lines = []
        for _ev in raw_evidence:
            _pmcid   = (_ev.get("pmcid") or "").strip()
            _section = (_ev.get("section") or "Results").strip().capitalize()
            _topic   = (_ev.get("topic") or "").strip()
            _snippet = (_ev.get("text") or "").strip()[:120].replace("\n", " ")
            _tag     = f"[{_pmcid}:{_section}]"
            if _tag and _tag not in _seen_ev:
                _seen_ev[_tag] = True
                _desc = _topic if _topic else _snippet
                _catalogue_lines.append(f"  {_tag}  — {_desc}")
        if _catalogue_lines:
            _cite_catalogue = (
                "AVAILABLE CITATIONS from retrieved evidence "
                "(you may use ANY of these):\n"
                + "\n".join(_catalogue_lines[:30])  # cap at 30 to control token use
            )

    system = textwrap.dedent("""\
    You are an expert biomedical writer and researcher specialising in Down syndrome (Trisomy 21).
    Your task is to produce a comprehensive, detailed Deep Dive synthesis in response to the question.
    Use the original summary as a starting point but expand it significantly with full scientific depth.

    ══════════════════════════════════════════════════════
    MANDATORY CITATION RULE — NON-NEGOTIABLE:
    ══════════════════════════════════════════════════════
    Every paragraph MUST contain at least one inline citation.
    Citation format: [PMCxxxxxxx:Section]  e.g. [PMC12345678:Results]
    Use SINGLE square brackets ONLY — NEVER double brackets [[...]].
    You MUST use citations from BOTH the original summary AND the AVAILABLE CITATIONS list.
    Do NOT invent PMC IDs — only use IDs present in the original summary or available list.
    A response with NO citations is unacceptable and will be rejected.
    ══════════════════════════════════════════════════════

    Formatting rules:
      - There is NO word limit. Write as much detail as the evidence supports.
      - Structure the response into MULTIPLE clearly separated paragraphs, each covering a distinct
        theme: (1) background/context, (2) key findings, (3) mechanisms or pathways (if relevant),
        (4) clinical implications, (5) limitations and open questions.
      - Separate every paragraph with a BLANK LINE (two newline characters).
        Do NOT merge all text into a single paragraph.
      - When citing multiple papers together in one location use semicolons inside ONE bracket:
          [PMC12345678:Results; PMC87654321:Discussion]  ← CORRECT
          [PMC12345678:Results][PMC87654321:Discussion]  ← WRONG
      - Do NOT repeat the same [PMCxxxxxxx:Section] tag more than TWICE in the entire response.
        Distribute citations across different findings and different papers.
      - Expand on the evidence: explain what each cited finding means, its significance, and how
        findings relate to each other.
      - Address any weaknesses, gaps, or caveats noted in the judge feedback.
      - Use precise scientific language appropriate for a research audience.
      - NEVER use em dashes or en dashes (the long dash characters) anywhere in the text.
        Use commas, semicolons, colons, parentheses, or separate sentences instead;
        use a hyphen only for numeric ranges (e.g. "5-10%").
      - Return ONLY the Deep Dive text. Do NOT add headers like "Deep Dive:", no preamble, no explanation.
    """)
    user_parts = [
        f"QUESTION:\n{question}",
        f"ORIGINAL SUMMARY (starting point — expand and detail this):\n{original_summary}",
    ]
    if _cite_catalogue:
        user_parts.append(_cite_catalogue)
    if judge_notes:
        user_parts.append(f"JUDGE FEEDBACK (address these gaps/weaknesses):\n{judge_notes}")
    user_parts.append(
        "Please write a comprehensive Deep Dive synthesis. "
        "REMINDER: Every paragraph MUST contain at least one [PMCxxxxxxx:Section] citation "
        "from the original summary or the AVAILABLE CITATIONS list above. "
        "A response with no citations is invalid. No word limit."
    )
    user = "\n\n".join(user_parts)

    try:
        client = _OllamaClient(
            host=_JUDGE_OLLAMA_BASE_URL,
            headers={"Authorization": f"Bearer {_JUDGE_OLLAMA_API_KEY}"}
        )
        resp = client.chat(
            model=_IMPROVE_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            options={"temperature": 0.0},
        )
        result = resp["message"]["content"].strip()
        if not result:
            return None
        # ── Post-process citation formatting ──────────────────────────────
        # Strip double brackets: [[PMCxx:Sec]] → [PMCxx:Sec]  (run 3x for nested)
        for _ in range(3):
            result = re.sub(r'\[\[([^\[\]]*)\]\]', r'[\1]', result)
        # Fix bare PMC with section outside bracket: [PMCxx]:Section → [PMCxx:Section]
        result = re.sub(r'\[(PMC\d{4,})\]:([A-Za-z]+)', r'[\1:\2]', result)
        # Merge adjacent citations into one bracket with semicolons:
        #   [PMCxx:Sec1] [PMCyy:Sec2] → [PMCxx:Sec1; PMCyy:Sec2]
        def _merge_adjacent(m):
            combined = re.sub(r'\]\s*\[', '; ', m.group(0)[1:-1])
            return '[' + combined + ']'
        result = re.sub(
            r'\[PMC\d{4,}:[A-Za-z0-9_,-]+(?:,\s*INCLUDE)?\]'
            r'(?:\s*\[PMC\d{4,}:[A-Za-z0-9_,-]+(?:,\s*INCLUDE)?\])+',
            _merge_adjacent, result
        )
        # Cap repeated citations: same [PMCxx:Section] appears at most 3 times
        _seen_cites: dict = {}
        def _cite_cap(m: re.Match) -> str:
            tag = m.group(0)
            _seen_cites[tag] = _seen_cites.get(tag, 0) + 1
            return tag if _seen_cites[tag] <= 3 else ''
        result = re.sub(r'\[PMC\d{4,}:[A-Za-z0-9_,;\s-]+?\]', _cite_cap, result)
        # Clean up stray spaces left before punctuation after citation removal
        result = re.sub(r' +([.,!?])', r'\1', result)
        result = re.sub(r'  +', ' ', result)
        # Ensure paragraphs are separated by blank lines for proper markdown rendering
        result = re.sub(r'(?<!\n)\n(?!\n)', '\n\n', result)
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result
    except Exception:
        return None


def _llm_improve_summary(question: str, original_summary: str, scores: dict) -> str | None:
    """Ask the judge model to rewrite the summary, targeting weak criteria."""
    if not _OLLAMA_AVAILABLE or not _JUDGE_OLLAMA_API_KEY:
        return None

    _score_labels = {1: "Very Poor", 2: "Poor", 3: "Fair", 4: "Good", 5: "Excellent"}
    weak = [
        f"{dim.title()} ({scores[dim]}/5 — {_score_labels.get(scores[dim], '')})"
        for dim in ("accuracy", "informativeness", "completeness")
        if scores.get(dim, 5) < 4
    ]
    weak_str = ", ".join(weak) if weak else "all criteria (general improvement)"
    judge_notes = scores.get("justification", "").strip()

    system = textwrap.dedent("""\
    You are an expert biomedical writer specialising in Down syndrome (Trisomy 21) research.
    Rewrite the provided summary to improve its quality, guided by the judge's specific feedback.
    Rules:
      - Keep the improved summary to 130-150 words.
      - IMPORTANT: Preserve ALL inline citations exactly as they appear in the original.
        Citations follow the format [PMCxxxxxxx:Section] e.g. [PMC12345678:Results].
        Every sentence that states a finding must retain its original citation.
        Do NOT add new citations and do NOT remove existing ones.
      - Address every specific point raised in the JUDGE FEEDBACK — fix omissions, inaccuracies,
        and gaps identified there. Do not merely rephrase; correct the actual content issues noted.
      - Do not fabricate new facts; only rephrase, reorganise, or correct existing content.
      - Return ONLY the improved summary text — no headers, no explanation, no preamble.
    """)
    user_parts = [
        f"QUESTION:\n{question}",
        f"ORIGINAL SUMMARY:\n{original_summary}",
        f"WEAK CRITERIA: {weak_str}",
    ]
    if judge_notes:
        user_parts.append(f"JUDGE FEEDBACK:\n{judge_notes}")
    user_parts.append("Please rewrite the summary to address the judge's feedback and weak criteria.")
    user = "\n\n".join(user_parts)

    try:
        client = _OllamaClient(
            host=_JUDGE_OLLAMA_BASE_URL,
            headers={"Authorization": f"Bearer {_JUDGE_OLLAMA_API_KEY}"}
        )
        resp = client.chat(
            model=_IMPROVE_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            options={"temperature": 0.3},
        )
        improved = resp["message"]["content"].strip()
        return improved if improved else None
    except Exception:
        return None



def get_current_conversation():
    """Get the current conversation or None."""
    if st.session_state.current_conv_id is None:
        return None
    for conv in st.session_state.conversations:
        if conv["id"] == st.session_state.current_conv_id:
            return conv
    return None

def create_new_conversation():
    """Create a new conversation and set it as current."""
    new_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_conv = {
        "id": new_id,
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().isoformat()
    }
    st.session_state.conversations.insert(0, new_conv)
    st.session_state.current_conv_id = new_id
    return new_conv

# -------------------- About Page --------------------
def _render_corpus_downloads():
    """Download buttons for the versioned corpus manifest and screening audit.

    Files are generated by srv/scripts/export_corpus_manifest.py into srv/ui/data/
    (re-run it after any corpus rebuild). Shown on the About page and the Corpus page.
    """
    _data_dir = Path(__file__).resolve().parent / "data"
    _dl_specs = [
        ("corpus_manifest_DS_INCLUDE_2026_09_02.csv",
         "Corpus manifest (1,594 papers, CSV)"),
        ("screening_audit_DS_INCLUDE_2026_09_02.csv",
         "Screening audit (2,075 records, CSV)"),
    ]
    _dl_cols = st.columns(len(_dl_specs))
    for _col, (_fname, _label) in zip(_dl_cols, _dl_specs):
        _fpath = _data_dir / _fname
        with _col:
            if _fpath.exists():
                st.download_button(_label, data=_fpath.read_bytes(),
                                   file_name=_fname, mime="text/csv",
                                   key=f"dl_{_fname}")
            else:
                st.caption(f"{_fname} not found on the server")


def render_corpus_page():
    """Render the Corpus page: what is indexed, how it was selected, and downloads."""
    st.markdown('<div style="height:140px;"></div>', unsafe_allow_html=True)
    st.markdown("## The Corpus")
    st.markdown("---")
    st.markdown(
        "The T21 Research Assistant answers questions from **1,594 open-access "
        "peer-reviewed original research articles** on Down Syndrome (Trisomy 21) from "
        "PubMed Central, published between **1970 and September 2026** and including "
        "**158 NIH INCLUDE-funded papers**. Every stored paper satisfies all selection "
        "criteria; nothing else is stored or searched."
    )
    st.markdown("### How the corpus was selected")
    st.markdown(
        "```\n"
        "2,075  records identified by the PMC query\n"
        "       (\"Down syndrome\"[MeSH] AND open access AND published article,\n"
        "        snapshot of 2026-09-02)\n"
        "1,891  eligible after screening (excluded publication types removed;\n"
        "       PMCID present; not retracted per MEDLINE and Crossref)\n"
        "1,837  with usable full text served by PMC\n"
        "1,669  with at least 4 of the 6 canonical sections\n"
        "1,594  original research articles (reviews, preprints, errata excluded)\n"
        "```\n"
    )
    st.markdown(
        "Each paper contributes section-level chunks (65,677 embeddings in total; "
        "Abstract, Introduction, Methods, Results, Discussion and Conclusion are stored, "
        "and retrieval searches Results, Discussion, Methods and Conclusion)."
    )
    st.markdown("### Downloads")
    st.markdown(
        "The complete, versioned paper list (PMID, PMCID, DOI, title, year, journal, "
        "MeSH terms, keywords, INCLUDE provenance, sections and per-paper chunk counts) "
        "and the screening audit recording the inclusion or exclusion status, with "
        "reason, of every record considered."
    )
    _render_corpus_downloads()
    st.markdown("---")
    st.markdown(
        "**Funding:** Supported by the NIH INCLUDE Project, grant **R01 HD109765**.  \n"
        "Developed by the **Costello Lab**, University of Colorado Anschutz."
    )


def render_about_page():
    """Render the About page."""
    st.markdown('<div style="height:140px;"></div>', unsafe_allow_html=True)
    st.markdown("## About the T21 Research Assistant")
    st.markdown("---")

    # Overview
    st.markdown(
        "The **T21 Research Assistant** is an AI-powered question-answering tool built on a "
        "curated corpus of **1,594 peer-reviewed open-access Down Syndrome (Trisomy 21) articles** "
        "from PubMed Central, including 158 NIH INCLUDE-funded papers. It provides evidence-based, "
        "citation-backed answers grounded exclusively in the published DS literature — helping "
        "researchers, clinicians, and educators explore findings quickly and rigorously."
    )

    # What it does section
    st.markdown("### What it does")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**Retrieval-Augmented Generation (RAG)**  \n"
            "Queries are matched against 65,677 section-level vectors using a Nemotron "
            "semantic search index (Milvus IVF_FLAT, 2048-dim). A cross-encoder then "
            "reranks candidates for precision before synthesis."
        )
    with col2:
        st.markdown(
            "**Structured Answers with Citations**  \n"
            "Every response is structured into *Summary*, *Key Evidence*, and *Reasoning* "
            "sections. All claims are pinned to clickable PubMed Central citations so "
            "results are fully traceable back to source. An optional **Deep Dive** expander "
            "provides on-demand extended synthesis with LLM-as-Judge quality scoring."
        )
    st.markdown("")

    # Corpus section
    st.markdown("### The Corpus")
    st.markdown(
        "| | |\n"
        "|---|---|\n"
        "| **Papers indexed** | 1,594 open-access peer-reviewed original research articles |\n"
        "| **Selection** | 2,075 PMC records screened → 1,891 eligible → 1,837 with full text → 1,669 with ≥4 sections → 1,594 original research articles |\n"
        "| **NIH INCLUDE-funded** | 158 articles (tagged with INCLUDE badge) |\n"
        "| **Vectors** | 65,677 section-level embeddings (6 sections stored per paper) |\n"
        "| **Sections searched** | Results · Discussion · Methods · Conclusion |\n"
        "| **Embedding model** | NVIDIA Llama-Nemotron-Embed-VL-1B-v2 (2048-dim, via DeepInfra) |\n"
        "| **Reranker** | NVIDIA Llama-Nemotron-Rerank-VL-1B-v2 (via DeepInfra) |\n"
        "| **Synthesis LLM** | NVIDIA Nemotron-3-Super-120B (via Ollama remote cloud) |\n"
    )

    # Corpus downloads: the versioned paper list and the screening audit, generated by
    # srv/scripts/export_corpus_manifest.py (re-run it after any corpus rebuild).
    st.markdown("#### Download the corpus list")
    st.markdown(
        "The complete, versioned list of all 1,594 corpus papers (PMID, PMCID, DOI, "
        "title, year, journal, MeSH terms, keywords, INCLUDE provenance, sections and "
        "chunk counts), and the full screening audit recording the inclusion or "
        "exclusion status of every one of the 2,075 records considered."
    )
    _render_corpus_downloads()

    # Pipeline section
    st.markdown("### Pipeline Overview")
    st.markdown(
        "```\n"
        "User Question\n"
        "  → Question Judge   (pre-flight validity check)\n"
        "  → Retrieve         (Milvus semantic search, 5× oversampling)\n"
        "  → Rerank           (Nemotron cross-encoder scoring)\n"
        "  → Synthesise       (LLM generates Summary · Key Evidence · Reasoning)\n"
        "  → Safety Guard     (prescription-language filter)\n"
        "  → Finalise         (citation linking, 150-word cap)\n"
        "\n"
        "  [on-demand]\n"
        "  → Deep Dive        (extended synthesis + LLM-as-Judge scoring)\n"
        "```\n"
    )

    # Disclaimer
    st.warning(
        "**Research use only.** This tool summarises published literature and does not "
        "provide medical advice, diagnoses, or treatment recommendations. Always consult "
        "a qualified clinician for health-related decisions."
    )

    st.markdown("---")
    st.markdown(
        "**Funding:** Supported by the NIH INCLUDE Project, grant **R01 HD109765**.  \n"
        "Developed by the **Costello Lab**, University of Colorado Anschutz."
    )


# -------------------- Team Page --------------------
def _member_card(name, title, institution, dept, phone, email, profile_url, accent="#1e4d8b", bg="#f8fafc", border="#dde6f0"):
    """Render a team member card."""
    phone_str = phone if phone else 'N/A'
    profile_html = f"<a href='{profile_url}' target='_blank'>University Profile ↗</a>" if profile_url else "N/A"
    email_html = f"<a href='mailto:{email}'>{email}</a>" if email else "N/A"
    st.markdown(
        f"""
        <style>
        @media (max-width: 768px) {{
            .member-grid {{ flex-direction: column !important; }}
            .member-grid > div {{ width: 100% !important; }}
        }}
        </style>
        <div style="border-left:4px solid {accent}; border:1px solid {border};
                    border-radius:10px; padding:14px 18px 14px 18px; background:{bg}; margin-bottom:12px;">
            <div style="font-size:1.25rem;font-weight:700;color:{accent};margin-bottom:10px;">{name}</div>
            <div class="member-grid" style="display:flex; flex-wrap:wrap; gap:6px 32px;">
                <div style="min-width:200px; flex:1;">
                    <p style='font-size:1.05rem;margin:3px 0;'><strong>Title:</strong> {title}</p>
                    <p style='font-size:1.05rem;margin:3px 0;'><strong>Institution:</strong> {institution}</p>
                    <p style='font-size:1.05rem;margin:3px 0;'><strong>Department:</strong> {dept}</p>
                </div>
                <div style="min-width:200px; flex:1;">
                    <p style='font-size:1.05rem;margin:3px 0;'><strong>Phone:</strong> {phone_str}</p>
                    <p style='font-size:1.05rem;margin:3px 0;'><strong>Email:</strong> {email_html}</p>
                    <p style='font-size:1.05rem;margin:3px 0;'>{profile_html}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


def render_team_page():
    """Render the Team information page."""
    st.markdown('<div style="height:140px;"></div>', unsafe_allow_html=True)

    st.markdown("## Our Team")
    st.markdown(
        "The T21 Research Assistant is developed by the **Costello Lab** at the "
        "University of Colorado Anschutz as part of the NIH INCLUDE "
        "project (R01 HD109765), advancing evidence-based tools for Down Syndrome research."
    )
    st.markdown("---")

    # ── Principal Investigator ──────────────────────────────────────────────
    st.markdown("### Principal Investigator")
    _member_card(
        name="James Costello, PhD",
        title="Professor",
        institution="University of Colorado Anschutz",
        dept="Department of Pharmacology",
        phone="(303) 724-8619",
        email="james.costello@cuanschutz.edu",
        profile_url="https://medschool.cuanschutz.edu/pharmacology/faculty/primary-faculty/james-costello-phd",
    )

    # ── Collaborators ───────────────────────────────────────────────────────
    st.markdown("### Collaborators")
    _member_card(
        name="Matthew Galbraith, PhD",
        title="Associate Research Professor",
        institution="University of Colorado Anschutz",
        dept="Pharmacology",
        phone="(303) 724-3661",
        email="matthew.galbraith@cuanschutz.edu",
        profile_url="https://profiles.ucdenver.edu/display/226300",
        accent="#2c6fad",
        bg="#f4f8fd",
        border="#c5d5ea",
    )
    _member_card(
        name="Kelly D. Sullivan, PhD",
        title="Associate Professor",
        institution="University of Colorado Anschutz",
        dept="SOM Pediatric Developmental Biology – General Operations",
        phone="(303) 724-8619",
        email="kelly.d.sullivan@cuanschutz.edu",
        profile_url="https://profiles.ucdenver.edu/display/226266",
        accent="#2c6fad",
        bg="#f4f8fd",
        border="#c5d5ea",
    )
    _member_card(
        name="Joaquin Espinosa, PhD",
        title="Professor",
        institution="University of Colorado Anschutz",
        dept="Pharmacology & Executive Director of the Crnic Institute",
        phone="(303)724-7389",
        email="Joaquin.Espinosa@cuanschutz.edu",
        profile_url="https://medschool.cuanschutz.edu/pharmacology/faculty/primary-faculty/joaquin-espinosa-phd",
        accent="#2c6fad",
        bg="#f4f8fd",
        border="#c5d5ea",
    )
    _member_card(
        name="Milton Pividori, PhD",
        title="Assistant Professor",
        institution="University of Colorado Anschutz",
        dept="Biomedical Informatics",
        phone="(312) 560-0558",
        email="milton.pividori@cuanschutz.edu",
        profile_url="https://som.cuanschutz.edu/Profiles/Faculty/Profile/38036",
        accent="#2c6fad",
        bg="#f4f8fd",
        border="#c5d5ea",
    )

    # ── Research Team ───────────────────────────────────────────────────────
    st.markdown("### Research Team")
    _member_card(
        name="Sutanu Nandi, PhD",
        title="Post-Doctoral Fellow",
        institution="University of Colorado Anschutz",
        dept="Department of Pharmacology",
        phone="(303) 724-8619",
        email="sutanu.nandi@cuanschutz.edu",
        profile_url="https://profiles.ucdenver.edu/display/45295805",
        accent="#4a90d9",
        bg="#f0f4fa",
        border="#c5d5ea",
    )
    _member_card(
        name="Zenitha Sundararajan",
        title="Bioinformatics Analyst",
        institution="University of Colorado Anschutz",
        dept="Anschutz Campus – Linda Crnic Institute for Down Syndrome",
        phone=None,
        email="zenitha.sundararajan@cuanschutz.edu",
        profile_url=None,
        accent="#4a90d9",
        bg="#f0f4fa",
        border="#c5d5ea",
    )
    _member_card(
        name="Marc Subirana-Granés, PhD",
        title="Post-Doctoral Fellow",
        institution="University of Colorado Anschutz",
        dept="Department of Biomedical Informatics",
        phone=None,
        email="marc.subiranagranes@cuanschutz.edu",
        profile_url="https://pivlab.org/members/marc-subirana.html",
        accent="#4a90d9",
        bg="#f0f4fa",
        border="#c5d5ea",
    )

    st.info(
        "**Funding:** This work is supported by the National Institutes of Health "
        "INCLUDE (INvestigation of Co-occurring conditions across the Lifespan to Understand "
        "Down syndromE) Project, grant **R01 HD109765**."
    )


# -------------------- FAQ Page --------------------
# Fallback corpus figures — used only if the live /health call fails, so the FAQ
# never shows a raw "N/A" to the reader. Kept roughly current, but these are NOT
# the source of truth: _build_faq_items() below always prefers live Milvus counts.
_FAQ_FALLBACK_DOCS = "1,594"
_FAQ_FALLBACK_INCLUDE = "158"
_FAQ_FALLBACK_VECTORS = "65,677"


def _build_faq_items(doc_count: str = "N/A", include_count: str = "N/A", vector_count: str = "N/A"):
    """Build the FAQ question/answer list with corpus figures pulled live from Milvus
    (via /health -> estimated_documents / include_count / total_vectors). Falls back to
    the last-known static figures if the backend didn't return a usable value, so the
    page never renders a bare "N/A" for the reader."""
    docs = doc_count if doc_count and doc_count != "N/A" else _FAQ_FALLBACK_DOCS
    incl = include_count if include_count and include_count != "N/A" else _FAQ_FALLBACK_INCLUDE
    vecs = vector_count if vector_count and vector_count != "N/A" else _FAQ_FALLBACK_VECTORS

    return [
    (
        "What is the T21 Research Assistant?",
        "The T21 Research Assistant is an AI-powered tool that enables evidence-based "
        f"question answering over a curated corpus of {docs} open-access peer-reviewed Down Syndrome "
        f"(Trisomy 21) articles from PubMed Central, including {incl} NIH INCLUDE-funded papers. "
        "It uses a multi-agent retrieval-augmented generation (RAG) pipeline to retrieve "
        "relevant sections, rerank them with a cross-encoder, and synthesise a structured "
        "answer backed by in-text citations.",
    ),
    (
        "What publications does this tool cover?",
        f"The corpus is **{docs} open-access peer-reviewed original research articles** from "
        "PubMed Central, covering Down Syndrome research published from 1970 through "
        "September 2026, "
        f"including **{incl} NIH INCLUDE-funded papers** (marked with an INCLUDE badge in "
        f"citations) and represented by **{vecs} section-level embeddings**. Every stored paper "
        "satisfies all selection criteria: identified by a PMC Down-syndrome MeSH query "
        "(2,075 records), screened for eligibility (excluded publication types, retractions "
        "checked against both MEDLINE and Crossref), full text available in PMC, at least 4 of "
        "the 6 canonical sections extracted, and original research article type (reviews, "
        "preprints and errata are excluded). Six sections are stored per paper (Abstract, "
        "Introduction, Methods, Results, Discussion, Conclusion), while retrieval searches "
        "**Results, Discussion, Methods, and Conclusion** only, to focus answers on primary "
        "findings. The complete versioned paper list (with PMCID, DOI, MeSH terms, INCLUDE "
        "provenance and per-paper chunk counts) and the full screening audit are downloadable "
        "from the **About** page.",
    ),
    (
        "What kinds of questions can I ask?",
        "You can ask any research question about Down Syndrome biology, clinical findings, "
        "co-occurring conditions, treatments, or molecular mechanisms — for example:\n\n"
        "- *What is the prevalence of autism in individuals with Down syndrome?*\n"
        "- *How does trisomy 21 affect immune function?*\n"
        "- *What therapies have been studied for cognitive decline in Down syndrome?*\n\n"
        "Questions must be **explicitly about Down syndrome or Trisomy 21**, or about a "
        "condition, gene, or mechanism known to be studied in the DS context (e.g. Alzheimer's, "
        "DYRK1A, leukemia, hypothyroidism). General biomedical questions with no DS connection "
        "will be flagged as out-of-scope. The tool is designed for researchers, clinicians, "
        "and educators — it is **not** for personal medical advice.",
    ),
    (
        "How are answers generated?",
        "Each query goes through a five-stage pipeline:\n\n"
        "1. **Question Judge** — A fast pre-flight LLM call classifies the question as "
        "valid (DS-relevant) or out-of-scope before any retrieval occurs.\n"
        "2. **Retrieve** — Milvus vector search identifies the most semantically relevant "
        "section-level chunks across the corpus (5× oversampling for diversity).\n"
        "3. **Rerank** — A Nemotron cross-encoder rescores each chunk for relevance to the "
        "specific question.\n"
        "4. **Synthesise** — A large language model (LLM) reads the top-ranked chunks and "
        "produces a structured answer: *Summary*, *Key Evidence*, and *Reasoning*.\n"
        "5. **Finalise** — Citation links are verified against the actual retrieved evidence "
        "(hallucinated IDs are removed), safety filters applied, and the response is "
        "formatted with clickable PMC hyperlinks.",
    ),
    (
        "What do the three response sections mean?",
        "Every answer contains three sections:\n\n"
        "- **Summary** — A concise synthesis (≤ 150 words) of the evidence with inline "
        "citations linking to PubMed Central.\n"
        "- **Key Evidence** — Bullet-point highlights of the most relevant study findings, "
        "each pinned to its source paper.\n"
        "- **Reasoning** — A plain-language explanation of how the evidence was selected, "
        "any gaps in the literature, and interpretive caveats the reader should keep in mind.\n\n"
        "An optional **Deep Dive** expander below each response lets you request a longer, "
        "more detailed synthesis — see the *What is Deep Dive?* entry below.",
    ),
    (
        "What is the Deep Dive feature?",
        "Each response includes an optional **Deep Dive** expander. Clicking the "
        "**Deep Dive** button runs two additional on-demand steps:\n\n"
        "1. **LLM-as-Judge scoring** — The original summary is independently scored on "
        "*Accuracy*, *Informativeness*, and *Completeness* (1–5 stars). Scores are shown "
        "in a collapsed *Original Summary — LLM-as-Judge scores* panel.\n"
        "2. **Extended synthesis** — A second LLM generates a detailed multi-paragraph "
        "synthesis drawing on the full retrieved evidence pool (up to 30 source papers with "
        "inline PMC citations). This is longer and more comprehensive than the standard "
        "150-word Summary.\n\n"
        "Use Deep Dive when you want a richer exploration of a topic or when the standard "
        "summary leaves questions unanswered. **Re-score Deep Dive** and **Re-generate Deep Dive** "
        "buttons allow on-demand refresh of either step.",
    ),
    (
        "What does \"Insufficient Evidence\" mean?",
        "If the retrieved chunks do not contain enough information to answer the question "
        "reliably, the tool returns an *Insufficient Evidence* response rather than "
        "fabricating an answer. This can happen because:\n\n"
        "- The question is outside the scope of Down Syndrome research (caught by the "
        "pre-flight Question Judge before any retrieval).\n"
        f"- The topic is not well-covered in the {docs}-article corpus.\n"
        "- The question references a biological entity not present in the DS literature.\n\n"
        "In all cases a brief explanation is provided stating which of these applies.",
    ),
    (
        "Can I trust the citations?",
        "Inline citations (e.g. `[PMC12345678:Results]`) link directly to the source article "
        "on PubMed Central. The system includes a hallucination filter that removes any "
        "citation IDs not present in the retrieved evidence, so every hyperlink corresponds "
        "to a real paper that was actually retrieved and used in the synthesis. Always verify "
        "important findings in the original publications.",
    ),
    (
        "How many evidence chunks are used per answer (Top-K)?",
        "The number of evidence chunks used for each answer is set automatically — there is "
        "no manual control. The system retrieves **12** top-ranked chunks by default and "
        "raises this automatically (up to **20**) for complex, comparative, or list-style "
        "questions, so broader questions draw on more evidence without any configuration. "
        "Candidates are always drawn evenly from the Results, Discussion, Methods, and "
        "Conclusion sections and reranked before the best chunks are passed to synthesis.",
    ),
    (
        "Is this tool a substitute for clinical advice?",
        "**No.** The T21 Research Assistant is for **research and educational use only**. "
        "It summarises published literature and does not provide personalised medical advice, "
        "diagnoses, or treatment recommendations. Always consult a qualified clinician for "
        "health-related decisions.",
    ),
    (
        "How can I provide feedback or report an issue?",
        "Use the **Contact** button in the navigation bar to send a message directly to "
        "the development team, or email "
        "[james.costello@cuanschutz.edu](mailto:james.costello@cuanschutz.edu) directly.",
    ),
]


def render_faq_page():
    """Render the FAQ page. Corpus figures (article count, INCLUDE count, vector
    count) are pulled live from Milvus via /health on every render (subject to the
    60s cache on fetch_system_info), so the FAQ text stays in sync with the actual
    indexed corpus without needing a manual edit after each re-index."""
    st.markdown('<div style="height:140px;"></div>', unsafe_allow_html=True)
    st.markdown("## Frequently Asked Questions")
    st.markdown(
        "Answers to common questions about the T21 Research Assistant — "
        "what it does, how it works, and how to get the most from it."
    )
    st.markdown("---")

    _faq_db_stats = fetch_system_info().get("db_stats", {})
    _faq_items = _build_faq_items(
        doc_count=_faq_db_stats.get("estimated_documents", "N/A"),
        include_count=_faq_db_stats.get("include_count", "N/A"),
        vector_count=_faq_db_stats.get("total_vectors", "N/A"),
    )

    for i, (question, answer) in enumerate(_faq_items):
        with st.expander(f"**Q{i+1}. {question}**", expanded=(i == 0)):
            st.markdown(answer)

    st.markdown("---")
    st.info(
        "Still have questions? Use the **Contact** button above or email "
        "[james.costello@cuanschutz.edu](mailto:james.costello@cuanschutz.edu)."
    )


# -------------------- Contact Page --------------------
_CONTACT_TO = "james.costello@cuanschutz.edu"
_SMTP_HOST   = os.getenv("SMTP_HOST", "localhost")
_SMTP_PORT   = int(os.getenv("SMTP_PORT", "25"))
_SMTP_USER   = os.getenv("SMTP_USER", "")
_SMTP_PASS   = os.getenv("SMTP_PASS", "")
_SMTP_TLS    = os.getenv("SMTP_USE_TLS", "false").lower() == "true"


def _send_contact_email(sender_name: str, sender_email: str, subject: str, message: str) -> None:
    """Send the contact form message via SMTP."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[T21 Research Assistant] {subject}"
    msg["From"]    = f"{sender_name} <{sender_email}>"
    msg["To"]      = _CONTACT_TO
    msg["Reply-To"] = sender_email

    body = (
        f"Message from T21 Research Assistant contact form\n"
        f"{'='*55}\n"
        f"Name   : {sender_name}\n"
        f"Email  : {sender_email}\n"
        f"Subject: {subject}\n"
        f"{'='*55}\n\n"
        f"{message}\n"
    )
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=10) as server:
        server.ehlo()
        if _SMTP_TLS:
            server.starttls()
            server.ehlo()
        if _SMTP_USER and _SMTP_PASS:
            server.login(_SMTP_USER, _SMTP_PASS)
        server.sendmail(sender_email, [_CONTACT_TO], msg.as_string())


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def render_contact_page():
    """Render the Contact Us form page."""
    st.markdown('<div style="height:140px;"></div>', unsafe_allow_html=True)
    st.markdown("## Contact Us")
    st.markdown(
        "Have a question, suggestion, or feedback about the T21 Research Assistant? "
        "Fill in the form below and your message will be sent directly to "
        "**James Costello, PhD** (Principal Investigator)."
    )
    st.markdown("---")

    with st.form(key="contact_form", clear_on_submit=True):
        col_l, col_r = st.columns(2)
        with col_l:
            name = st.text_input("Your Name *", placeholder="Full name")
        with col_r:
            email = st.text_input("Your Email *", placeholder="your@email.com")

        subject = st.text_input("Subject *", placeholder="Brief description of your message")
        message = st.text_area("Message *", placeholder="Your message...", height=200)
        submitted = st.form_submit_button("Send Message", type="primary")

    if submitted:
        errors = []
        if not name.strip():
            errors.append("Name is required.")
        if not email.strip():
            errors.append("Email is required.")
        elif not _EMAIL_RE.match(email.strip()):
            errors.append("Please enter a valid email address.")
        if not subject.strip():
            errors.append("Subject is required.")
        if not message.strip():
            errors.append("Message is required.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            try:
                _send_contact_email(
                    sender_name=name.strip(),
                    sender_email=email.strip(),
                    subject=subject.strip(),
                    message=message.strip(),
                )
                st.success(
                    f"Thank you, {name.strip()}! Your message has been sent to Prof. Costello. "
                    "We will get back to you at the email address you provided."
                )
            except Exception as exc:  # noqa: BLE001
                st.error(
                    f"Could not send your message automatically ({exc}). "
                    f"Please email Prof. Costello directly at "
                    f"[james.costello@cuanschutz.edu](mailto:james.costello@cuanschutz.edu)."
                )

    st.markdown("---")
    st.markdown(
        "You can also reach us directly: "
        "[james.costello@cuanschutz.edu](mailto:james.costello@cuanschutz.edu)"
    )

# -------------------- ChatGPT-Style CSS --------------------
st.markdown("""
<style>
    /* Global styles */
    html, body, .stApp {
        font-family: Arial, 'Helvetica Neue', Helvetica, sans-serif;
        font-size: 17px !important;
    }

    /* Scale up all standard text elements */
    p, li, span, label, div, td, th, input, textarea, select, button {
        font-size: 1.05rem;
    }

    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        font-size: 1.05rem !important;
    }

    /* Ensure scroll anchors respect the fixed header */
    html {
        scroll-padding-top: 180px;
    }

    /* Hide Streamlit branding but keep header for sidebar toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Keep header always visible for sidebar toggle */
    header, header[data-testid="stHeader"] {
        visibility: visible !important;
        display: block !important;
        z-index: 999999 !important;
        position: fixed !important;
        top: 0 !important;
        background-color: #f8f9fa !important;
    }

    /* Ensure sidebar toggle button is always visible */
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="baseButton-header"],
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        z-index: 999999 !important;
    }

    /* Main container - light professional theme */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Streamlit's real scroll container — push it below the fixed header */
    section[data-testid="stMain"] {
        padding-bottom: 130px !important;
    }

    /* Also target the inner block container */
    section[data-testid="stMain"] .block-container,
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        max-width: 100% !important;
    }

    /* AppViewBlockContainer */
    div[data-testid="stAppViewBlockContainer"] {
        padding-top: 0 !important;
    }

    /* Chat message container scroll margin */
    div[data-testid="stChatMessageContainer"],
    div[data-testid="stVerticalBlock"] {
        scroll-margin-top: 160px;
    }

    /* Every chat message bubble — when scrolled into view, clear the fixed header */
    div[data-testid="stChatMessage"] {
        scroll-margin-top: 180px !important;
    }

    /* The stMain scroll container — native scroll anchoring */
    section[data-testid="stMain"] {
        scroll-padding-top: 180px !important;
    }

    /* ── Mobile responsive ── */
    @media (max-width: 768px) {
        html, body, .stApp {
            font-size: 15px !important;
        }
        /* On mobile Streamlit collapses the sidebar — nav bar spans full width */
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        section[data-testid="stMain"] .block-container,
        .main .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        div[data-testid="stChatMessage"] {
            scroll-margin-top: 220px !important;
        }
        section[data-testid="stMain"] {
            scroll-padding-top: 220px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# JS: after Streamlit auto-scrolls a new message into view, nudge the scroll
# container back down by the header height so text isn't hidden.
st.markdown("""
<script>
(function() {
    const HEADER_H = 160;  // px — matches fixed nav+title height
    function nudgeScroll() {
        // Streamlit scrolls the stMain section element
        const main = document.querySelector('section[data-testid="stMain"]');
        if (!main) return;
        // If the scrollTop is very small (near top) don't nudge — page just loaded
        if (main.scrollTop < HEADER_H) return;
        // Check if the last chat message top is hidden behind the header
        const msgs = main.querySelectorAll('[data-testid="stChatMessage"]');
        if (!msgs.length) return;
        const last = msgs[msgs.length - 1];
        const rect = last.getBoundingClientRect();
        if (rect.top < HEADER_H + 8) {
            main.scrollTop += (HEADER_H + 8 - rect.top);
        }
    }
    // Watch for new DOM nodes (new messages being added)
    const observer = new MutationObserver(function() {
        setTimeout(nudgeScroll, 80);
    });
    function attach() {
        const main = document.querySelector('section[data-testid="stMain"]');
        if (main) {
            observer.observe(main, { childList: true, subtree: true });
        } else {
            setTimeout(attach, 200);
        }
    }
    attach();
})();
</script>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e5e7eb;
        visibility: visible !important;
        display: block !important;
        min-width: 320px !important;
        width: 320px !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        width: 320px !important;
    }

    section[data-testid="stSidebar"] > div {
        background-color: #ffffff !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] * {
        color: #1f2937 !important;
    }

    /* Slider styling for light theme */
    section[data-testid="stSidebar"] .stSlider > div > div > div {
        background-color: #e5e7eb !important;
    }
    section[data-testid="stSidebar"] .stSlider > div > div > div > div {
        background-color: #10a37f !important;
    }
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: #1f2937 !important;
    }

    /* Checkbox styling for light theme */
    section[data-testid="stSidebar"] .stCheckbox label span {
        color: #1f2937 !important;
    }
    section[data-testid="stSidebar"] .stCheckbox > label > div[data-testid="stCheckbox"] {
        background-color: #ffffff !important;
        border-color: #d1d5db !important;
    }

    /* Ensure sidebar subheaders are visible */
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] .stSubheader {
        color: #1f2937 !important;
        visibility: visible !important;
        display: block !important;
    }

    /* Settings section styling */
    section[data-testid="stSidebar"] .stSlider,
    section[data-testid="stSidebar"] .stCheckbox,
    section[data-testid="stSidebar"] .stNumberInput {
        visibility: visible !important;
        display: block !important;
        margin: 8px 0 !important;
    }

    /* Number input styling for light theme */
    section[data-testid="stSidebar"] .stNumberInput input {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border: 1px solid #d1d5db !important;
    }
    section[data-testid="stSidebar"] .stNumberInput button {
        background-color: #f3f4f6 !important;
        color: #1f2937 !important;
    }

    /* New chat button - Light Navy Blue */
    .new-chat-btn {
        background: #e8eef7;
        border: 1px solid #c5d5ea;
        border-radius: 8px;
        padding: 12px 16px;
        color: #1e3a5f;
        font-size: 14px;
        cursor: pointer;
        width: 100%;
        text-align: left;
        margin-bottom: 8px;
        transition: background 0.2s;
    }

    .new-chat-btn:hover {
        background: #d0dff0;
    }

    /* Conversation item in sidebar */
    .conv-item {
        padding: 10px 12px;
        border-radius: 8px;
        margin-bottom: 4px;
        cursor: pointer;
        transition: background 0.2s;
        font-size: 14px;
        color: #1f2937;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .conv-item:hover {
        background: #f3f4f6;
    }

    .conv-item.active {
        background: #e5e7eb;
    }

    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 24px;
    }

    /* Message styles */
    .message-row {
        padding: 24px 0;
        border-bottom: 1px solid #e5e7eb;
    }

    .message-row.user {
        background-color: #ffffff;
    }

    .message-row.assistant {
        background-color: #f3f4f6;
    }

    .message-content {
        max-width: 800px;
        margin: 0 auto;
        display: flex;
        gap: 24px;
        padding: 0 24px;
    }

    .message-avatar {
        width: 36px;
        height: 36px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        flex-shrink: 0;
    }

    .message-avatar.user {
        background: linear-gradient(135deg, #0f2557, #1e4d8b);
        color: white;
    }

    .message-avatar.assistant {
        background: #10a37f;
        color: white;
    }

    .message-text {
        flex: 1;
        color: #1f2937;
        line-height: 1.75;
        font-size: 16px;
    }

    .message-text p {
        margin-bottom: 1rem;
    }

    .message-text a {
        color: #2563eb;
        text-decoration: none;
    }

    .message-text a:hover {
        text-decoration: underline;
    }

    /* Input area at bottom */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 300px;
        right: 0;
        background: linear-gradient(180deg, transparent 0%, #f8f9fa 20%);
        padding: 24px;
    }

    .input-wrapper {
        max-width: 800px;
        margin: 0 auto;
        position: relative;
    }

    /* Streamlit chat input styling - Light Navy Blue background */
    .stChatInput {
        background: #e8eef7 !important;
        border: 1px solid #c5d5ea !important;
        border-radius: 12px !important;
    }

    .stChatInput > div {
        background: #e8eef7 !important;
    }

    .stChatInput textarea {
        background: #e8eef7 !important;
        color: #1e3a5f !important;
        border: none !important;
    }

    .stChatInput textarea::placeholder {
        color: #4a6fa5 !important;
    }

    /* Welcome screen */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 60vh;
        text-align: center;
        color: #1f2937;
    }

    .welcome-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #1e4d8b, #10a37f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .welcome-subtitle {
        font-size: 1.1rem;
        color: #4b5563;
        max-width: 600px;
        line-height: 1.6;
    }

    /* Example prompts */
    .examples-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        max-width: 700px;
        margin-top: 2rem;
    }

    .example-card {
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        padding: 16px;
        text-align: left;
        cursor: pointer;
        transition: all 0.2s;
    }

    .example-card:hover {
        background: #f3f4f6;
        border-color: #2563eb;
    }

    .example-title {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 4px;
        font-size: 14px;
    }

    .example-text {
        color: #4b5563;
        font-size: 13px;
    }

    /* Expander styling for light theme */
    .streamlit-expanderHeader {
        background: #ffffff !important;
        border-radius: 8px !important;
        color: #1f2937 !important;
    }

    .streamlit-expanderContent {
        background: #ffffff !important;
        border-radius: 0 0 8px 8px !important;
        padding: 8px 4px !important;
    }

    /* Summary and Key Evidence text alignment */
    .streamlit-expanderContent p {
        text-align: justify;
        line-height: 1.85;
        font-size: 15px;
        color: #1f2937;
        margin-bottom: 0.75rem;
    }

    .streamlit-expanderContent li {
        text-align: justify;
        line-height: 1.85;
        font-size: 15px;
        color: #1f2937;
        margin-bottom: 0.4rem;
    }

    /* Button styling - Light Navy Blue */
    .stButton > button {
        background: #e8eef7 !important;
        color: #1e3a5f !important;
        border: 1px solid #c5d5ea !important;
        border-radius: 8px !important;
    }

    .stButton > button:hover {
        background: #d0dff0 !important;
        border-color: #1e4d8b !important;
    }

    /* Download buttons - Light Navy Blue */
    .stDownloadButton > button {
        background: #d0dff0 !important;
        color: #1e3a5f !important;
        border: 1px solid #a8c5e0 !important;
    }

    .stDownloadButton > button:hover {
        background: #b8d0e8 !important;
        border-color: #1e4d8b !important;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f3f4f6;
    }

    ::-webkit-scrollbar-thumb {
        background: #9ca3af;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #6b7280;
    }

    /* Loading spinner */
    .loading-dots {
        display: flex;
        gap: 8px;
        padding: 8px 0;
    }

    .loading-dot {
        width: 8px;
        height: 8px;
        background: #10a37f;
        border-radius: 50%;
        animation: bounce 1.4s ease-in-out infinite;
    }

    .loading-dot:nth-child(1) { animation-delay: 0s; }
    .loading-dot:nth-child(2) { animation-delay: 0.2s; }
    .loading-dot:nth-child(3) { animation-delay: 0.4s; }

    @keyframes bounce {
        0%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-8px); }
    }

    /* Enhanced loading animation */
    .loading-container {
        display: flex;
        flex-direction: column;
        gap: 16px;
        padding: 20px;
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfeff 100%);
        border-radius: 12px;
        border: 1px solid #d1fae5;
    }

    .loading-header {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .loading-spinner {
        width: 24px;
        height: 24px;
        border: 3px solid #d1fae5;
        border-top-color: #10a37f;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .loading-text {
        font-size: 15px;
        font-weight: 500;
        color: #065f46;
    }

    .loading-progress {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .loading-step {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 13px;
        color: #6b7280;
        padding: 6px 0;
    }

    .loading-step.active {
        color: #10a37f;
        font-weight: 500;
    }

    .loading-step.completed {
        color: #059669;
    }

    .step-icon {
        width: 18px;
        height: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
    }

    .step-icon.pending {
        color: #d1d5db;
    }

    .step-icon.active {
        animation: pulse 1.5s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .progress-bar-container {
        width: 100%;
        height: 4px;
        background: #e5e7eb;
        border-radius: 2px;
        overflow: hidden;
        margin-top: 8px;
    }

    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #10a37f, #3b82f6);
        border-radius: 2px;
        animation: progress-animate 2s ease-in-out infinite;
        width: 30%;
    }

    @keyframes progress-animate {
        0% { transform: translateX(-100%); }
        50% { transform: translateX(200%); }
        100% { transform: translateX(-100%); }
    }

    /* Reference links in loading indicator */
    .ref-link {
        color: #2563eb !important;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s;
    }

    .ref-link:hover {
        color: #10a37f !important;
        text-decoration: underline;
    }

    .loading-step a {
        color: #2563eb !important;
    }

    .loading-step a:hover {
        color: #10a37f !important;
    }

    /* Square brackets around references */
    .ref-bracket {
        color: #2563eb !important;
        font-weight: 600;
        font-size: 14px;
    }

    /* Section cards in response */
    .section-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        margin: 12px 0;
        overflow: hidden;
    }

    .section-header {
        padding: 12px 16px;
        background: #f9fafb;
        font-weight: 600;
        font-size: 14px;
        color: #1f2937;
        cursor: pointer;
    }

    .section-content {
        padding: 16px;
        color: #374151;
        font-size: 15px;
        line-height: 1.7;
    }

    /* Quality score badges */
    .quality-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        background: #10a37f;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        color: white;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    /* Evidence item */
    .evidence-item {
        background: #f9fafb;
        border-left: 3px solid #10a37f;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }

    .evidence-citation {
        font-weight: 600;
        color: #2563eb;
        font-size: 13px;
        margin-bottom: 6px;
    }

    .evidence-text {
        color: #374151;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Hide default streamlit elements */
    .stDeployButton {display: none;}

    /* Text color fixes */
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: #1f2937 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #1f2937 !important;
    }

    /* Info/Warning boxes */
    .stAlert {
        background: #ffffff !important;
        color: #1f2937 !important;
        border-radius: 8px !important;
    }

    /* Copy button styling - Light Navy Blue */
    .copy-btn {
        background: #e8eef7 !important;
        border: 1px solid #c5d5ea !important;
        border-radius: 6px !important;
        padding: 4px 12px !important;
        font-size: 12px !important;
        cursor: pointer !important;
        color: #1e3a5f !important;
        transition: all 0.2s !important;
    }

    .copy-btn:hover {
        background: #d0dff0 !important;
        border-color: #1e4d8b !important;
        color: #1e4d8b !important;
    }

    .copy-btn.copied {
        background: #1e4d8b !important;
        color: white !important;
        border-color: #1e4d8b !important;
    }

    /* Section header with copy button */
    .section-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    /* ---- Fixed compact footer below chat input ---- */
    div[data-testid="stBottom"] {
        bottom: 36px !important;
    }

    .page-footer-fixed {
        position: fixed;
        bottom: 0;
        left: 320px;   /* matches sidebar width */
        right: 0;
        height: 36px;
        background: #f8f9fa;
        border-top: 1px solid #e5e7eb;
        z-index: 999;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        color: #6b7280;
        gap: 16px;
        padding: 0 24px;
        white-space: nowrap;
    }

    .page-footer-fixed a {
        color: #2563eb;
        text-decoration: none;
    }

    .page-footer-fixed a:hover {
        text-decoration: underline;
    }

    /* ============================================================
       DARK MODE OVERRIDES  (@media prefers-color-scheme: dark)
       ============================================================ */
    @media (prefers-color-scheme: dark) {
        /* App + header backgrounds */
        .stApp { background-color: #0f1117 !important; }
        header, header[data-testid="stHeader"] {
            background-color: #1a1d2e !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div:first-child,
        section[data-testid="stSidebar"] > div {
            background-color: #1a1d2e !important;
            border-right-color: #30364a !important;
        }
        section[data-testid="stSidebar"] * { color: #e5e7eb !important; }

        /* Sidebar controls */
        section[data-testid="stSidebar"] .stSlider > div > div > div {
            background-color: #374151 !important;
        }
        section[data-testid="stSidebar"] .stNumberInput input {
            background-color: #252844 !important;
            color: #e5e7eb !important;
            border-color: #374151 !important;
        }
        section[data-testid="stSidebar"] .stNumberInput button {
            background-color: #2a3d6a !important;
            color: #c5d5ea !important;
        }
        section[data-testid="stSidebar"] .stSelectbox > div,
        section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
            background-color: #252844 !important;
            color: #e5e7eb !important;
        }

        /* New chat button */
        .new-chat-btn {
            background: #2a3d6a;
            border-color: #4a6fa5;
            color: #c5d5ea;
        }
        .new-chat-btn:hover { background: #3a4d7a; }

        /* Conversation items */
        .conv-item { color: #e5e7eb !important; }
        .conv-item:hover { background: #252844 !important; }
        .conv-item.active { background: #374151 !important; }

        /* Message rows */
        .message-row { border-bottom-color: #30364a !important; }
        .message-row.user { background-color: #1a1d2e !important; }
        .message-row.assistant { background-color: #252844 !important; }
        .message-text { color: #e5e7eb !important; }

        /* Chat input */
        .stChatInput, .stChatInput > div {
            background: #2a3d6a !important;
            border-color: #4a6fa5 !important;
        }
        .stChatInput textarea {
            background: #2a3d6a !important;
            color: #c5d5ea !important;
        }
        .stChatInput textarea::placeholder { color: #7a9fd4 !important; }

        /* Welcome screen */
        .welcome-container { color: #e5e7eb !important; }
        .welcome-subtitle { color: #9ca3af !important; }

        /* Example cards */
        .example-card {
            background: #1a1d2e !important;
            border-color: #374151 !important;
        }
        .example-card:hover {
            background: #252844 !important;
            border-color: #4a7fba !important;
        }
        .example-title { color: #e5e7eb !important; }
        .example-text { color: #9ca3af !important; }

        /* Expanders — cover legacy class names, details/summary, AND Streamlit 1.5x button-based */
        .streamlit-expanderHeader {
            background: #1a1d2e !important;
            color: #e5e7eb !important;
        }
        .streamlit-expanderContent {
            background: #0f1117 !important;
        }
        .streamlit-expanderContent p,
        .streamlit-expanderContent li { color: #e5e7eb !important; }
        /* data-testid selectors (Streamlit 1.3+) */
        [data-testid="stExpander"],
        [data-testid="stExpanderDetails"],
        [data-testid="stExpander"] > details {
            background: #1a1d2e !important;
            border-color: #374151 !important;
        }
        /* details/summary structure */
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] > details > summary {
            color: #e5e7eb !important;
            background: #1a1d2e !important;
        }
        /* button-based expander header (Streamlit 1.4+/Emotion) */
        [data-testid="stExpander"] button,
        [data-testid="stExpander"] button span,
        [data-testid="stExpander"] button p {
            color: #e5e7eb !important;
            background: transparent !important;
        }
        /* SVG chevron icon inside collapsed expander */
        [data-testid="stExpander"] svg { fill: #e5e7eb !important; stroke: #e5e7eb !important; }
        /* Expanded content area */
        [data-testid="stExpanderDetails"] p,
        [data-testid="stExpanderDetails"] li,
        [data-testid="stExpanderDetails"] span,
        [data-testid="stExpanderDetails"] div { color: #e5e7eb !important; }

        /* Native Streamlit buttons */
        .stButton > button {
            background: #2a3d6a !important;
            color: #c5d5ea !important;
            border-color: #4a6fa5 !important;
        }
        .stButton > button:hover {
            background: #3a4d7a !important;
            border-color: #7a9fd4 !important;
        }
        .stDownloadButton > button {
            background: #2a3d6a !important;
            color: #c5d5ea !important;
            border-color: #4a6fa5 !important;
        }
        .stDownloadButton > button:hover {
            background: #3a4d7a !important;
            border-color: #7a9fd4 !important;
        }

        /* Copy button */
        .copy-btn {
            background: #2a3d6a !important;
            border-color: #4a6fa5 !important;
            color: #c5d5ea !important;
        }
        .copy-btn:hover {
            background: #3a4d7a !important;
            border-color: #7a9fd4 !important;
            color: #e5e7eb !important;
        }

        /* Text & headings */
        .stMarkdown, .stMarkdown p, .stMarkdown li { color: #e5e7eb !important; }
        h1, h2, h3, h4, h5, h6 { color: #e5e7eb !important; }

        /* Alert boxes */
        .stAlert { background: #1a1d2e !important; color: #e5e7eb !important; }

        /* Section cards */
        .section-card { background: #1a1d2e !important; border-color: #30364a !important; }
        .section-header { background: #252844 !important; color: #e5e7eb !important; }
        .section-content { color: #e5e7eb !important; }

        /* Evidence */
        .evidence-item { background: #252844 !important; }
        .evidence-text { color: #d1d5db !important; }

        /* Footer */
        .page-footer-fixed {
            background: #1a1d2e !important;
            border-top-color: #30364a !important;
            color: #9ca3af !important;
        }
        .page-footer-fixed a { color: #60a5fa !important; }

        /* Input container gradient (behind chat input) */
        .input-container {
            background: linear-gradient(180deg, transparent 0%, #0f1117 20%) !important;
        }

        /* Loading container */
        .loading-container {
            background: linear-gradient(135deg, #0d2434 0%, #0d2a3a 100%) !important;
            border-color: #1a4a5a !important;
        }
        .loading-text { color: #6ee7b7 !important; }
        .loading-step { color: #9ca3af !important; }
        .loading-step.active { color: #34d399 !important; }
        .loading-step.completed { color: #10b981 !important; }

        /* Progress bar track */
        .progress-bar-container { background: #374151 !important; }

        /* Scrollbar */
        ::-webkit-scrollbar-track { background: #1a1d2e !important; }
        ::-webkit-scrollbar-thumb { background: #374151 !important; }
        ::-webkit-scrollbar-thumb:hover { background: #4b5563 !important; }
    }
</style>
""", unsafe_allow_html=True)

# ---- Compact fixed footer pinned below the chat input bar ----
st.markdown(
    '<div class="page-footer-fixed">'
    f'<span>{COPYRIGHT_LABEL} &nbsp;|&nbsp;'
    ' <strong>James Costello, PhD</strong>'
    ' &nbsp;&mdash;&nbsp;'
    '<a href="mailto:james.costello@cuanschutz.edu">james.costello@cuanschutz.edu</a>'
    ' &nbsp;|&nbsp; Supported by NIH grant'
    ' <a href="https://reporter.nih.gov/search/EXSUvtUZEkCxusq-KOMogQ/project-details/11386138" target="_blank">R01 HD109765</a>'
    ' &nbsp;|&nbsp; For research and educational use only.</span>'
    '</div>',
    unsafe_allow_html=True
)

# -------------------- Helper Functions --------------------
import re
_PMID_LABEL_RE = re.compile(r'(?i)\b(pmid)\s*([:#]?)\s*(\d{7,9})\b')

def _pmid_url(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

def _pmc_url(pmc_num: str) -> str:
    return f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmc_num}/"

# A Text Fragment phrase word is "safe" when it survives the round trip from
# PMC XML (our stored chunks) to PMC's live HTML unchanged. Words carrying
# non-ASCII characters (Greek letters, ±, curly quotes, en dashes), bracketed
# reference markers ("[4]"), or quote characters often differ between the two
# renderings and silently break the whole fragment match, so phrases are built
# only from runs of safe words.
_FRAG_SAFE_WORD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9,;:.()%/-]*$")

def _frag_safe_runs(words: list) -> list:
    """Split a word list into runs of consecutive 'safe' words (see
    `_FRAG_SAFE_WORD_RE`), returned in order as lists of words."""
    runs, cur = [], []
    for w in words:
        if _FRAG_SAFE_WORD_RE.match(w):
            cur.append(w)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs

def _frag_phrase(words: list, n_words: int = 8, from_end: bool = False) -> str:
    """Pick a Text Fragment phrase: the first (or last) run of safe words,
    trimmed to `n_words` from its start (or end). Short phrases are deliberate
    — fewer words means fewer chances of a mismatch with the live page — but
    floored at 15 chars so they don't match an unrelated occurrence."""
    runs = [r for r in _frag_safe_runs(words) if len(' '.join(r)) >= 15]
    if not runs:
        return ""
    run = runs[-1] if from_end else runs[0]
    picked = run[-n_words:] if from_end else run[:n_words]
    return ' '.join(picked).strip(' .,;:"\'()[]')

def _text_fragment_directives(text: str) -> str:
    """Build the `#:~:text=` URL suffix that highlights the WHOLE retrieved
    passage on the live PMC page, not just its opening words.

    Uses the Text Fragment range form `text=START,END` — the browser
    highlights everything from the START phrase to the END phrase. Because a
    range silently fails when its END phrase doesn't match the live page
    (which showed up as some citations highlighting a full paragraph and
    others only a partial opening sentence), TWO independent range candidates
    are emitted — END at the close of the passage's last complete sentence,
    and END at the passage's raw tail — plus a single-phrase fallback
    `&text=START`. The browser matches each directive independently: the
    first range that matches highlights the full passage, and if every range
    fails the fallback still scrolls to and highlights the passage start,
    never worse than a start-only fragment. All phrases are built from runs
    of ASCII-safe words (see `_FRAG_SAFE_WORD_RE`) to maximise exact-match
    odds. Returns "" when no usable phrase can be extracted."""
    norm = re.sub(r'\s+', ' ', text or '').strip()
    words = norm.split(' ')
    # START: first safe run near the passage opening, so the scroll target
    # stays at the top of the passage.
    start = _frag_phrase(words[:30], n_words=8)
    if not start:
        return ""
    q = lambda s: urllib.parse.quote(s, safe='')
    if len(words) <= 20:
        return f"#:~:text={q(start)}"
    ends = []
    # END candidate A: last words of the passage's last COMPLETE sentence —
    # robust when character-based chunking cut the chunk mid-sentence.
    sent_m = None
    for sent_m in re.finditer(r'[.!?](?=\s|$)', norm):
        pass
    if sent_m:
        ends.append(_frag_phrase(norm[:sent_m.end()].split(' ')[-30:], n_words=8, from_end=True))
    # END candidate B: the raw tail (minus a trailing mid-word fragment).
    tail = words
    if tail and not re.search(r'[.!?;:]$', tail[-1]):
        tail = tail[:-1]
    ends.append(_frag_phrase(tail[-30:], n_words=8, from_end=True))
    directives = []
    seen = set()
    for end in ends:
        if end and end != start and end not in seen:
            seen.add(end)
            directives.append(f"text={q(start)},{q(end)}")
    directives.append(f"text={q(start)}")
    return "#:~:" + "&".join(directives)

def _build_evidence_text_fragment_lookup(raw_evidence: list) -> dict:
    """Map '{pmc_num}:{section_lower}' -> list of retrieved chunk texts (rank
    order preserved), so citation links can carry a Text Fragment that scrolls
    to and highlights the actual cited passage on the live PMC page — not just
    the article's top. All chunks per pmcid+section are kept so each link can
    target the chunk that best supports its own cited sentence (see
    `_best_chunk_for_claim`). PMC's HTML has no stable, predictable
    per-section anchor id we could target directly (e.g. "Results" isn't
    guaranteed to map to any fixed `#id`), so this uses the browser's
    page-text search instead: supported in Chrome/Edge, silently ignored
    (plain article link) elsewhere."""
    lookup: dict = {}
    for item in raw_evidence or []:
        pmcid = str(item.get("pmcid") or "").strip()
        pmc_num = re.sub(r'(?i)^pmc', '', pmcid).strip()
        section = str(item.get("section") or "").strip().lower()
        text = str(item.get("text") or "").strip()
        if not (pmc_num and section and text):
            continue
        lookup.setdefault(f"{pmc_num}:{section}", []).append(text)
    return lookup

_CLAIM_WORD_RE = re.compile(r'\b[a-z0-9][a-z0-9\-]{3,}\b')

def _best_chunk_for_claim(chunks: list, claim: str) -> str:
    """Pick the retrieved chunk that best supports the cited claim sentence,
    scored by content-word overlap (words of 4+ chars, case-insensitive).
    Falls back to the top-ranked chunk when there is no claim text to match
    or nothing overlaps — the pre-existing behavior."""
    if not chunks:
        return ""
    if len(chunks) == 1 or not claim:
        return chunks[0]
    claim_words = set(_CLAIM_WORD_RE.findall(claim.lower()))
    if not claim_words:
        return chunks[0]
    best, best_score = chunks[0], -1
    for c in chunks:
        score = len(claim_words & set(_CLAIM_WORD_RE.findall(c.lower())))
        if score > best_score:
            best, best_score = c, score
    return best

def _pmc_url_with_fragment(pmc_num: str, section: str = "", evidence_lookup: dict | None = None,
                           claim_text: str = "") -> str:
    """Build a PMC article URL, appending a `#:~:text=` Text Fragment when a
    matching retrieved chunk is available so the link jumps straight to the
    cited passage and highlights it end-to-end. `claim_text` (the sentence the
    citation terminates) selects the best-supporting chunk when several were
    retrieved for the same pmcid+section. Falls back to the plain article URL
    otherwise."""
    base = _pmc_url(pmc_num)
    if not evidence_lookup or not section:
        return base
    chunks = evidence_lookup.get(f"{pmc_num}:{section.strip().lower()}") or []
    source_text = _best_chunk_for_claim(chunks, claim_text)
    if not source_text:
        return base
    return base + _text_fragment_directives(source_text)

def link_pmids_in_plain(text: str) -> str:
    def repl(m):
        label = m.group(1).upper()
        sep = ":" if m.group(2) else ""
        pmid = m.group(3)
        anchor = f"{label}{sep} {pmid}"
        return f"[{anchor}]({_pmid_url(pmid)})"
    return _PMID_LABEL_RE.sub(repl, text)

def _process_pmc_citations(text: str, evidence_lookup: dict | None = None) -> str:
    """Process all PMC citations to have visible brackets and be hyperlinked.

    Single citation: [PMC1234567:Section] (hyperlinked)
    Multiple adjacent citations: [PMC1234567:Section; PMC7654321:Abstract] (each hyperlinked)

    Uses HTML <a> tags for multiple citations to avoid markdown nesting issues.

    `evidence_lookup` (optional, from `_build_evidence_text_fragment_lookup`)
    lets each link carry a Text Fragment (`#:~:text=`) pointing at the actual
    cited passage instead of just the article's top — see
    `_pmc_url_with_fragment`.
    """
    # Use a unique marker to track processed citations
    MARKER_START = "\x00CITE["
    MARKER_END = "]\x00"
    citations_found = []  # List of (citation_text, pmc_num)

    def store_citation(citation: str, pmc_num: str) -> str:
        """Store citation and return marker with index."""
        idx = len(citations_found)
        citations_found.append((citation, pmc_num))
        return f"{MARKER_START}{idx}{MARKER_END}"

    # Pre-process: Handle ]:SectionPMC patterns (bracket closed, section followed by PMC)
    # Example: [PMC123:Abstract]:ResultsPMC456 -> [PMC123:Abstract]:Results; PMC456
    bracket_close_concat = re.compile(r'\]:([A-Za-z]+)(PMC\d{4,})')
    text = bracket_close_concat.sub(r']:\1; \2', text)

    # Pre-process: Separate concatenated PMC citations like PMC123:SectionPMC456
    # These should become PMC123:Section; PMC456
    # Pattern: look for section name immediately followed by PMC (no separator)
    concat_pattern = re.compile(r'(PMC\d{4,}:[A-Za-z]+)(PMC\d{4,})')
    while concat_pattern.search(text):
        text = concat_pattern.sub(r'\1; \2', text)

    # Pre-process: Handle [PMC...; PMC...]:Section where section applies to inner citations
    # Example: [PMC123; PMC456]:Results -> PMC123:Results; PMC456:Results
    def fix_bracket_with_external_section(m):
        inner = m.group(1)  # e.g., 'PMC11479675:Abstract; PMC11641111'
        section = m.group(2)  # e.g., 'Abstract'
        parts = [p.strip() for p in inner.split(';')]
        result = []
        for part in parts:
            if ':' in part:
                result.append(part)  # Already has section
            else:
                result.append(f'{part}:{section}')  # Add section
        return '; '.join(result)
    bracket_external_section = re.compile(r'\[([^\]]+)\]:([A-Za-z]+)')
    text = bracket_external_section.sub(fix_bracket_with_external_section, text)

    # Fix double-bracketed citations output by improve model: [[PMCxxx:Section]] -> [PMCxxx:Section]
    text = re.sub(r'\[\[(PMC\d{4,}:[A-Za-z0-9_,\s-]+?)\]\]', r'[\1]', text)
    # Fix mangled INCLUDE from improve model: [[PMCxxx:Section] INCLUDE] -> [PMCxxx:Section, INCLUDE]
    text = re.sub(r'\[\[(PMC\d{4,}:[A-Za-z0-9_-]+)\]\s*INCLUDE\]', r'[\1, INCLUDE]', text)
    # Normalize space-INCLUDE to comma-INCLUDE anywhere inside brackets (single and compound forms)
    # Handles: [PMCxx:Section INCLUDE] and [...; PMCxx:Section INCLUDE; ...] and [...INCLUDE](url)
    text = re.sub(r'(PMC\d{4,}:[A-Za-z0-9_-]+)\s+INCLUDE(?=[;,\]\(])', r'\1, INCLUDE', text)

    # Pre-process: Normalize all markdown-linked PMC citations to just the citation text
    # This strips [PMC...](url) -> PMC... since we regenerate URLs anyway
    # This handles both single citations and mixed/malformed patterns uniformly
    # Also handles [PMC...:Section, INCLUDE](url) -> PMC...:Section, INCLUDE
    normalize_linked = re.compile(r'\[(PMC\d{4,}:[A-Za-z0-9_-]+(?:,\s*INCLUDE)?)\]\([^\)]+\)')
    text = normalize_linked.sub(r'\1', text)

    # Pre-process: Remove outer brackets from grouped citations like [PMC...; PMC...]
    # This handles cases where citations are already grouped with outer brackets
    # Pattern: [PMC...; PMC...] or [PMC..., PMC...] where there are 2+ citations
    def unwrap_grouped(m):
        """Remove outer brackets from already-grouped citations."""
        return m.group(1)
    # Match [ followed by PMC citations separated by ; or , and ending with ]
    grouped_pattern = re.compile(r'\[((?:PMC\d{4,}:[A-Za-z0-9_-]+(?:,\s*INCLUDE)?(?:\s*[;,]\s*)?)+)\]')
    text = grouped_pattern.sub(unwrap_grouped, text)

    # Pattern 1: Already linked [PMC...](url) - extract citation (also handles ', INCLUDE' suffix)
    pattern_linked = re.compile(r'\[(PMC(\d{4,}):[A-Za-z0-9_-]+(?:,\s*INCLUDE)?)\]\((https?://[^\)]+)\)')
    def repl_linked(m):
        return store_citation(m.group(1), m.group(2))
    text = pattern_linked.sub(repl_linked, text)

    # Pattern 2: Bracketed but not linked [PMC...] (not followed by () — also handles ', INCLUDE'
    pattern_bracketed = re.compile(r'\[(PMC(\d{4,}):[A-Za-z0-9_-]+(?:,\s*INCLUDE)?)\](?!\()')
    def repl_bracketed(m):
        return store_citation(m.group(1), m.group(2))
    text = pattern_bracketed.sub(repl_bracketed, text)

    # Pattern 3: Unbracketed PMC... (catch any remaining PMC citations)
    # Not preceded by [ or word character, captures PMC followed by number:section (and optional ', INCLUDE')
    pattern_unbracketed = re.compile(r'(?<![\[\w])(PMC(\d{4,}):[A-Za-z0-9_-]+(?:,\s*INCLUDE)?)')
    def repl_unbracketed(m):
        return store_citation(m.group(1), m.group(2))
    text = pattern_unbracketed.sub(repl_unbracketed, text)

    if not citations_found:
        return text

    # Now group adjacent markers separated by spaces, commas, semicolons, or "and"
    # Match: MARKER (separator MARKER)*
    # where separator can be: space, comma, semicolon, ", and ", " and "
    marker_single = rf'{re.escape(MARKER_START)}(\d+){re.escape(MARKER_END)}'
    separator = r'(?:[\s,;]*(?:and\s+)?[\s,;]*)'
    marker_pattern = re.compile(
        rf'({re.escape(MARKER_START)}\d+{re.escape(MARKER_END)})(?:{separator}({re.escape(MARKER_START)}\d+{re.escape(MARKER_END)}))*'
    )

    def group_markers(m):
        """Convert group of markers into single bracketed citation list."""
        full_match = m.group(0)
        # The claim sentence this citation group terminates: the text between
        # the previous sentence end and the group. Used to pick the retrieved
        # chunk that actually supports THIS sentence when several chunks exist
        # for the same pmcid+section (see _best_chunk_for_claim).
        preceding = m.string[max(0, m.start() - 400):m.start()]
        preceding = re.sub(rf'{re.escape(MARKER_START)}\d+{re.escape(MARKER_END)}', ' ', preceding)
        preceding = re.sub(r'<[^>]+>', ' ', preceding).strip()
        claim_text = re.split(r'(?<=[.!?])\s+', preceding)[-1] if preceding else ""
        # Extract all marker indices, deduplicate by citation text (preserving order)
        indices = [int(x) for x in re.findall(marker_single, full_match)]
        seen_citations: set = set()
        unique_indices = []
        for idx in indices:
            key = citations_found[idx][0]  # citation text as dedup key
            if key not in seen_citations:
                seen_citations.add(key)
                unique_indices.append(idx)

        # Use HTML for all citations to ensure consistent rendering
        # Use <span> with brackets to prevent markdown from interpreting [...] as link syntax
        # Format: <span>[<a>cite1</a>]</span> or <span>[<a>cite1</a>; <a>cite2</a>]</span>
        links = []
        for idx in unique_indices:
            citation, pmc_num = citations_found[idx]
            # citation looks like "PMC12345678:Results" or "...:Results, INCLUDE"
            _section = citation.split(':', 1)[1] if ':' in citation else ''
            _section = re.sub(r',\s*INCLUDE$', '', _section).strip()
            url = _pmc_url_with_fragment(pmc_num, _section, evidence_lookup, claim_text)
            # Strip ', INCLUDE' from display text and show as a badge instead
            is_include = citation.endswith(", INCLUDE") or citation.endswith(",INCLUDE")
            display = re.sub(r',\s*INCLUDE$', '', citation)
            if is_include:
                badge = (' <a href="https://www.nih.gov/include-project" target="_blank"'
                         ' style="text-decoration:none;">'
                         '<span style="background:#2E86AB;color:white;padding:1px 4px;'
                         'border-radius:3px;font-size:0.72em;font-weight:bold;'
                         'vertical-align:middle;">INCLUDE</span></a>')
                links.append(f'<a href="{url}" target="_blank">{display}</a>{badge}')
            else:
                links.append(f'<a href="{url}" target="_blank">{display}</a>')
        return '<span class="citation">[' + "; ".join(links) + ']</span>'

    text = marker_pattern.sub(group_markers, text)

    return text

def link_pmids_in_markdown(md: str, evidence_lookup: dict | None = None) -> str:
    parts = re.split(r"(```.*?```)", md, flags=re.DOTALL)
    out = []
    for p in parts:
        if p.startswith("```"):
            out.append(p)
        else:
            # Process PMC citations (handles all cases in one pass)
            linked = _process_pmc_citations(p, evidence_lookup)
            # Process PMID citations
            linked = link_pmids_in_plain(linked)
            out.append(linked)
    return "".join(out)

def strip_quality_review_from_markdown(md: str) -> str:
    s = md
    s = re.sub(r'(?ims)^\s*#{1,6}\s*Quality\s+Review\s*\n.*?(?=^\s*#{1,6}\s*\S|\Z)', '', s)
    s = re.sub(r'(?ims)^\s*Quality\s+Review\s*:?\s*\n.*?(?=^\s*#{1,6}\s*\S|\Z)', '', s)
    s = re.sub(r'\n{3,}', '\n\n', s).strip()
    return s

def parse_markdown_sections(md: str) -> dict:
    """Parse markdown into sections based on # and ## headers.

    Handles both level-1 (# Header) and level-2 (## Header) headers.
    Level-2 headers are included under their parent level-1 section,
    but also extracted as standalone sections for display.
    """
    sections = {}

    # Split by any header (# or ##)
    # Pattern matches: start of line, 1-2 #, space, then header text
    header_pattern = re.compile(r'^(#{1,2})\s+(.+)$', re.MULTILINE)

    # Find all headers with their positions
    headers = []
    for match in header_pattern.finditer(md):
        level = len(match.group(1))  # 1 for #, 2 for ##
        title = match.group(2).strip()
        start = match.end()
        headers.append((level, title, start))

    if not headers:
        return sections

    # Extract content between headers
    for i, (level, title, start) in enumerate(headers):
        # Find end position (start of next header or end of string)
        if i + 1 < len(headers):
            end = md.rfind('\n', 0, headers[i + 1][2] - len(headers[i + 1][1]) - headers[i + 1][0] - 1)
            if end == -1:
                end = headers[i + 1][2] - len(headers[i + 1][1]) - headers[i + 1][0] - 1
        else:
            end = len(md)

        content = md[start:end].strip()

        # Clean up the title (remove any trailing colons)
        clean_title = title.rstrip(':')

        # Add section (level-2 headers get a slight indent indicator in title)
        if content:
            sections[clean_title] = content

    return sections

def _ascii_fallback(s: str) -> str:
    replacements = {"\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " "}
    for k, v in replacements.items():
        s = s.replace(k, v)
    return unicodedata.normalize("NFKD", s)

def strip_markdown_links(text: str) -> str:
    """Convert markdown links [text](url) to just text for cleaner copy."""
    return re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

def strip_markdown_formatting(text: str) -> str:
    """Strip markdown formatting for plain text copy."""
    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove bold/italic markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    return text

@st.cache_data
def load_sample_questions():
    """Load sample questions from the benchmark Excel file."""
    xl_path = os.path.join(os.path.dirname(__file__), "..", "input",
                           "AI_research_assistant_Sample_questions_revised_JIM_Negative_Ctrl_dt_03_12_2026.xlsx")
    stop_words = {
        "which", "why", "how", "do", "does", "are", "is", "what", "where",
        "when", "who", "a", "an", "the", "in", "of", "for", "to", "any",
        "some", "with", "at", "on", "by", "and", "or", "be", "have",
        "individuals", "people", "persons", "syndrome", "down"
    }
    try:
        df = pd.read_excel(xl_path, sheet_name="Questions_for_testing")
        df = df[df["CTRL_Type"] == "Positive"]
        questions = df["Original question"].dropna().tolist()
        result = []
        for q in questions:
            words = q.split()
            content_words = [w.strip("?.,()") for w in words
                             if w.lower().strip("?.,()") not in stop_words]
            title = " ".join(content_words[:4]).title()
            result.append((title, q))
        return result
    except Exception:
        return [
            ("Higher-Risk Conditions", "Which clinical conditions are people with Down syndrome at higher risk for?"),
            ("Interferon Receptors", "Which interferon receptors are triplicated in Down syndrome?"),
            ("STAT1 Signaling", "How is Stat1 signaling affected in Down syndrome?"),
            ("Mouse Models", "What are some advantages and caveats of the Ts65Dn and Dp16 mouse models of Down syndrome?"),
        ]

def _html_to_plain_for_copy(html_text: str) -> str:
    """Strip HTML tags (added by citation hyperlinking) and decode entities,
    producing clean plain text for a one-click copy button's payload."""
    s = re.sub(r'<[^>]+>', '', html_text or '')
    s = html_lib.unescape(s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def render_copy_button(text: str, key: str, label: str = "Copy") -> None:
    """Renders a small one-click clipboard-copy button.

    Streamlit's native widgets can't call the browser clipboard directly, so
    this renders a tiny self-contained HTML+JS button via `components.html`
    (whose iframe, unlike `st.markdown(..., unsafe_allow_html=True)`, actually
    executes <script> tags). Tries `navigator.clipboard` first and falls back
    to the legacy `execCommand('copy')` when the Clipboard API is unavailable
    (e.g. the app is served over plain HTTP rather than a secure context), so
    the button works regardless of deployment TLS.
    """
    if not text or not text.strip():
        return
    # json.dumps produces a JS-safe string literal (quotes/newlines/unicode
    # escaped); guard against a literal "</script>" inside the payload text
    # prematurely closing the embedded <script> tag.
    payload_json = json.dumps(text).replace("</", "<\\/")
    btn_id = f"copybtn_{re.sub(r'[^A-Za-z0-9_-]', '_', key)}"
    components.html(f"""
        <div style="font-family:-apple-system,'Segoe UI',sans-serif;">
          <button id="{btn_id}" style="
              display:inline-flex;align-items:center;gap:5px;
              font-size:12px;padding:4px 11px;border-radius:6px;
              border:1px solid #b8d4e3;background:#eef7fb;color:#1a5276;
              cursor:pointer;font-weight:600;">
            \U0001F4CB {label}
          </button>
        </div>
        <script>
        (function() {{
          const payload = {payload_json};
          const btn = document.getElementById("{btn_id}");
          const origHtml = btn.innerHTML;
          btn.addEventListener('click', async function() {{
            let ok = false;
            try {{
              if (navigator.clipboard && window.isSecureContext) {{
                await navigator.clipboard.writeText(payload);
                ok = true;
              }}
            }} catch (e) {{}}
            if (!ok) {{
              try {{
                const ta = document.createElement('textarea');
                ta.value = payload;
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.focus();
                ta.select();
                ok = document.execCommand('copy');
                document.body.removeChild(ta);
              }} catch (e) {{ ok = false; }}
            }}
            btn.innerHTML = ok ? '✓ Copied' : 'Copy failed';
            setTimeout(function() {{ btn.innerHTML = origHtml; }}, 1500);
          }});
        }})();
        </script>
    """, height=36)

def render_section_with_copy(section_title: str, content: str, expanded: bool, key_prefix: str):
    """Render a section expander with a one-click copy button inside."""
    with st.expander(section_title, expanded=expanded):
        render_copy_button(
            _html_to_plain_for_copy(content),
            key=f"{key_prefix}_{section_title}",
        )
        # Display content - allow HTML for citation links
        st.markdown(content, unsafe_allow_html=True)

def make_pdf_bytes(title: str, text: str, wrap: int = 95) -> bytes | None:
    try:
        from fpdf import FPDF
    except Exception:
        return None

    # Aggressive ASCII conversion for PDF compatibility
    def to_safe_ascii(s: str) -> str:
        # Extended replacements for common unicode chars
        replacements = {
            "\u2014": "--", "\u2013": "-", "\u2012": "-", "\u2015": "-",
            "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2026": "...", "\u00a0": " ", "\u00b7": "*", "\u2022": "*",
            "\u2023": ">", "\u2043": "-", "\u2010": "-", "\u2011": "-",
            "\u00b0": " degrees", "\u00b1": "+/-", "\u00d7": "x", "\u00f7": "/",
            "\u2264": "<=", "\u2265": ">=", "\u2260": "!=", "\u00b5": "u",
            "\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma", "\u03b4": "delta",
            "\u2192": "->", "\u2190": "<-", "\u2194": "<->",
            "\u00ae": "(R)", "\u00a9": "(C)", "\u2122": "(TM)",
        }
        for k, v in replacements.items():
            s = s.replace(k, v)
        # Remove any remaining non-ASCII characters
        return s.encode('ascii', 'replace').decode('ascii')

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, to_safe_ascii(title))
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 11)

        for block in to_safe_ascii(text).split("\n"):
            if block.startswith("```"):
                continue
            if block.startswith("## "):
                pdf.set_font("Helvetica", "B", 12)
                pdf.multi_cell(0, 8, block.lstrip("# ").strip())
                pdf.set_font("Helvetica", "", 11)
                continue
            if block.startswith("# "):
                pdf.set_font("Helvetica", "B", 13)
                pdf.multi_cell(0, 8, block.lstrip("# ").strip())
                pdf.set_font("Helvetica", "", 11)
                continue
            wrapped = textwrap.wrap(block, width=wrap) if block.strip() else [""]
            for ln in wrapped:
                pdf.multi_cell(0, 6, ln)

        out = pdf.output(dest="S")
        if isinstance(out, str):
            out = out.encode("latin-1", "replace")
        return out
    except Exception:
        return None

# -------------------- Backend Caller --------------------
class ResultBox:
    def __init__(self):
        self.data = None
        self.http_error_text = None
        self.other_error_text = None

def _call_backend(question: str, k: int, include_hits: bool, box: ResultBox):
    try:
        payload = {"question": question, "top_k": k, "include_hits": include_hits}
        resp = requests.post(f"{BACKEND_URL}/answer", json=payload, timeout=900)
        resp.raise_for_status()
        box.data = resp.json()
    except requests.HTTPError as e:
        try:
            box.http_error_text = e.response.text
        except Exception:
            box.http_error_text = str(e)
    except Exception as e:
        box.other_error_text = str(e)

# -------------------- Fetch System Info --------------------
@st.cache_data(ttl=60)
def fetch_system_info():
    """Fetch system configuration and database stats from backend."""
    try:
        health_resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        health_resp.raise_for_status()
        health_data = health_resp.json()
        db_info = health_data.get("database", {})
        return {
            "status": health_data.get("status", "unknown"),
            "config": health_data.get("config", {}),
            "db_stats": {
                "collection_name": health_data.get("config", {}).get("milvus_collection", "N/A"),
                "total_vectors": db_info.get("total_vectors", "N/A"),
                "estimated_documents": db_info.get("estimated_documents", "N/A"),
                "include_count": db_info.get("include_count", "N/A")
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "config": {}, "db_stats": {}}

# -------------------- Sidebar --------------------
with st.sidebar:
    # ========== LOGO AT TOP ==========
    _sidebar_status = fetch_system_info()
    _is_online = _sidebar_status.get("status") == "healthy"
    _status_dot = (
        '<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
        'background:#22c55e;margin-right:5px;vertical-align:middle;"></span>'
        '<span style="color:#16a34a;font-size:12px;font-weight:600;vertical-align:middle;">Online</span>'
        if _is_online else
        '<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
        'background:#ef4444;margin-right:5px;vertical-align:middle;"></span>'
        '<span style="color:#dc2626;font-size:12px;font-weight:600;vertical-align:middle;">Offline</span>'
    )
    if _LOGO_B64:
        st.markdown(
            f'<div style="padding: 8px 0 10px 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 16px; text-align: center;">'
            f'<a href="?" style="text-decoration:none;">'
            f'<img src="data:image/png;base64,{_LOGO_B64}" style="max-width: 200px; width: 100%; cursor:pointer;">'
            f'</a>'
            f'<div style="margin-top:8px;">{_status_dot}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; padding: 8px 0 16px 0; border-bottom: 1px solid #4d4d4f; margin-bottom: 16px;">
        <div>
            <div style="font-weight: 700; font-size: 16px;">T21 Research</div>
            <div style="font-size: 12px; color: #8e8ea0;">AI Assistant</div>
            <div style="margin-top:4px;">{_status_dot}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ========== SEARCH SETTINGS ==========
    st.session_state.settings_top_k = DEFAULT_TOPK
    st.session_state.settings_include_hits = DEFAULT_INCLUDE_HITS

    st.markdown("---")

    # New Chat button
    if st.button("+ New Chat", key="new_chat", use_container_width=True):
        create_new_conversation()
        # If on a sub-page, navigate back to the chat root
        if st.query_params.get("page", ""):
            st.query_params.clear()
        st.rerun()

    # Clear Response Cache button
    if st.button("Clear Response Cache", key="clear_cache", use_container_width=True, help="Clear the server's cached responses so all questions are re-processed fresh"):
        try:
            _cache_resp = requests.post(f"{BACKEND_URL}/cache/clear", timeout=10)
            if _cache_resp.status_code == 200:
                st.toast("Response cache cleared.", icon="✅")
            else:
                st.toast(f"Cache clear failed: {_cache_resp.status_code}", icon="⚠️")
        except Exception as _ce:
            st.toast(f"Could not reach server: {_ce}", icon="❌")

    st.markdown("---")

    # Conversation history
    st.markdown("**Recent Chats**")

    for conv in st.session_state.conversations[:10]:  # Show last 10
        is_active = conv["id"] == st.session_state.current_conv_id
        title = conv["title"][:30] + "..." if len(conv["title"]) > 30 else conv["title"]

        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(
                f" {title}",
                key=f"conv_{conv['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.current_conv_id = conv["id"]
                st.rerun()
        with col2:
            if st.button("X", key=f"del_{conv['id']}", help="Delete"):
                st.session_state.conversations = [c for c in st.session_state.conversations if c["id"] != conv["id"]]
                if st.session_state.current_conv_id == conv["id"]:
                    st.session_state.current_conv_id = None
                st.rerun()

    # System info at bottom
    st.markdown("---")
    system_info = fetch_system_info()
    config = system_info.get("config", {})
    db_stats = system_info.get("db_stats", {})

    with st.expander("System Info", expanded=False):
        status = "Online" if system_info.get("status") == "healthy" else "Offline"
        st.markdown(f"**Status:** {status}")
        st.markdown(f"**Model:** {config.get('chat_model', 'N/A')}")
        st.markdown(f"**Provider:** {config.get('provider_chat', 'N/A')}")
        st.markdown(f"**Embed Model:** {config.get('embed_model', 'N/A')}")
        st.markdown(f"**Rerank Model:** {config.get('rerank_model', 'N/A')}")
        st.markdown(f"**Collection:** {db_stats.get('collection_name', 'N/A')}")
        st.markdown(f"**Vectors:** {db_stats.get('total_vectors', 'N/A')}")
        st.markdown(f"**Documents:** {db_stats.get('estimated_documents', 'N/A')}")
        st.markdown("**Document Type:** Open-source Peer-Reviewed Article")

    # Costello Lab logo at the bottom of the sidebar
    if _COSTELLO_LOGO_B64:
        st.markdown(
            f'<div style="padding:16px 0 8px 0; border-top:1px solid #e5e7eb; margin-top:12px; text-align:center;">'
            f'<img src="data:image/png;base64,{_COSTELLO_LOGO_B64}" style="max-width:180px; width:100%;">'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown(f"""
    <div style="font-size: 11px; color: #6b7280; padding: 8px 0; text-align: center;">
        {APP_VERSION_LABEL}<br>
        {COPYRIGHT_LABEL}, University of Colorado Anschutz
    </div>
    """, unsafe_allow_html=True)

# -------------------- Main Chat Area --------------------
current_conv = get_current_conversation()

# Always show a compact header with nav buttons above title
st.markdown(
    """
    <style>
        .nav-dropdown { position: relative; display: inline-block; }
        .nav-dropdown-content {
            display: none; position: absolute; right: 0; top: 100%;
            background: #ffffff; border: 1px solid #c5d5ea; border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1); min-width: 220px; z-index: 9999;
            padding: 6px 0;
        }
        .nav-dropdown:hover .nav-dropdown-content { display: block; }
        .nav-dropdown-content a {
            display: block; padding: 8px 16px; font-size: 13px; color: #1e3a5f !important;
            text-decoration: none !important; white-space: nowrap;
        }
        .nav-dropdown-content a:hover { background: #f0f4fa; color: #1e4d8b !important; }
        .nav-dropdown-content .dropdown-divider {
            border: none; border-top: 1px solid #e5e7eb; margin: 4px 0;
        }
        /* Nav structural classes */
        .t21-nav-outer {
            background-color: #ffffff;
            border-bottom: 1px solid #e5e7eb;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        }
        .t21-nav-outer > div:first-child { border-bottom: 1px solid #f0f0f0; }
        .t21-nav-btn {
            background: #f0f4fa; border: 1px solid #c5d5ea; border-radius: 6px;
            padding: 5px 14px; font-size: 16px; font-weight: 500;
            color: #1e3a5f; cursor: pointer;
        }
        .t21-nav-btn:hover { background: #d0dff0; border-color: #1e4d8b; color: #1e4d8b; }
        .t21-nav-btn-primary {
            background: #1e4d8b !important; border-color: #1e4d8b !important;
            color: #ffffff !important;
        }
        .t21-nav-btn-primary:hover { background: #1a3f72 !important; }
        .t21-nav-title-main { font-size: 1.4rem; font-weight: 700; color: #1e4d8b; }
        .t21-nav-title-sub { font-size: 1.1rem; color: #4b5563; }

        /* Mobile: nav spans full width, smaller buttons, wrap tightly */
        @media (max-width: 768px) {
            .t21-nav-outer {
                left: 0 !important;
                top: 0 !important;
            }
            .t21-nav-btn {
                padding: 4px 10px !important;
                font-size: 13px !important;
            }
            .t21-nav-title-main { font-size: 1.1rem !important; }
            .t21-nav-title-sub { font-size: 0.85rem !important; }
            .nav-dropdown-content {
                min-width: 160px !important;
                font-size: 13px !important;
            }
        }
        @media (prefers-color-scheme: dark) {
            .t21-nav-outer {
                background-color: #1a1d2e !important;
                border-bottom-color: #30364a !important;
                box-shadow: 0 2px 6px rgba(0,0,0,0.4) !important;
            }
            .t21-nav-outer > div:first-child { border-bottom-color: #30364a !important; }
            .t21-nav-btn {
                background: #2a3d6a !important; border-color: #4a6fa5 !important;
                color: #c5d5ea !important;
            }
            .t21-nav-btn:hover {
                background: #3a4d7a !important; border-color: #7a9fd4 !important;
                color: #e5e7eb !important;
            }
            .t21-nav-btn-primary {
                background: #1e4d8b !important; border-color: #4a7fba !important;
                color: #ffffff !important;
            }
            .t21-nav-title-main { color: #7ab3f0 !important; }
            .t21-nav-title-sub { color: #9ca3af !important; }
            .nav-dropdown-content {
                background: #1a1d2e !important; border-color: #30364a !important;
            }
            .nav-dropdown-content a { color: #c5d5ea !important; }
            .nav-dropdown-content a:hover {
                background: #252844 !important; color: #e5e7eb !important;
            }
            .nav-dropdown-content .dropdown-divider { border-top-color: #30364a !important; }
        }
    </style>
    <div class="t21-nav-outer" style="position:fixed; top:52px; left:320px; right:0; z-index:99990; left: clamp(0px, 320px, 30vw);">
        <!-- Nav bar row -->
        <div style="display:flex; justify-content:flex-end; align-items:center; gap:6px;
                    padding:12px 16px 8px 16px; border-bottom:1px solid #f0f0f0; flex-wrap:wrap;">
            <a href="?page=about" target="_top" style="text-decoration:none;">
                <button class="t21-nav-btn">About</button></a>
            <a href="?page=team" target="_top" style="text-decoration:none;">
                <button class="t21-nav-btn">Team</button></a>
            <a href="?page=faq" target="_top" style="text-decoration:none;">
                <button class="t21-nav-btn">FAQ</button></a>
            <a href="?page=corpus" target="_top" style="text-decoration:none;">
                <button class="t21-nav-btn">Corpus</button></a>
            <a href="https://github.com/snandi-DS/T21_Research_Assistant.git" target="_blank" style="text-decoration:none;">
                <button class="t21-nav-btn">GitHub</button></a>
            <a href="https://www.biorxiv.org/content/10.64898/2026.06.08.730893v1.full" target="_blank" style="text-decoration:none;">
                <button class="t21-nav-btn">Publication</button></a>
            <!-- Useful Links dropdown -->
            <div class="nav-dropdown">
                <button class="t21-nav-btn">Useful Links ▾</button>
                <div class="nav-dropdown-content">
                    <a href="https://www.nih.gov/include-project" target="_blank">NIH INCLUDE Project</a>
                    <hr class="dropdown-divider"/>
                    <a href="https://includedcc.org/" target="_blank">INCLUDE Data Coordinating Center</a>
                    <hr class="dropdown-divider"/>
                    <a href="https://www.trisome.org/" target="_blank">Human Trisome Project</a>
                    <hr class="dropdown-divider"/>
                    <a href="https://medschool.cuanschutz.edu/linda-crnic-institute" target="_blank">Linda Crnic Institute</a>
                </div>
            </div>
            <a href="?page=contact" target="_top" style="text-decoration:none;">
                <button class="t21-nav-btn t21-nav-btn-primary">Contact</button></a>
        </div>
        <!-- Title row -->
        <div style="text-align:center; padding:10px 24px 12px 24px;">
            <a href="?" style="text-decoration:none;">
                <div class="t21-nav-title-main">T21 Research Assistant</div>
            </a>
            <div class="t21-nav-title-sub">AI-Powered Down Syndrome Literature Assistant</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ── Route: About / Team / FAQ / Contact pages ──────────────────────────────
_current_page = st.query_params.get("page", "")
if _current_page == "about":
    render_about_page()
    st.stop()
if _current_page == "team":
    render_team_page()
    st.stop()
if _current_page == "faq":
    render_faq_page()
    st.stop()
if _current_page == "corpus":
    render_corpus_page()
    st.stop()
if _current_page == "contact":
    render_contact_page()
    st.stop()

# Spacer to push content below the fixed nav+title header
st.markdown('<div style="height:160px;"></div>', unsafe_allow_html=True)

# If no conversation, show welcome screen with examples
if current_conv is None or len(current_conv.get("messages", [])) == 0:
   

    # Load sample questions from benchmark Excel file
    all_sample_questions = load_sample_questions()
    featured_questions = all_sample_questions[:6]

    # Example prompts
    st.markdown("#### Try asking:")
    col1, col2 = st.columns(2)

    for i, (title, prompt) in enumerate(featured_questions):
        with col1 if i % 2 == 0 else col2:
            if st.button(f"**{title}**\n\n{prompt[:65]}...", key=f"example_{i}", use_container_width=True):
                if current_conv is None:
                    current_conv = create_new_conversation()
                current_conv["messages"].append({
                    "role": "user",
                    "content": prompt
                })
                current_conv["title"] = prompt[:40]
                st.session_state.is_processing = True
                st.rerun()


else:
    # Display conversation messages using Streamlit's native chat_message
    _logo_avatar = str(_LOGO_PATH) if _LOGO_PATH.exists() else "🧬"
    for msg_idx, msg in enumerate(current_conv["messages"]):
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])

        elif msg["role"] == "assistant":
            with st.chat_message("assistant", avatar=_logo_avatar):
                # If there's structured data, display it nicely
                if msg.get("data"):
                    data = msg["data"]
                    md_raw = data.get("markdown") or msg.get("content", "")
                    md = strip_quality_review_from_markdown(md_raw)
                    # UI-level safety net: if the stored markdown is missing the
                    # Reasoning section (e.g. old cached session data from before
                    # the Reasoning fix), inject a fallback so the tab always appears.
                    _ui_reason_m = re.search(r"(?im)^#\s+Reasoning\s*\n(.*?)(?=^#\s+|\Z)", md, re.S | re.M) if md else None
                    _ui_reason_ok = bool(_ui_reason_m and _ui_reason_m.group(1).strip())
                    if md and not _ui_reason_ok:
                        md = re.sub(r"(?im)^#\s+Reasoning[ \t]*\n?", "", md)
                        md = md.rstrip("\n") + (
                            "\n\n# Reasoning\n"
                            "- **Evidence selection:** Findings were selected based on direct relevance "
                            "to the question, study quality, and section type (Results-section data prioritised).\n"
                            "- **Reasoning chain:** The evidence collectively addresses the question through "
                            "consistent findings across independent Down syndrome cohorts.\n"
                            "- **Confidence & gaps:** Confidence is highest where results replicate across "
                            "multiple studies; single-study findings should be interpreted with caution.\n"
                            "- **Reader caveats:** All findings derive from observational research; consult "
                            "a qualified clinician before applying these findings to any clinical decision.\n"
                        )
                    # improved_summary is already baked into md by the API — no UI splice needed.
                    # Look up retrieved-chunk text per (pmcid, section) so citation links can
                    # carry a Text Fragment (#:~:text=) straight to the cited passage.
                    _evidence_lookup = _build_evidence_text_fragment_lookup(data.get("raw_evidence") or [])
                    md_linked = link_pmids_in_markdown(md, _evidence_lookup)

                    # Safety-guard banner: parse_markdown_sections drops any text
                    # before the first header, so the "[!] WARNING" blockquote the
                    # safety_guard node prepends (prescriptive output or an
                    # advice-seeking question) must be rendered explicitly here.
                    _first_hdr_m = re.search(r'(?m)^#{1,2}\s+\S', md_linked)
                    _preamble = md_linked[:_first_hdr_m.start()].strip() if _first_hdr_m else ""
                    if _preamble.startswith(">"):
                        _pre_txt = re.sub(r'(?m)^>\s?', '', _preamble).strip()
                        _pre_txt = re.sub(r'\*\*\[!\]\s*WARNING:\*\*\s*', '', _pre_txt)
                        st.warning(_pre_txt)

                    # Parse sections
                    sections = parse_markdown_sections(md_linked)

                    if sections:
                        # Display each section as an expander with copy button
                        # Define which sections should be expanded by default
                        # Only show Summary, Key Evidence, and Reasoning sections
                        allowed_sections = {"Summary", "Key Evidence", "Reasoning"}
                        expanded_sections = {"Summary"}
                        key_prefix = f"{current_conv['id']}_{msg_idx}"
                        for section_title, content in sections.items():
                            # Normalize title: strip parenthetical suffixes like "(140 words)"
                            base_title = re.sub(r'\s*\(.*?\)\s*$', '', section_title).strip()
                            if base_title in allowed_sections:
                                expanded = base_title in expanded_sections
                                # Guard: if the Summary content contains an inline
                                # '# Key Evidence' or '# Reasoning' header that the
                                # section parser missed (no preceding newline), strip
                                # everything from that point so it doesn't render as
                                # a heading + bullets inside the Summary expander.
                                if base_title == "Summary":
                                    _hdr_leak = re.search(
                                        r'#\s+(?:Key Evidence|Reasoning)\b', content
                                    )
                                    if _hdr_leak:
                                        content = content[:_hdr_leak.start()].strip()
                                render_section_with_copy(base_title, content, expanded, key_prefix)

                    # ── LLM-as-Judge Evaluation ──────────────────────────────
                    _judge_cache_key = f"judge_{current_conv['id']}_{msg_idx}"
                    if "judge_scores" not in st.session_state:
                        st.session_state["judge_scores"] = {}

                    # Pre-populate scores from API — improved summary is already the main Summary,
                    # so do NOT seed judge_improvements (expander starts with fresh Improve button).
                    if data.get("judge_scores") and _judge_cache_key not in st.session_state["judge_scores"]:
                        st.session_state["judge_scores"][_judge_cache_key] = data["judge_scores"]

                    # ── Deep Dive ──
                    with st.expander("Deep Dive", expanded=False):
                        _existing_scores = st.session_state["judge_scores"].get(_judge_cache_key)
                        _improve_cache_key = f"improve_{current_conv['id']}_{msg_idx}"
                        _rescore_cache_key = f"rescore_{current_conv['id']}_{msg_idx}"
                        if "judge_improvements" not in st.session_state:
                            st.session_state["judge_improvements"] = {}
                        if "judge_rescores" not in st.session_state:
                            st.session_state["judge_rescores"] = {}
                        _existing_scores      = st.session_state["judge_scores"].get(_judge_cache_key)
                        _existing_improvement = st.session_state["judge_improvements"].get(_improve_cache_key)
                        _existing_rescore     = st.session_state["judge_rescores"].get(_rescore_cache_key)

                        # ── ONE button: score original + generate Deep Dive atomically ──
                        _dd_auto_key = f"_dd_auto_{_improve_cache_key}"
                        _auto_dd = st.session_state.pop(_dd_auto_key, False)
                        if not _existing_improvement:
                            if not _OLLAMA_AVAILABLE:
                                st.warning("Ollama package not installed (`pip install ollama`).", icon="⚠️")
                            elif not _JUDGE_OLLAMA_API_KEY:
                                st.warning("OLLAMA_API_KEY not set — judge model unavailable.", icon="⚠️")
                            else:
                                _trigger_dd = _auto_dd
                                if not _auto_dd:
                                    _trigger_dd = st.button(
                                        "Deep Dive",
                                        key=f"deepdive_btn_{current_conv['id']}_{msg_idx}",
                                        help=f"Score original summary with {_JUDGE_MODEL}, then generate detailed synthesis with {_IMPROVE_MODEL}"
                                    )
                                if _trigger_dd:
                                    _q_text = msg.get("question", "")
                                    if not _q_text:
                                        _msgs = current_conv.get("messages", [])
                                        for _prev in reversed(_msgs[:msg_idx]):
                                            if _prev.get("role") == "user":
                                                _q_text = _prev.get("content", "")
                                                break
                                    # Step 1: Score original summary (skip if already scored)
                                    if not _existing_scores:
                                        _summary_m = re.search(
                                            r'#{1,2}\s*Summary(?:\s*\(.*?\))?\s*\n(.*?)(?=\n#{1,2}\s|\Z)',
                                            md_raw, re.DOTALL | re.IGNORECASE
                                        )
                                        _summary_plain = re.sub(r'\[.*?\](?:\(.*?\))?', '', _summary_m.group(1) if _summary_m else md_raw).strip()
                                        with st.spinner("Evaluating original summary..."):
                                            _result = _llm_judge_score(_q_text, _summary_plain)
                                        if _result:
                                            st.session_state["judge_scores"][_judge_cache_key] = _result
                                            _existing_scores = _result
                                        else:
                                            st.error("Judge scoring failed.", icon="❌")
                                    # Step 2: Generate Deep Dive synthesis
                                    if _existing_scores:
                                        _sum_m2 = re.search(
                                            r'#{1,2}\s*Summary(?:\s*\(.*?\))?\s*\n(.*?)(?=\n#{1,2}\s|\Z)',
                                            md_raw, re.DOTALL | re.IGNORECASE
                                        )
                                        _sum_with_cites_raw = (_sum_m2.group(1) if _sum_m2 else md_raw).strip()
                                        _sum_with_cites = re.sub(r'\[([^\]]+)\]\(https?://[^)]+\)', r'[\1]', _sum_with_cites_raw)
                                        _sum_with_cites = re.sub(r',\s*INCLUDE', '', _sum_with_cites)
                                        with st.spinner("Generating Deep Dive..."):
                                            _improved = _llm_deep_dive_summary(
                                                _q_text, _sum_with_cites, _existing_scores,
                                                raw_evidence=data.get("raw_evidence"),
                                            )
                                        if _improved:
                                            st.session_state["judge_improvements"][_improve_cache_key] = _improved
                                            st.rerun()
                                        else:
                                            st.error("Deep Dive generation failed.", icon="❌")

                        # ── Display scores (hidden) ─────────────────────────────
                        _scores_to_show = _existing_scores
                        if False and _scores_to_show:
                            with st.expander("Original Summary — LLM-as-Judge scores", expanded=False):
                                _score_colors = {1: "#c0392b", 2: "#e67e22", 3: "#f39c12", 4: "#27ae60", 5: "#1a5276"}
                                _score_labels = {1: "Very Poor", 2: "Poor", 3: "Fair", 4: "Good", 5: "Excellent"}
                                st.markdown("##### Evaluated Original Summary by LLM as Judge")
                                for _dim, _label in [
                                    ("accuracy",        "Accuracy"),
                                    ("informativeness", "Informativeness"),
                                    ("completeness",    "Completeness"),
                                ]:
                                    _v = _scores_to_show[_dim]
                                    _color = _score_colors.get(_v, "#888")
                                    _stars = "★" * _v + "☆" * (5 - _v)
                                    _lbl = _score_labels.get(_v, "")
                                    st.markdown(
                                        f"{_label}   "
                                        f"<span style='color:{_color};font-size:18px;'>{_stars}</span> "
                                        f"<span style='color:{_color};font-size:12px;'>({_lbl})</span>",
                                        unsafe_allow_html=True
                                    )
                                st.markdown(f"**Judge notes:** {_scores_to_show['justification']}")
                                st.caption(f"Judge: {_JUDGE_MODEL} | Improve: {_IMPROVE_MODEL}")

                        # ── Display Deep Dive synthesis ─────────────────────────
                        if _existing_improvement:
                            st.divider()
                            st.markdown("**Deep Dive**")
                            st.caption("Detailed summary")
                            _include_pmcids = {
                                (("PMC" + str(ev.get("pmcid", "")).strip()).upper()
                                 if not str(ev.get("pmcid", "")).strip().upper().startswith("PMC")
                                 else str(ev.get("pmcid", "")).strip().upper())
                                for ev in (data.get("raw_evidence") or [])
                                if ev.get("is_include")
                            }
                            def _dd_normalize(txt):
                                """Pre-normalize Deep Dive text: collapse double brackets,
                                fix malformed citations, then inject INCLUDE badges."""
                                # 0. Nuclear: strip double brackets even if content has HTML tags
                                #    Handles [[<a href>PMC...</a>]] from already-processed cached content
                                while '[[' in txt and ']]' in txt:
                                    new = re.sub(r'\[\[([^\]]*?PMC[^\]]*?)\]\]', r'[\1]', txt)
                                    if new == txt:
                                        break
                                    txt = new
                                # 1a. Strip ANY number of outer brackets: [[...]] or [[[...]]] -> [...]
                                txt = re.sub(r'\[{2,}(PMC[^\[\]]+)\]{2,}', r'[\1]', txt)
                                # 1b. Second pass for edge cases
                                txt = re.sub(r'\[{2,}(PMC[^\[\]]+)\]{2,}', r'[\1]', txt)
                                # 2. [[PMCxx:Section] INCLUDE] -> [PMCxx:Section, INCLUDE]
                                txt = re.sub(
                                    r'\[\[(PMC\d{4,}:[A-Za-z0-9_-]+)\]\s*INCLUDE\]',
                                    r'[\1, INCLUDE]', txt
                                )
                                # 2b. Space-INCLUDE to comma-INCLUDE (single and compound bracket forms)
                                txt = re.sub(
                                    r'(PMC\d{4,}:[A-Za-z0-9_-]+)\s+INCLUDE(?=[;,\]\(])',
                                    r'\1, INCLUDE', txt
                                )
                                # 3. Fix bare [PMCxx]:Section -> [PMCxx:Section]
                                txt = re.sub(
                                    r'\[(PMC\d{4,})\]:([A-Za-z]+)',
                                    r'[\1:\2]', txt
                                )
                                # 4. Ensure space between adjacent citation brackets ][ -> ] [
                                txt = re.sub(r'\]\s*\[', '] [', txt)
                                # 5. Inject INCLUDE badge for INCLUDE-funded papers
                                #    Handles single [PMCxx:Sec] AND multi [PMCxx:Sec; PMCyy:Sec]
                                def _inject_include(m):
                                    content   = m.group(1)
                                    parts     = [p.strip() for p in re.split(r'\s*;\s*', content)]
                                    new_parts = []
                                    changed   = False
                                    for part in parts:
                                        pm = re.match(r'(PMC\d{4,}):([A-Za-z0-9_,-]+)', part)
                                        if pm and pm.group(1).upper() in _include_pmcids:
                                            sec = pm.group(2).rstrip(',').strip()
                                            if ', INCLUDE' not in part and ',INCLUDE' not in part:
                                                part    = f"{pm.group(1)}:{sec}, INCLUDE"
                                                changed = True
                                        new_parts.append(part)
                                    return ('[' + '; '.join(new_parts) + ']') if changed else m.group(0)
                                return re.sub(
                                    r'\[(PMC[^\[\]]+)\](?!\()',
                                    _inject_include, txt
                                )
                            _improved_linked = _process_pmc_citations(
                                _dd_normalize(_existing_improvement), _evidence_lookup
                            )
                            st.markdown(_improved_linked, unsafe_allow_html=True)

                            # ── Re-generate Deep Dive ─────────────────────────
                            st.divider()
                            if st.button("Re-generate Deep Dive", key=f"reimprove_btn_{current_conv['id']}_{msg_idx}"):
                                st.session_state[_dd_auto_key] = True
                                del st.session_state["judge_improvements"][_improve_cache_key]
                                if _rescore_cache_key in st.session_state.get("judge_rescores", {}):
                                    del st.session_state["judge_rescores"][_rescore_cache_key]
                                st.rerun()
                    raw_evidence = data.get("raw_evidence")
                    if raw_evidence and isinstance(raw_evidence, list):
                        import re as _re
                        import json as _json

                        # Extract PMCIDs cited in the Summary section
                        # Use md_linked (which has the improved summary spliced in)
                        summary_match = _re.search(
                            r'#{1,2}\s*Summary(?:\s*\(.*?\))?\s*\n(.*?)(?=\n#{1,2}\s|\Z)',
                            md_linked, _re.DOTALL | _re.IGNORECASE
                        )
                        summary_text  = summary_match.group(1) if summary_match else ""
                        summary_lower = summary_text.lower()
                        summary_pmcids = set(
                            _re.findall(r'\[?(PMC\d+)\s*:', summary_text, _re.IGNORECASE)
                        ) if summary_text else set()

                        # Filter to cited papers, deduplicated by PMCID
                        seen_pmcids    = set()
                        cited_evidence = []
                        for ev in raw_evidence:
                            pmcid = (ev.get("pmcid") or "").strip()
                            if pmcid in summary_pmcids and pmcid not in seen_pmcids:
                                cited_evidence.append(ev)
                                seen_pmcids.add(pmcid)
                        if not cited_evidence:
                            cited_evidence = raw_evidence[:6]

                        def _parse_terms(val):
                            if not val:
                                return []
                            try:
                                items = _json.loads(val)
                                if isinstance(items, list):
                                    return [str(i).strip() for i in items if str(i).strip()]
                            except Exception:
                                pass
                            return [v.strip() for v in str(val).split(";") if v.strip()]

                        def _pill(term, bg, matched):
                            """Render a single pill tag; bold + brighter if matched."""
                            if matched:
                                style = (
                                    f"display:inline-block;background:{bg};color:white;"
                                    f"padding:3px 10px;border-radius:12px;font-size:11px;"
                                    f"margin:2px 3px;font-weight:700;"
                                    f"border:2px solid rgba(0,0,0,0.25);"
                                )
                            else:
                                style = (
                                    f"display:inline-block;background:{bg};color:white;"
                                    f"padding:3px 10px;border-radius:12px;font-size:11px;"
                                    f"margin:2px 3px;opacity:0.55;"
                                )
                            return f'<span style="{style}">{term}</span>'

                        cards_html = ""
                        for i, ev in enumerate(cited_evidence):
                            topic  = (ev.get("topic") or "").strip() or "—"
                            pmcid  = ev.get("pmcid") or ""
                            mesh   = _parse_terms(ev.get("mesh_terms"))
                            kw     = _parse_terms(ev.get("keywords"))
                            # Deep-link straight to the cited passage (Text Fragment) when
                            # the retrieved chunk text yields a usable phrase; the range
                            # form highlights the passage end-to-end, not just its start.
                            _card_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
                            _card_url += _text_fragment_directives(str(ev.get("text") or ""))

                            mesh_pills = "".join(
                                _pill(t, "#1abc9c", t.lower() in summary_lower)
                                for t in mesh
                            ) or '<span style="color:#aaa;font-size:11px;">—</span>'

                            kw_pills = "".join(
                                _pill(t, "#e67e22", t.lower() in summary_lower)
                                for t in kw
                            ) or '<span style="color:#aaa;font-size:11px;">—</span>'

                            cards_html += f"""
<div style="border:1px solid #d5e8f0;border-radius:10px;padding:14px 16px;
            margin-bottom:10px;background:#f4fafd;">
  <div style="margin-bottom:7px;">
    <span style="font-size:13px;font-weight:700;color:#1a5276;margin-right:6px;">Topic:</span>
    <span style="font-size:13px;font-weight:700;color:#1a5276;">{topic}</span>
    <a href="{_card_url}" target="_blank" style="font-size:11px;font-weight:500;color:#2563eb;margin-left:8px;text-decoration:none;">{pmcid}</a>
  </div>
  <div style="margin-bottom:5px;">
    <span style="font-size:13px;font-weight:700;color:#148f77;margin-right:6px;">MeSH Terms:</span>{mesh_pills}
  </div>
  <div>
    <span style="font-size:13px;font-weight:700;color:#ca6f1e;margin-right:6px;">Keywords:</span>{kw_pills}
  </div>
</div>"""

                        legend = (
                            '<div style="font-size:10px;color:#777;margin-bottom:8px;">'
                            '● <b>Bright</b> = term appears in summary text &nbsp;|&nbsp;'
                            '● <b>Faded</b> = term not in summary text</div>'
                        )

                        with st.expander(
                            f" Highlighted Topic, MeSH Terms and Keywords from Top Retrieved Papers ({len(cited_evidence)})",
                            expanded=False
                        ):
                            st.markdown(
                                legend + cards_html,
                                unsafe_allow_html=True
                            )
                    else:
                        # No sections found AND no raw_evidence - display raw markdown with copy all button
                        if not sections:
                            col1, col2 = st.columns([6, 1])
                            with col2:
                                render_copy_button(
                                    _html_to_plain_for_copy(md_linked),
                                    key=f"copyall_{current_conv['id']}_{msg_idx}",
                                    label="Copy All",
                                )
                            st.markdown(md_linked, unsafe_allow_html=True)

                    # Evidence section - grouped by section type - HIDDEN
                    # raw_evidence = data.get("raw_evidence")
                    # if raw_evidence and isinstance(raw_evidence, list):
                    #     # Group evidence by section
                    #     section_order = ["Results", "Methods", "Conclusion", "Discussion", "Abstract", "Introduction", "FullText", "NA"]
                    #     section_icons = {
                    #         "Results": "", "Methods": "", "Conclusion": "[OK]",
                    #         "Discussion": "", "Abstract": "", "Introduction": "",
                    #         "FullText": "", "NA": "?"
                    #     }

                    #     # Count by section
                    #     section_groups = {}
                    #     for item in raw_evidence:
                    #         sec = item.get("section", "NA")
                    #         if sec not in section_groups:
                    #             section_groups[sec] = []
                    #         section_groups[sec].append(item)

                    #     # Section summary
                    #     section_summary = " | ".join([f"{section_icons.get(s, '')} {s}: {len(items)}"
                    #                                   for s, items in section_groups.items()])

                    #     with st.expander(f" Source Evidence ({len(raw_evidence)} citations)", expanded=False):
                    #         st.markdown(f"**Sections:** {section_summary}")
                    #         st.divider()

                    #         # Display by section in priority order
                    #         for section in section_order:
                    #             if section in section_groups:
                    #                 icon = section_icons.get(section, "")
                    #                 st.markdown(f"### {icon} {section} Section")

                    #             for item in section_groups[section]:
                    #                 pmcid = item.get("pmcid", "N/A")
                    #                 citation_tag = item.get("citation_tag", pmcid)
                    #                 text_content = item.get("text", "")[:300] + "..." if len(item.get("text", "")) > 300 else item.get("text", "")
                    #                 score = item.get("score", 0)
                    #                 section_bonus = item.get("section_bonus", 0)
                    #                 pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"

                    #                 # Display with score and section bonus
                    #                 score_str = f"Score: {score:.3f}" if score else ""
                    #                 bonus_str = f" (+{section_bonus:.1%} bonus)" if section_bonus > 0 else ""

                    #                 st.markdown(f"**[{citation_tag}]({pmc_url})** {score_str}{bonus_str}")
                    #                 st.caption(text_content)
                    #                 st.markdown("---")

                    # Quality scores - HIDDEN
                    # if data.get("quality_scores"):
                    #     with st.expander(" Quality Assessment", expanded=False):
                    #         scores = data["quality_scores"]
                    #         if isinstance(scores, dict):
                    #             cols = st.columns(min(5, len(scores)))
                    #             for i, (dim, score_data) in enumerate(scores.items()):
                    #                 with cols[i % len(cols)]:
                    #                     score_val = score_data.get("score", score_data) if isinstance(score_data, dict) else score_data
                    #                     st.metric(dim[:12], f"{score_val}/5")

                    # Pipeline Trace - LangGraph Workflow visualization for this query - HIDDEN
                    # trace_data = data.get("trace", [])
                    # # Deduplicate trace entries (keep only first occurrence of each agent)
                    # seen_agents = set()
                    # deduped_trace = []
                    # for t in trace_data:
                    #     agent = t.get("agent", "")
                    #     if agent not in seen_agents:
                    #         seen_agents.add(agent)
                    #         deduped_trace.append(t)
                    # trace_data = deduped_trace
                    # if trace_data:
                    #     if False:  # Disabled - comment out entire block
                    #         # Load T21 logo as base64 for synthesis agent
                    #         import base64
                    #         import os
                    #         logo_path = os.path.join(os.path.dirname(__file__), "T21_researchAssistant_logo.png")
                    #         t21_logo_b64 = ""
                    #         if os.path.exists(logo_path):
                    #             with open(logo_path, "rb") as img_file:
                    #                 t21_logo_b64 = base64.b64encode(img_file.read()).decode()

                    #         # Agent icons and colors (synthesis uses image tag)
                    #         agent_icons = {
                    #             "retrieval": "",
                    #             "reranker": "",
                    #             "synthesis": f'<img src="data:image/png;base64,{t21_logo_b64}" style="height:18px;vertical-align:middle;">' if t21_logo_b64 else "",
                    #             "raw_evidence": "",
                    #             "safety_guard": "",
                    #             "citation_audit": "[OK]",
                    #             "quality_review": "*"
                    #         }
                    #         agent_colors = {
                    #             "retrieval": "#e8eef7",
                    #             "reranker": "#d0dff0",
                    #             "synthesis": "#d0dff0",
                    #             "raw_evidence": "#e8eef7",
                    #             "safety_guard": "#fef3c7",
                    #             "citation_audit": "#e8eef7",
                    #             "quality_review": "#d0dff0"
                    #         }

                    #         # Calculate total time
                    #         total_time = sum(t.get("timing_ms", 0) for t in trace_data)
                    #         st.markdown(f"**Total Pipeline Time:** {total_time:,} ms ({total_time/1000:.1f}s)")
                    #         st.markdown("---")

                    #         # Display each step in the workflow
                    #         for i, step in enumerate(trace_data):
                    #             agent = step.get("agent", "unknown")
                    #             timing = step.get("timing_ms", 0)
                    #             note = step.get("note", "")
                    #             icon = agent_icons.get(agent, "")
                    #             color = agent_colors.get(agent, "#f3f4f6")

                    #             # Calculate percentage of total time
                    #             pct = (timing / total_time * 100) if total_time > 0 else 0

                    #             # Create a styled box for each step
                    #             step_html = f"""
                    #             <div style="display: flex; align-items: center; margin: 8px 0; gap: 10px;">
                    #                 <div style="background: {color}; padding: 8px 12px; border-radius: 6px;
                    #                             border: 1px solid #c5d5ea; min-width: 120px; text-align: center;">
                    #                     <span style="font-size: 16px;">{icon}</span>
                    #                     <span style="font-weight: 500; color: #1e3a5f; font-size: 12px;">{agent.replace('_', ' ').title()}</span>
                    #                 </div>
                    #                 <div style="flex: 1; background: #f3f4f6; border-radius: 4px; height: 8px; overflow: hidden;">
                    #                     <div style="background: linear-gradient(90deg, #1e4d8b, #10a37f); width: {min(pct*2, 100)}%; height: 100%;"></div>
                    #                 </div>
                    #                 <div style="min-width: 80px; text-align: right; font-size: 11px; color: #4b5563;">
                    #                     {timing:,}ms ({pct:.1f}%)
                    #                 </div>
                    #             </div>
                    #             <div style="margin-left: 140px; margin-bottom: 12px; font-size: 11px; color: #6b7280;">
                    #                 {note}
                    #             </div>
                    #             """
                    #             st.markdown(step_html, unsafe_allow_html=True)

                    #             # Add arrow between steps (except for last)
                    #             if i < len(trace_data) - 1:
                    #                 st.markdown("<div style='text-align: center; color: #9ca3af; margin: -8px 0;'>↓</div>", unsafe_allow_html=True)

                    # Export buttons
                    with st.expander(" Export & Copy", expanded=False):
                        # Resolve the user question for this response
                        _export_question = msg.get("question", "")
                        if not _export_question:
                            for _prev in reversed(current_conv.get("messages", [])[:msg_idx]):
                                if _prev.get("role") == "user":
                                    _export_question = _prev.get("content", "")
                                    break

                        # Gather Deep Dive text if it exists
                        _dd_export_key = f"improve_{current_conv['id']}_{msg_idx}"
                        _dd_text = st.session_state.get("judge_improvements", {}).get(_dd_export_key)

                        # Build export content — prepend question, append Deep Dive if available
                        plain_text = strip_markdown_formatting(md)
                        md_export = md
                        export_data = dict(data)
                        if _export_question:
                            plain_text = f"Question: {_export_question}\n\n" + plain_text
                            md_export  = f"**Question:** {_export_question}\n\n---\n\n" + md_export
                            export_data = dict({"question": _export_question}, **export_data)
                        if _dd_text:
                            _dd_plain = strip_markdown_formatting(_dd_text)
                            plain_text = plain_text.rstrip() + "\n\n---\nDeep Dive\n---\n\n" + _dd_plain
                            md_export  = md_export.rstrip() + "\n\n---\n\n## Deep Dive\n\n" + _dd_text
                            export_data = dict(data, deep_dive=_dd_text)

                        st.markdown("**Copy Section Text:**")
                        if _dd_text:
                            st.caption("Includes Deep Dive results")
                        st.text_area(
                            "Select all (Ctrl+A) and copy (Ctrl+C):",
                            plain_text,
                            height=150,
                            key=f"copy_area_{current_conv['id']}_{msg_idx}",
                            label_visibility="collapsed"
                        )
                        st.caption(" Click in the text box, press Ctrl+A to select all, then Ctrl+C to copy")

                        st.divider()
                        st.markdown("**Download Files:**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.download_button(
                                " Text",
                                data=plain_text.encode("utf-8"),
                                file_name=f"response_{msg_idx}.txt",
                                mime="text/plain",
                                key=f"dl_txt_{current_conv['id']}_{msg_idx}"
                            )
                        with col2:
                            st.download_button(
                                " Markdown",
                                data=md_export.encode("utf-8"),
                                file_name=f"response_{msg_idx}.md",
                                mime="text/markdown",
                                key=f"dl_md_{current_conv['id']}_{msg_idx}"
                            )
                        with col3:
                            st.download_button(
                                " JSON",
                                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                                file_name=f"response_{msg_idx}.json",
                                mime="application/json",
                                key=f"dl_json_{current_conv['id']}_{msg_idx}"
                            )
                        with col4:
                            pdf_bytes = make_pdf_bytes("T21 Research", md_export)
                            if pdf_bytes:
                                st.download_button(
                                    " PDF",
                                    data=pdf_bytes,
                                    file_name=f"response_{msg_idx}.pdf",
                                    mime="application/pdf",
                                    key=f"dl_pdf_{current_conv['id']}_{msg_idx}"
                                )
                else:
                    # Plain text response or error
                    st.markdown(msg.get("content", ""))

# Show disclaimer once in scrollable area after first assistant response
_has_response = (
    current_conv is not None
    and any(m.get("role") == "assistant" for m in current_conv.get("messages", []))
)
if _has_response and not st.session_state.is_processing:
    st.markdown(
        '<div style="text-align:center; padding:12px 24px 4px 24px; '
        'font-size:12px; color:#9ca3af;">'
        '⚠️ <em>T21 Research Assistant can make mistakes. '
        'Please verify all information before use.</em></div>',
        unsafe_allow_html=True
    )

def _render_t21_ring(ph, secs: float):
    """Render the circular running indicator: a thin 'comet' arc (donut ring)
    that rotates with real elapsed time, the elapsed time in its centre, and a
    two-line label beside it. The arc angle is derived from elapsed time and the
    ring is drawn with a conic-gradient + radial mask, so each re-render repaints
    cleanly with no CSS-keyframe restart flicker."""
    _el = int(secs)
    _big = f"{_el}s" if _el < 60 else f"{_el // 60}:{_el % 60:02d}"
    _angle = int(secs * 200) % 360
    # 7px-thick ring via a radial mask (turns the disc into a donut).
    _ring_mask = ("-webkit-mask:radial-gradient(farthest-side,#0000 calc(100% - 7px),#000 calc(100% - 7px));"
                  "mask:radial-gradient(farthest-side,#0000 calc(100% - 7px),#000 calc(100% - 7px));")
    ph.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:18px;margin:8px 0 6px 0;padding:18px 24px;
             background:linear-gradient(135deg,#f7faff 0%,#eaf2ff 100%);
             border:1px solid #d6e4ff;border-radius:16px;
             box-shadow:0 6px 22px rgba(37,99,235,0.12);max-width:430px;">
          <div style="position:relative;width:70px;height:70px;flex:none;">
            <div style="position:absolute;inset:0;border-radius:50%;background:#dfe9ff;{_ring_mask}"></div>
            <div style="position:absolute;inset:0;border-radius:50%;
                 background:conic-gradient(from {_angle}deg,
                   rgba(37,99,235,0) 0deg, rgba(96,165,250,0.10) 35deg,
                   rgba(59,130,246,0.45) 105deg, #2563eb 158deg, #1d4ed8 176deg,
                   rgba(37,99,235,0) 180deg, rgba(37,99,235,0) 360deg);{_ring_mask}"></div>
            <div style="position:absolute;inset:0;display:flex;
                 align-items:center;justify-content:center;line-height:1;">
              <span style="font-size:18px;font-weight:800;color:#1e3a8a;
                   font-variant-numeric:tabular-nums;">{_big}</span>
            </div>
          </div>
          <div style="display:flex;align-items:center;">
            <span style="font-size:15.5px;color:#1e3a8a;font-weight:700;">T21 RA is running&hellip;</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- Processing Indicator --------------------
if st.session_state.is_processing:
    _logo_avatar = str(_LOGO_PATH) if _LOGO_PATH.exists() else "🧬"
    with st.chat_message("assistant", avatar=_logo_avatar):
        # Single running indicator: the circular elapsed-time ring (updated from the
        # polling loop in the processing block below). Render an initial frame so it
        # is visible immediately, with no gap before the loop takes over.
        _t21_elapsed_ph = st.empty()
        _render_t21_ring(_t21_elapsed_ph, 0.0)

# -------------------- Server Status Gate --------------------
_gate_info = fetch_system_info()
_api_online = _gate_info.get("status") == "healthy"
_vec_count = _gate_info.get("db_stats", {}).get("total_vectors", "N/A")
try:
    _db_has_data = int(str(_vec_count).replace(",", "").strip()) > 0
except (TypeError, ValueError):
    _db_has_data = False
_server_blocked = not _api_online or not _db_has_data

if not _api_online:
    st.markdown(
        '<div style="background:#fef2f2;border:1.5px solid #fca5a5;border-radius:10px;'
        'padding:18px 22px;margin:18px 0 10px 0;display:flex;align-items:flex-start;gap:14px;">'
        '<span style="font-size:28px;">🔴</span>'
        '<div><div style="font-size:17px;font-weight:700;color:#991b1b;margin-bottom:4px;">'
        'Server Offline</div>'
        '<div style="font-size:14px;color:#7f1d1d;">'
        'The research API is currently unreachable. Queries are disabled until the server comes back online.'
        '</div></div></div>',
        unsafe_allow_html=True
    )
elif not _db_has_data:
    st.markdown(
        '<div style="background:#fffbeb;border:1.5px solid #fcd34d;border-radius:10px;'
        'padding:18px 22px;margin:18px 0 10px 0;display:flex;align-items:flex-start;gap:14px;">'
        '<span style="font-size:28px;">⚠️</span>'
        '<div><div style="font-size:17px;font-weight:700;color:#92400e;margin-bottom:4px;">'
        'Vector Database Empty</div>'
        '<div style="font-size:14px;color:#78350f;">'
        'The publication corpus has not been loaded yet. Queries are disabled until data is available.'
        '</div></div></div>',
        unsafe_allow_html=True
    )

# -------------------- Chat Input --------------------
st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)

_chat_disabled = st.session_state.is_processing or _server_blocked
_chat_placeholder = (
    "Server offline — queries disabled" if not _api_online
    else "Vector database empty — queries disabled" if not _db_has_data
    else "Ask a research question about Down syndrome..."
)
user_input = st.chat_input(_chat_placeholder, disabled=_chat_disabled)

if user_input and not st.session_state.is_processing and not _server_blocked:
    # Create conversation if needed
    if current_conv is None:
        current_conv = create_new_conversation()

    # Add user message
    current_conv["messages"].append({
        "role": "user",
        "content": user_input
    })

    # Update title if first message
    if len(current_conv["messages"]) == 1:
        current_conv["title"] = user_input[:40] + ("..." if len(user_input) > 40 else "")

    # Set processing state
    st.session_state.is_processing = True
    st.rerun()

# -------------------- Process Pending Request --------------------
if st.session_state.is_processing and current_conv:
    messages = current_conv["messages"]
    # Find the last user message without a response
    if messages and messages[-1]["role"] == "user":
        question = messages[-1]["content"]

        # Call backend with settings from sidebar
        top_k = st.session_state.get("settings_top_k", DEFAULT_TOPK)
        include_hits = st.session_state.get("settings_include_hits", DEFAULT_INCLUDE_HITS)
        box = ResultBox()
        # Run the (blocking) backend call in a worker thread so the main thread can
        # live-update the elapsed-time counter shown inside the running indicator.
        # The worker touches only `box` (no Streamlit calls) — Streamlit writes stay on the main thread.
        import threading as _threading, time as _time
        _worker = _threading.Thread(
            target=_call_backend, args=(question, top_k, include_hits, box), daemon=True
        )
        _start_ts = _time.time()
        _worker.start()
        _elapsed_ph = globals().get("_t21_elapsed_ph")
        while _worker.is_alive():
            _secs = _time.time() - _start_ts
            if _elapsed_ph is not None:
                _render_t21_ring(_elapsed_ph, _secs)
            _time.sleep(0.2)    # smoother ring rotation (cheap markdown update)
        _worker.join()
        if _elapsed_ph is not None:
            _elapsed_ph.empty()

        # Process response
        if box.http_error_text:
            current_conv["messages"].append({
                "role": "assistant",
                "content": f"Error: {box.http_error_text}",
                "data": None,
                "error": True
            })
        elif box.other_error_text:
            current_conv["messages"].append({
                "role": "assistant",
                "content": f"Error: {box.other_error_text}",
                "data": None,
                "error": True
            })
        elif box.data:
            current_conv["messages"].append({
                "role": "assistant",
                "content": box.data.get("markdown", ""),
                "data": box.data,
                "error": False
            })
        else:
            current_conv["messages"].append({
                "role": "assistant",
                "content": "No response from backend.",
                "data": None,
                "error": True
            })

        st.session_state.is_processing = False
        st.rerun()
