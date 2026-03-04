"""
app.py
------
GraphRAG Document Intelligence: Single-Page Document QA Web Application.

A clean, responsive, enterprise-grade interface for document-grounded question
answering, verifiable source citations, claim-level verification, and physical
PDF inspection.

Key Architecture:
- Single-page layout with clear visual hierarchy and zero distracting clutter.
- Strict document-only policy by default: never invents facts or legal conclusions.
- Distinct status badges: Supported, Partially Supported, and Insufficient Evidence.
- Verbatim evidence cards paired with an interactive PyMuPDF physical page inspector.
- Multi-question session support for follow-up questions without re-indexing.
- Clean session reset and document isolation preventing cross-document contamination.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import streamlit as st

from graphrag.app_service import DocumentIntelligenceService
from graphrag.qa_schemas import ClaimVerificationStatus, QAStatus

# ── Streamlit Page Configuration ──────────────────────────────────────────────

st.set_page_config(
    page_title="GraphRAG Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Polished, High-Contrast Custom CSS ─────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Global layout & typography */
    .stApp {
        background-color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #0f172a;
    }
    .main .block-container {
        max-width: 1220px;
        padding-top: 1.2rem;
        padding-bottom: 3.5rem;
    }

    /* Professional Top Header */
    .app-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .app-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.025em;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .app-tagline {
        font-size: 0.96rem;
        color: #64748b;
        margin-bottom: 1rem;
        line-height: 1.45;
    }
    .header-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.28rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .badge-mock { background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; }
    .badge-live { background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }
    .badge-policy { background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }
    .badge-doc-active { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
    .badge-doc-none { background: #f8fafc; color: #64748b; border: 1px solid #e2e8f0; }

    /* Document Status Bar */
    .doc-bar {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .doc-bar-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .doc-bar-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
    }
    .meta-chip {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.3rem;
    }

    /* Answer Container & Badges */
    .answer-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .answer-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 0.85rem;
        margin-bottom: 1rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.95rem;
        border-radius: 6px;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .status-pill-supported { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .status-pill-partial { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .status-pill-insufficient { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .status-pill-conflict { background: #f3e8ff; color: #6b21a8; border: 1px solid #e9d5ff; }
    .status-pill-error { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }

    .answer-text {
        font-size: 1.05rem;
        color: #1e293b;
        line-height: 1.7;
        margin: 0.5rem 0 1rem 0;
    }

    /* Limitations & Callouts */
    .limitations-box {
        background: #fffbeb;
        border: 1px solid #fef3c7;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 0.9rem 1.15rem;
        margin: 1.1rem 0;
        color: #92400e;
    }
    .limitations-title {
        font-weight: 700;
        font-size: 0.92rem;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    /* Claim Cards */
    .claim-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.65rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
    }
    .claim-card.verified { border-left: 4px solid #10b981; }
    .claim-card.unsupported { border-left: 4px solid #ef4444; }
    .claim-card.contradicted { border-left: 4px solid #dc2626; }
    .claim-card.inconclusive { border-left: 4px solid #f59e0b; }

    /* Evidence Cards */
    .evidence-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
    }
    .quote-box {
        background: #f8fafc;
        border-left: 3px solid #3b82f6;
        padding: 0.75rem 1rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.88rem;
        color: #334155;
        margin-top: 0.6rem;
        line-height: 1.55;
        border-radius: 0 6px 6px 0;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session State Initialization ──────────────────────────────────────────────

def _safe_get_total_pages(svc: Any, fallback: int = 76) -> int:
    """Safely retrieves total physical pages from service or active index."""
    fn = getattr(svc, "get_total_pages", None)
    if callable(fn):
        try:
            val = fn()
            if val > 0:
                return val
        except Exception:
            pass
    idx = getattr(svc, "active_index", None)
    if idx is not None and getattr(idx, "total_pages", 0) > 0:
        return idx.total_pages
    return fallback


if "service" not in st.session_state or not hasattr(st.session_state.service, "get_total_pages"):
    st.session_state.service = DocumentIntelligenceService()

if "qa_response" not in st.session_state:
    st.session_state.qa_response = None

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "rendered_page_bytes" not in st.session_state:
    st.session_state.rendered_page_bytes = None

if "rendered_page_num" not in st.session_state:
    st.session_state.rendered_page_num = None

if "rendered_total_pages" not in st.session_state:
    st.session_state.rendered_total_pages = 0

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

service: DocumentIntelligenceService = st.session_state.service


# ── Sidebar: Settings & Execution Controls ─────────────────────────────────────

with st.sidebar:
    st.title("⚙️ System Settings")
    st.caption("Configure execution mode, model identity, and grounding constraints.")
    st.markdown("---")

    st.subheader("Model Execution Mode")
    exec_mode = st.radio(
        "Execution Provider:",
        options=["Deterministic Mock (Offline)", "Live Anthropic Claude"],
        index=0,
        help="Use deterministic mock for offline testing with zero API cost, or live Anthropic Claude.",
    )

    is_mock = exec_mode == "Deterministic Mock (Offline)"
    api_key: Optional[str] = None
    model_name: Optional[str] = None

    if not is_mock:
        default_key = os.environ.get("ANTHROPIC_API_KEY", "")
        api_key_input = st.text_input(
            "Anthropic API Key:",
            value=default_key,
            type="password",
            help="Set ANTHROPIC_API_KEY environment variable or enter securely here.",
        )
        api_key = api_key_input if api_key_input else None

        env_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        supported_models = [
            "claude-sonnet-5-5",
            "claude-haiku-5-5",
            "claude-opus-5-5",
            "claude-fable-5-1",
        ]
        if env_model not in supported_models:
            model_options = [env_model] + supported_models
        else:
            model_options = [env_model] + [m for m in supported_models if m != env_model]

        model_name = st.selectbox(
            "Model Identifier:",
            options=model_options,
            index=0,
            help=f"Active model for live QA. Configurable via ANTHROPIC_MODEL env var (default: {env_model}).",
        )

        if not api_key:
            st.warning("⚠ API key is required for Live Mode. Queries will not proceed without valid credentials.")

    st.markdown("---")
    st.subheader("Document Grounding Policy")
    strict_policy = st.checkbox(
        "Strict Document-Only Mode",
        value=True,
        help="Strictly prohibits outside knowledge, unverified premises, and speculative assumptions.",
    )

    top_k = st.slider(
        "Max Retrieved Passages (top-k):",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of most relevant passage chunks fed to the question-answering synthesizer.",
    )

    st.markdown("---")
    st.subheader("Session Controls")
    doc_summary = service.get_document_summary()
    if doc_summary.get("has_active_document"):
        st.write(f"Active Document: **{doc_summary['source_file']}**")
        if st.button("🗑️ Clear Document & Reset Session", use_container_width=True):
            service.clear_active_document()
            st.session_state.qa_response = None
            st.session_state.qa_history = []
            st.session_state.rendered_page_bytes = None
            st.session_state.rendered_page_num = None
            st.session_state.query_input = ""
            st.rerun()
    else:
        st.info("No active document loaded.")


# ── Top Header & Operating Mode Badges ────────────────────────────────────────

mode_label = "Mode: Offline Mock (Deterministic)" if is_mock else f"Mode: Live Anthropic ({model_name})"
badge_mode_class = "badge-mock" if is_mock else "badge-live"
doc_summary = service.get_document_summary()
has_active_doc = doc_summary.get("has_active_document", False)

if has_active_doc:
    doc_badge_html = f'<span class="badge badge-doc-active">📁 Active: {doc_summary["source_file"]} ({doc_summary["total_pages"]} pages)</span>'
else:
    doc_badge_html = '<span class="badge badge-doc-none">📁 No Document Loaded</span>'

policy_badge_html = f'<span class="badge badge-policy">🔒 Policy: {"Strict Document-Only" if strict_policy else "Permissive"}</span>'

st.markdown(
    f"""
    <div class="app-header">
        <div class="app-title">
            <span>📄</span> GraphRAG Document Intelligence
        </div>
        <div class="app-tagline">
            Evidence-grounded question answering with verifiable source citations, claim-level validation, and physical PDF inspection.
        </div>
        <div class="header-badges">
            <span class="badge {badge_mode_class}">⚡ {mode_label}</span>
            {policy_badge_html}
            {doc_badge_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Section 1: Single-Page Document Workspace (Upload & Status) ────────────────

if not has_active_doc:
    st.subheader("1. Load or Upload a Document")
    st.caption("Upload any digital PDF or immediately load the bundled CITES treaty document to start asking questions.")

    col_up, col_sample = st.columns([1.1, 0.9])

    with col_up:
        st.markdown("**Option A: Upload Your PDF Document**")
        uploaded_file = st.file_uploader(
            "Drag and drop or browse for a PDF file (Max 50MB):",
            type=["pdf"],
            help="Native or digital PDF with readable text and tables.",
        )

        if uploaded_file is not None:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.caption(f"**Filename:** `{uploaded_file.name}`")
            with col_f2:
                st.caption(f"**Size:** `{len(uploaded_file.getvalue()) / 1024:.1f} KB`")

            if st.button("⚡ Process & Index PDF Document", type="primary", use_container_width=True):
                with st.spinner("Validating, extracting pages, and constructing searchable chunks..."):
                    try:
                        file_bytes = uploaded_file.getvalue()
                        index, is_cached = service.upload_and_index_document(
                            file_bytes=file_bytes,
                            filename=uploaded_file.name,
                        )
                        st.session_state.qa_response = None
                        st.session_state.qa_history = []
                        st.session_state.rendered_page_bytes = None
                        st.session_state.rendered_page_num = None
                        st.session_state.query_input = ""
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error processing PDF: {exc}")

    with col_sample:
        st.markdown("**Option B: Instant Sample Document**")
        st.markdown(
            """
            **CITES Appendices (Trade in Endangered Species)**  
            Standard international regulatory treaty containing structured classification tables, 
            Appendices I, II, and III listings, taxonomy rules, exceptions, and annotations.
            """
        )
        cites_pdf = Path("CITES_E-Appendices.pdf")
        cites_idx = Path("outputs/cites_index.json")

        if cites_idx.exists():
            st.caption("Pre-indexed treaty: 76 physical pages, 206 searchable chunks, 74 structured tables.")
            if st.button("📚 Load CITES Appendices (Instant Load)", type="secondary", use_container_width=True):
                with st.spinner("Loading CITES Appendices treaty index..."):
                    try:
                        service.load_existing_index(
                            index_path=cites_idx,
                            pdf_path=cites_pdf if cites_pdf.exists() else None,
                        )
                        st.session_state.qa_response = None
                        st.session_state.qa_history = []
                        st.session_state.rendered_page_bytes = None
                        st.session_state.rendered_page_num = None
                        st.session_state.query_input = ""
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to load CITES index: {exc}")
        else:
            st.info("Sample CITES index not found in outputs/cites_index.json.")

else:
    # Active Document Banner with stats and quick actions
    st.markdown(
        f"""
        <div class="doc-bar">
            <div class="doc-bar-header">
                <div class="doc-bar-title">📁 Active Document: <code>{doc_summary['source_file']}</code></div>
                <span class="badge badge-doc-active">● Indexed & Ready</span>
            </div>
            <div>
                <span class="meta-chip">📄 {doc_summary['total_pages']} Physical Pages</span>
                <span class="meta-chip">🧩 {doc_summary['total_chunks']} Searchable Chunks</span>
                <span class="meta-chip">📊 {doc_summary['total_tables']} Detected Tables</span>
                <span class="meta-chip">🔒 SHA-256: {doc_summary['document_hash'][:16]}…</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_act1, col_act2, col_act3 = st.columns([2, 1, 1])
    with col_act2:
        if st.button("🔄 Change Document", use_container_width=True):
            service.clear_active_document()
            st.session_state.qa_response = None
            st.session_state.qa_history = []
            st.session_state.rendered_page_bytes = None
            st.session_state.rendered_page_num = None
            st.session_state.query_input = ""
            st.rerun()
    with col_act3:
        if st.button("🗑️ Reset Session", use_container_width=True):
            service.clear_active_document()
            st.session_state.qa_response = None
            st.session_state.qa_history = []
            st.session_state.rendered_page_bytes = None
            st.session_state.rendered_page_num = None
            st.session_state.query_input = ""
            st.rerun()


# ── Section 2: Question Formulation Interface ──────────────────────────────────

if has_active_doc:
    st.markdown("---")
    st.subheader("2. Ask a Document-Grounded Question")

    is_cites = "cites" in doc_summary["source_file"].lower()
    if is_cites:
        st.caption("Suggested CITES Treaty Questions:")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("🐒 Can Rhesus Monkeys be exported to the USA for pharmaceutical research?", use_container_width=True):
                st.session_state.query_input = "Can Rhesus Monkeys be exported to the USA for pharmaceutical research?"
        with col_s2:
            if st.button("🏛️ Are specific Indian central-government approvals required for this export?", use_container_width=True):
                st.session_state.query_input = "Are specific Indian central-government approvals required for this export, and which approvals or ministries are identified?"

    current_val = st.session_state.get("query_input", "")
    question = st.text_area(
        "Enter your question (grounded strictly in the active document):",
        value=current_val,
        height=85,
        placeholder="Type a specific question about provisions, species, quotas, or rules in the uploaded document…",
    )

    col_btn_ask, col_btn_clear, col_btn_space = st.columns([1.2, 0.8, 3])
    with col_btn_ask:
        ask_clicked = st.button("🔍 Ask Question", type="primary", use_container_width=True)
    with col_btn_clear:
        if st.button("Clear Input", use_container_width=True):
            st.session_state.query_input = ""
            st.rerun()

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question before submitting.")
        elif not is_mock and not api_key:
            st.error("❌ Anthropic API Key is required for Live Mode. Please enter your API key in the sidebar or switch to Deterministic Mock Mode.")
        else:
            mode_desc = "Deterministic Mock" if is_mock else f"Live Anthropic ({model_name})"
            with st.spinner(f"Retrieving passages, validating citations, and synthesizing answer using {mode_desc}..."):
                try:
                    response = service.submit_question(
                        question=question.strip(),
                        mock=is_mock,
                        api_key=api_key,
                        model=model_name,
                        top_k=top_k,
                    )
                    st.session_state.qa_response = response
                    # Add to session history
                    st.session_state.qa_history.insert(0, {
                        "question": question.strip(),
                        "answer": response.answer,
                        "status": response.status,
                    })
                    # Set default rendered page to first cited evidence page if available
                    if response.evidence:
                        first_page = response.evidence[0].pdf_page_index
                        st.session_state.rendered_page_bytes = service.render_page_image(first_page, zoom=1.5)
                        st.session_state.rendered_page_num = first_page
                        st.session_state.rendered_total_pages = _safe_get_total_pages(service)
                    else:
                        st.session_state.rendered_page_bytes = None
                        st.session_state.rendered_page_num = None
                        st.session_state.rendered_total_pages = 0
                except Exception as exc:
                    st.error(f"Error executing QA Engine: {exc}")


# ── Section 3: Verified Grounded Answer ────────────────────────────────────────

response: Optional[QAResponse] = st.session_state.qa_response

if response is not None and has_active_doc:
    st.markdown("---")
    st.subheader("3. Verified Grounded Answer")

    # Explicit user-facing statuses as requested
    status_configs = {
        QAStatus.ANSWERED.value: (
            "status-pill-supported",
            "✅ Supported",
            "The question is fully answered by verified document evidence.",
        ),
        QAStatus.PARTIALLY_ANSWERED.value: (
            "status-pill-partial",
            "ℹ️ Partially Supported",
            "The document provides partial evidence; additional details are unstated.",
        ),
        QAStatus.INSUFFICIENT_EVIDENCE.value: (
            "status-pill-insufficient",
            "⚠ Insufficient Evidence (Abstained)",
            "The document does not contain enough information to support an answer.",
        ),
        QAStatus.CONFLICTING_EVIDENCE.value: (
            "status-pill-conflict",
            "⚡ Conflicting Evidence",
            "The document contains contradictory directives or passages.",
        ),
        QAStatus.AMBIGUOUS.value: (
            "status-pill-partial",
            "❓ Ambiguous Evidence",
            "The document language is discretionary or open to multiple interpretations.",
        ),
        QAStatus.PROCESSING_ERROR.value: (
            "status-pill-error",
            "❌ Processing Error",
            "An error occurred during query retrieval or LLM execution.",
        ),
    }

    pill_class, pill_label, pill_desc = status_configs.get(
        response.status,
        ("status-pill-partial", f"Status: {response.status.upper()}", ""),
    )

    mode_label_short = "Mock (Deterministic)" if is_mock else f"Live ({model_name})"

    st.markdown(
        f"""
        <div class="answer-container">
            <div class="answer-header">
                <div>
                    <span class="status-pill {pill_class}">{pill_label}</span>
                    <span style="margin-left: 0.6rem; font-size: 0.88rem; color: #64748b;">
                        {pill_desc}
                    </span>
                </div>
                <div style="font-size: 0.82rem; color: #64748b;">
                    Execution: <strong>{mode_label_short}</strong> │ Passages Retrieved: <strong>{response.metadata.get('retrieved_passages_count', len(response.evidence))}</strong>
                </div>
            </div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #475569;">Question:</div>
            <div style="font-size: 1.05rem; font-weight: 500; color: #0f172a; margin-bottom: 0.8rem;">
                "{response.question}"
            </div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #475569;">Answer:</div>
            <div class="answer-text">
                {response.answer}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Missing Information & Document Limitations Callout ────────────────────
    if response.limitations:
        lim_bullets = "".join(f"<li>{lim}</li>" for lim in response.limitations)
        st.markdown(
            f"""
            <div class="limitations-box">
                <div class="limitations-title">⚠ Missing Information & Document Limitations:</div>
                <ul style="margin: 0.3rem 0 0 1.2rem; padding: 0;">
                    {lim_bullets}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if response.validation_warnings:
        with st.expander("Validation Integrity Notices", expanded=False):
            for w in response.validation_warnings:
                st.caption(f"• {w}")


    # ── Section 4: Claim-Level Verification Breakdown ─────────────────────────
    if response.claims:
        st.markdown("#### Factual Claims & Verification Status")
        st.caption("Each individual assertion in the answer is verified against retrieved document passages:")

        # Split claims into supported vs uncertain
        supported_claims = [c for c in response.claims if c.verification_status == ClaimVerificationStatus.VERIFIED_SUPPORTED.value]
        uncertain_claims = [c for c in response.claims if c.verification_status != ClaimVerificationStatus.VERIFIED_SUPPORTED.value]

        tab_claims_all, tab_claims_supp, tab_claims_uncert = st.tabs([
            f"All Claims ({len(response.claims)})",
            f"Supported ({len(supported_claims)})",
            f"Uncertain / Unsupported ({len(uncertain_claims)})",
        ])

        def render_claim_card(claim, idx):
            v_status = claim.verification_status
            cid = claim.claim_id or f"C{idx}"
            eids = ", ".join(claim.evidence_ids) if claim.evidence_ids else "None"
            inf_tag = " <span style='color: #0284c7; font-size: 0.8rem;'>[Multi-Passage Inference]</span>" if claim.is_inference else " <span style='color: #16a34a; font-size: 0.8rem;'>[Direct Document Fact]</span>"

            if v_status == ClaimVerificationStatus.VERIFIED_SUPPORTED.value:
                icon = "✅"
                card_class = "verified"
                status_text = "SUPPORTED"
            elif v_status == ClaimVerificationStatus.UNSUPPORTED.value:
                icon = "❌"
                card_class = "unsupported"
                status_text = "UNSUPPORTED"
            elif v_status == ClaimVerificationStatus.CONTRADICTED.value:
                icon = "⚡"
                card_class = "contradicted"
                status_text = "CONTRADICTED"
            else:
                icon = "❓"
                card_class = "inconclusive"
                status_text = v_status.upper()

            st.markdown(
                f"""
                <div class="claim-card {card_class}">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #334155; margin-bottom: 0.25rem;">
                        [{cid}] {icon} {status_text} {inf_tag} &nbsp;│&nbsp; <em>Citations: {eids}</em>
                    </div>
                    <div style="font-size: 0.96rem; color: #1e293b; line-height: 1.5;">
                        {claim.claim}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if claim.explanation:
                st.caption(f"Verification method: {claim.verification_method} — {claim.explanation}")

        with tab_claims_all:
            for idx, c in enumerate(response.claims, 1):
                render_claim_card(c, idx)

        with tab_claims_supp:
            if supported_claims:
                for idx, c in enumerate(supported_claims, 1):
                    render_claim_card(c, idx)
            else:
                st.info("No supported claims identified in this response.")

        with tab_claims_uncert:
            if uncertain_claims:
                for idx, c in enumerate(uncertain_claims, 1):
                    render_claim_card(c, idx)
            else:
                st.success("All synthesized claims are fully supported by document evidence.")


# ── Section 5: Source Evidence & Interactive PDF Page Inspector ───────────────

if response is not None and response.evidence and has_active_doc:
    st.markdown("---")
    st.subheader("4. Source Evidence & Verbatim Passages")
    st.caption("Inspect exact quotations alongside physical PDF page renderings:")

    col_passages, col_pdf_preview = st.columns([1.1, 0.9])

    with col_passages:
        st.markdown("**Cited Passages in Active Document:**")
        for ev in response.evidence:
            heading_info = f" • Section: *{ev.section_heading}*" if ev.section_heading else ""
            printed_info = f" (printed page: {ev.printed_page_number})" if ev.printed_page_number else ""

            st.markdown(
                f"""
                <div class="evidence-card">
                    <div style="font-size: 0.88rem; font-weight: 700; color: #1e293b;">
                        [{ev.evidence_id}] {ev.source_file} │ Physical Page {ev.pdf_page_index}{printed_info}
                    </div>
                    <div style="font-size: 0.78rem; color: #64748b; margin-top: 0.1rem;">
                        Chunk ID: <code>{ev.chunk_id}</code>{heading_info}
                    </div>
                    <div class="quote-box">"{ev.quote}"</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                f"👁️ View Physical Page {ev.pdf_page_index} in PDF",
                key=f"inspect_pg_{ev.evidence_id}",
                use_container_width=True,
            ):
                with st.spinner(f"Rendering page {ev.pdf_page_index} from PDF..."):
                    img_bytes = service.render_page_image(ev.pdf_page_index, zoom=1.5)
                    st.session_state.rendered_page_bytes = img_bytes
                    st.session_state.rendered_page_num = ev.pdf_page_index
                    st.session_state.rendered_total_pages = _safe_get_total_pages(service)
                    st.rerun()

    with col_pdf_preview:
        st.markdown("**Physical PDF Page Visual Inspector:**")
        if st.session_state.rendered_page_bytes:
            total_pages = (
                st.session_state.get("rendered_total_pages")
                or _safe_get_total_pages(service)
                or doc_summary.get("total_pages", 76)
            )
            st.success(f"Displaying Physical PDF Page **{st.session_state.rendered_page_num}** of **{total_pages}**:")
            st.image(
                st.session_state.rendered_page_bytes,
                caption=f"Physical Page {st.session_state.rendered_page_num} of {total_pages} ({doc_summary.get('source_file', 'Document')})",
                use_container_width=True,
            )
        else:
            st.info("Click 'View Physical Page' on any cited evidence passage to inspect the original PDF page.")


# ── Section 6: Follow-Up Questions & Session History ──────────────────────────

if has_active_doc and len(st.session_state.qa_history) > 1:
    st.markdown("---")
    with st.expander(f"📜 Session Question History ({len(st.session_state.qa_history)} Questions Asked)", expanded=False):
        for idx, item in enumerate(st.session_state.qa_history, 1):
            st.markdown(f"**Q{idx}:** {item['question']}")
            st.markdown(f"**Status:** `{item['status']}`")
            st.caption(f"**Answer:** {item['answer'][:200]}…" if len(item['answer']) > 200 else f"**Answer:** {item['answer']}")
            st.markdown("---")


# ── Section 7: Regulatory Table Reasoner (Phase 4 Inspector) ──────────────────

if has_active_doc:
    st.markdown("---")
    with st.expander("🔬 Regulatory Table Reasoner & Taxon Inspector (Phase 4)", expanded=False):
        st.markdown(
            """
            Inspects structured regulatory tables with **column-association preservation** (Appendix I vs II vs III).  
            Strictly distinguishes **direct document listings** from **conditional taxonomic inferences**.
            """
        )

        col_t1, col_t2, col_t3 = st.columns([1.5, 1.2, 0.8])
        with col_t1:
            inspect_taxon = st.text_input("Taxon or Species Name:", value="Macaca mulatta")
        with col_t2:
            inspect_higher = st.text_input("Higher Taxon Premise (optional):", value="PRIMATES")
        with col_t3:
            st.write("")
            st.write("")
            run_table_inspect = st.button("Inspect Table", use_container_width=True)

        if run_table_inspect and inspect_taxon:
            with st.spinner(f"Inspecting regulatory tables for '{inspect_taxon}'..."):
                try:
                    res = service.inspect_table_taxon(
                        taxon_name=inspect_taxon,
                        higher_taxon=inspect_higher if inspect_higher else None,
                        strict_document_only=strict_policy,
                    )

                    status = res.get("status", "")
                    if status == "direct_listing":
                        st.success(f"**[DIRECT DOCUMENT LISTING]** Classification: **{res['classification']}**")
                    elif status == "conditional_inference":
                        st.warning(f"**[CONDITIONAL TAXONOMIC INFERENCE (NOT IN DOCUMENT)]** Classification: **{res['classification']}**")
                    else:
                        st.info(f"**[UNLISTED / INSUFFICIENT EVIDENCE]** Classification: **{res['classification']}**")

                    st.write(f"**Rationale:** {res['rationale']}")

                    if res.get("assumptions"):
                        st.markdown("**External Assumptions / Premises:**")
                        for a in res["assumptions"]:
                            st.caption(f"⚠ {a}")

                    if res.get("provenance"):
                        prov = res["provenance"]
                        st.markdown(
                            f"**Provenance:** Page `{prov['pdf_page_index']}`, Column `{prov['column_heading']}` │ Table `{prov['table_id']}` (Row {prov['row_index']})"
                        )
                        if res.get("verbatim_source_entry"):
                            st.code(res["verbatim_source_entry"])
                except Exception as exc:
                    st.error(f"Table inspection error: {exc}")
