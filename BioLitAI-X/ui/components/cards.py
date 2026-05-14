"""
Reusable card components for BioLitAI-X.
Streamlit is imported inside each function per project conventions.
"""
import re


def _confidence_badge(confidence_score: float) -> str:
    """Return an HTML badge string based on a 0-1 confidence score."""
    if confidence_score >= 0.7:
        level, color, bg = "High", "#065F46", "#D1FAE5"
        dark_color, dark_bg = "#34D399", "#064E3B"
    elif confidence_score >= 0.45:
        level, color, bg = "Medium", "#92400E", "#FEF3C7"
        dark_color, dark_bg = "#FCD34D", "#451A03"
    else:
        level, color, bg = "Low", "#7F1D1D", "#FEE2E2"
        dark_color, dark_bg = "#F87171", "#450A0A"
    return (
        f'<span style="background:{dark_bg};color:{dark_color};'
        f'padding:3px 10px;border-radius:20px;font-size:0.72rem;'
        f'font-weight:700;text-transform:uppercase;letter-spacing:0.05em;">'
        f'{level}</span>'
    )


def _highlight_text(text: str, query: str) -> str:
    """Wrap occurrences of query terms in a highlight span."""
    if not query or not text:
        return text
    tokens = [t.strip() for t in re.split(r'\s+', query) if len(t.strip()) >= 3]
    result = text
    for token in tokens:
        pattern = re.compile(re.escape(token), re.IGNORECASE)
        result = pattern.sub(
            lambda m: (
                f'<mark style="background:rgba(59,130,246,0.3);'
                f'color:#93C5FD;border-radius:2px;padding:0 2px;">'
                f'{m.group()}</mark>'
            ),
            result,
        )
    return result


def render_hypothesis_card(hyp: dict, index: int, on_expand=None):
    """
    Render a hypothesis card with title, confidence badge, hypothesis text,
    and an expandable section for rationale, experiment, and PMIDs.

    Parameters
    ----------
    hyp      : hypothesis dict with keys:
               concept_a, concept_b, hypothesis_text, confidence_score,
               rationale, suggested_experiment, evidence_pmids, novelty,
               supporting_evidence, created_at
    index    : numeric index (used for unique expander keys)
    on_expand: unused; kept for API compatibility
    """
    import streamlit as st

    concept_a = hyp.get("concept_a", "Concept A")
    concept_b = hyp.get("concept_b", "Concept B")
    hyp_text = hyp.get("hypothesis_text", "")
    confidence = float(hyp.get("confidence_score", 0.25))
    rationale = hyp.get("rationale", "")
    experiment = hyp.get("suggested_experiment", "")
    evidence_pmids = hyp.get("evidence_pmids", [])
    novelty = hyp.get("novelty", "")
    support_evidence = hyp.get("supporting_evidence", "")
    created_at = hyp.get("created_at", "")

    badge_html = _confidence_badge(confidence)

    # Card header
    st.markdown(
        f"""
        <div style="
          background:#111827;
          border:1px solid #1F2937;
          border-radius:10px;
          padding:18px 20px 12px 20px;
          margin-bottom:4px;
        ">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
            <div style="flex:1;min-width:0;">
              <div style="font-size:0.85rem;font-weight:700;color:#9CA3AF;
                         text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">
                Hypothesis {index + 1}
              </div>
              <div style="font-size:1rem;font-weight:700;color:#F9FAFB;line-height:1.4;">
                <span style="color:#4E9AF1;">{concept_a}</span>
                <span style="color:#9CA3AF;margin:0 8px;">→</span>
                <span style="color:#34C78A;">{concept_b}</span>
              </div>
            </div>
            <div style="flex-shrink:0;margin-top:4px;">{badge_html}</div>
          </div>
          <div style="margin-top:12px;font-size:0.875rem;color:#D1D5DB;
                     line-height:1.6;font-style:italic;">
            "{hyp_text}"
          </div>
          {
            f'<div style="margin-top:10px;font-size:0.72rem;color:#9B72CF;'
            f'font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">'
            f'✦ {novelty}</div>'
            if novelty else ""
          }
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Expandable detail section
    with st.expander(f"Details — Hypothesis {index + 1}", expanded=False):
        if rationale:
            st.markdown(
                f'<div style="margin-bottom:12px;">'
                f'<div style="font-size:0.75rem;font-weight:700;color:#9CA3AF;'
                f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">'
                f'Biological Rationale</div>'
                f'<div style="font-size:0.875rem;color:#D1D5DB;line-height:1.6;">'
                f'{rationale}</div></div>',
                unsafe_allow_html=True,
            )

        if experiment:
            st.markdown(
                f'<div style="margin-bottom:12px;">'
                f'<div style="font-size:0.75rem;font-weight:700;color:#9CA3AF;'
                f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">'
                f'Suggested Experiment</div>'
                f'<div style="font-size:0.875rem;color:#D1D5DB;line-height:1.6;'
                f'background:#1C2539;border-radius:6px;padding:10px 14px;">'
                f'{experiment}</div></div>',
                unsafe_allow_html=True,
            )

        if support_evidence:
            st.markdown(
                f'<div style="margin-bottom:12px;">'
                f'<div style="font-size:0.75rem;font-weight:700;color:#9CA3AF;'
                f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">'
                f'Supporting Evidence</div>'
                f'<div style="font-size:0.82rem;color:#9CA3AF;line-height:1.6;'
                f'font-family:monospace;">'
                f'{support_evidence}</div></div>',
                unsafe_allow_html=True,
            )

        if evidence_pmids:
            pmid_links = " ".join(
                f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" '
                f'target="_blank" rel="noopener" '
                f'style="background:#1C2539;border:1px solid #1F2937;border-radius:4px;'
                f'padding:2px 8px;font-size:0.75rem;color:#3B82F6;text-decoration:none;'
                f'margin-right:4px;display:inline-block;margin-bottom:4px;">'
                f'PMID:{pmid}</a>'
                for pmid in evidence_pmids
            )
            st.markdown(
                f'<div>'
                f'<div style="font-size:0.75rem;font-weight:700;color:#9CA3AF;'
                f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">'
                f'Supporting Papers</div>'
                f'{pmid_links}'
                f'</div>',
                unsafe_allow_html=True,
            )

        if created_at:
            st.markdown(
                f'<div style="margin-top:10px;font-size:0.7rem;color:#6B7280;">'
                f'Generated: {str(created_at)[:19].replace("T", " ")} UTC'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_paper_card(
    paper: dict,
    similarity_score: float = None,
    query_highlight: str = None,
):
    """
    Render a search result card for a single paper.

    Parameters
    ----------
    paper            : paper dict with keys pmid, title, authors, journal,
                       abstract, pub_year, doi
    similarity_score : 0-1 cosine similarity; shown as a coloured progress bar
    query_highlight  : query string whose tokens are highlighted in the abstract
    """
    import streamlit as st

    pmid = str(paper.get("pmid", ""))
    title = paper.get("title") or "Untitled"
    authors = paper.get("authors") or []
    journal = paper.get("journal") or ""
    abstract = str(paper.get("abstract") or "")
    pub_year = paper.get("pub_year") or paper.get("pub_date", "")
    doi = paper.get("doi") or ""

    # Author string
    if isinstance(authors, list):
        author_names = [
            a.get("name", "") if isinstance(a, dict) else str(a)
            for a in authors[:3]
        ]
        author_str = "; ".join(n for n in author_names if n)
        if len(authors) > 3:
            author_str += f" et al. (+{len(authors)-3})"
    else:
        author_str = str(authors)

    # Abstract excerpt with optional highlighting
    excerpt = abstract[:300].replace("\n", " ").strip()
    if len(abstract) > 300:
        excerpt += "…"
    if query_highlight:
        excerpt = _highlight_text(excerpt, query_highlight)

    # Title with PubMed link
    if pmid:
        title_html = (
            f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" '
            f'target="_blank" rel="noopener" '
            f'style="color:#4E9AF1;text-decoration:none;font-size:0.95rem;'
            f'font-weight:700;line-height:1.4;">{title}</a>'
        )
    else:
        title_html = (
            f'<span style="color:#F9FAFB;font-size:0.95rem;'
            f'font-weight:700;line-height:1.4;">{title}</span>'
        )

    # Similarity bar
    sim_html = ""
    if similarity_score is not None:
        pct = int(similarity_score * 100)
        if pct >= 80:
            bar_color = "#10B981"
        elif pct >= 60:
            bar_color = "#F59E0B"
        else:
            bar_color = "#3B82F6"
        sim_html = f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
          <div style="flex:1;background:#1F2937;border-radius:4px;height:6px;">
            <div style="width:{pct}%;background:{bar_color};border-radius:4px;height:100%;
                       transition:width 400ms ease;"></div>
          </div>
          <span style="font-size:0.75rem;font-weight:700;color:{bar_color};
                      white-space:nowrap;">{pct}% match</span>
        </div>
        """

    # PMID and year badges
    badges = ""
    if pmid:
        badges += (
            f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" '
            f'target="_blank" rel="noopener" '
            f'style="background:#1C2539;border:1px solid #1F2937;border-radius:4px;'
            f'padding:2px 8px;font-size:0.72rem;color:#3B82F6;'
            f'text-decoration:none;margin-right:6px;">PMID:{pmid}</a>'
        )
    if pub_year:
        badges += (
            f'<span style="background:#1C2539;border:1px solid #1F2937;'
            f'border-radius:4px;padding:2px 8px;font-size:0.72rem;color:#9CA3AF;">'
            f'{pub_year}</span>'
        )

    html = f"""
    <div style="
      background:#111827;
      border:1px solid #1F2937;
      border-radius:10px;
      padding:16px 20px;
      margin-bottom:12px;
      transition:border-color 150ms ease;
    "
    onmouseover="this.style.borderColor='#3B82F6';"
    onmouseout="this.style.borderColor='#1F2937';"
    >
      {sim_html}
      <div style="margin-bottom:6px;">{title_html}</div>
      <div style="font-size:0.78rem;color:#9CA3AF;margin-bottom:8px;line-height:1.4;">
        {author_str}
        {f' &nbsp;|&nbsp; <em>{journal}</em>' if journal else ''}
      </div>
      <div style="font-size:0.82rem;color:#D1D5DB;line-height:1.6;margin-bottom:10px;">
        {excerpt}
      </div>
      <div>{badges}</div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


def render_chat_message(role: str, content: str, sources: list = None):
    """
    Render a single chat message bubble.

    Parameters
    ----------
    role    : 'user' or 'assistant'
    content : message text (may contain markdown-style line breaks)
    sources : list of dicts with keys 'pmid' and/or 'title' (assistant only)
    """
    import streamlit as st

    content_escaped = (
        content.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )

    if role == "user":
        html = f"""
        <div style="display:flex;justify-content:flex-end;margin-bottom:12px;">
          <div style="
            max-width:75%;
            background:#3B82F6;
            border-radius:18px 18px 4px 18px;
            padding:12px 16px;
            color:#FFFFFF;
            font-size:0.875rem;
            line-height:1.6;
            box-shadow:0 2px 8px rgba(59,130,246,0.3);
          ">
            {content_escaped}
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    else:
        # Assistant bubble
        html = f"""
        <div style="display:flex;justify-content:flex-start;margin-bottom:12px;">
          <div style="margin-right:10px;flex-shrink:0;margin-top:4px;">
            <div style="
              width:32px;height:32px;border-radius:50%;
              background:linear-gradient(135deg,#3B82F6,#9B72CF);
              display:flex;align-items:center;justify-content:center;
              font-size:0.9rem;
            ">🧬</div>
          </div>
          <div style="
            max-width:80%;
            background:#1C2539;
            border:1px solid #1F2937;
            border-radius:4px 18px 18px 18px;
            padding:12px 16px;
            color:#F9FAFB;
            font-size:0.875rem;
            line-height:1.6;
          ">
            {content_escaped}
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

        # Sources collapsible section
        if sources:
            valid_sources = [
                s for s in sources
                if isinstance(s, dict) and (s.get("pmid") or s.get("title"))
            ]
            if valid_sources:
                with st.expander("Sources", expanded=False):
                    for src in valid_sources:
                        pmid = src.get("pmid", "")
                        title = src.get("title", "")
                        if pmid:
                            st.markdown(
                                f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" '
                                f'target="_blank" rel="noopener" '
                                f'style="color:#3B82F6;text-decoration:none;'
                                f'font-size:0.82rem;">'
                                f'PMID:{pmid} — {title[:80] if title else "View on PubMed"}'
                                f'</a><br>',
                                unsafe_allow_html=True,
                            )


def render_gap_card(gap: dict, index: int):
    """
    Render a single research gap card for the Hypotheses page gap browser.

    Parameters
    ----------
    gap   : gap dict with keys: entity_a/concept_a, entity_b/concept_b,
            gap_type, score, evidence_pmids, description
    index : numeric index for display
    """
    import streamlit as st

    entity_a = gap.get("entity_a") or gap.get("concept_a", "Unknown")
    entity_b = gap.get("entity_b") or gap.get("concept_b", "")
    gap_type = gap.get("gap_type", "structural")
    score = float(gap.get("score", 0.0))
    description = gap.get("description", "")
    evidence_pmids = gap.get("evidence_pmids", [])

    _type_colors = {
        "structural":   "#3B82F6",
        "cross_domain": "#8B5CF6",
        "temporal":     "#F59E0B",
    }
    color = _type_colors.get(gap_type, "#3B82F6")
    type_label = gap_type.replace("_", " ").title()

    pmid_badges = ""
    if evidence_pmids:
        for pmid in list(evidence_pmids)[:4]:
            pmid_badges += (
                f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" '
                f'target="_blank" rel="noopener" '
                f'style="background:#1C2539;border:1px solid #1F2937;border-radius:4px;'
                f'padding:2px 7px;font-size:0.68rem;color:#3B82F6;'
                f'text-decoration:none;margin-right:4px;">PMID:{pmid}</a>'
            )

    html = f"""
    <div style="
      background:#111827;
      border:1px solid #1F2937;
      border-left:3px solid {color};
      border-radius:0 8px 8px 0;
      padding:12px 16px;
      margin-bottom:8px;
    ">
      <div style="display:flex;justify-content:space-between;align-items:center;
                 margin-bottom:8px;">
        <div>
          <span style="background:{color}20;color:{color};border-radius:10px;
                      padding:2px 10px;font-size:0.68rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.05em;margin-right:8px;">
            {type_label}
          </span>
          <span style="font-size:0.875rem;font-weight:600;color:#F9FAFB;">
            {entity_a}
          </span>
          <span style="color:#6B7280;margin:0 8px;">↔</span>
          <span style="font-size:0.875rem;font-weight:600;color:#F9FAFB;">
            {entity_b}
          </span>
        </div>
        <span style="font-size:0.78rem;color:#9CA3AF;white-space:nowrap;margin-left:12px;">
          Score: <b style="color:#F9FAFB;">{score:.2f}</b>
        </span>
      </div>
      {f'<div style="font-size:0.8rem;color:#9CA3AF;margin-bottom:8px;">{description}</div>' if description else ""}
      {f'<div>{pmid_badges}</div>' if pmid_badges else ""}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
