"""
Sidebar component for BioLitAI-X.
Streamlit is imported inside the function per project conventions.
"""


def render_sidebar(session_state) -> str:
    """
    Render the full application sidebar and return the selected page name.

    Sections
    --------
    1. Logo + app title
    2. Navigation radio (returns selected page)
    3. Divider
    4. Pipeline status badge
    5. Active query chip
    6. Dataset stats
    7. Past sessions list

    Returns
    -------
    str : selected page name, one of:
          "Home" | "Analysis" | "Knowledge Graph" |
          "Hypotheses" | "Semantic Search" | "Chat"
    """
    import streamlit as st

    with st.sidebar:
        # ── Logo + Title ──────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align:center;padding:0.5rem 0 1rem 0;">
              <div style="font-size:2.8rem;margin-bottom:4px;">🧬</div>
              <div style="font-size:1.25rem;font-weight:700;
                         color:#F9FAFB;letter-spacing:0.02em;">BioLitAI-X</div>
              <div style="font-size:0.72rem;color:#9CA3AF;
                         letter-spacing:0.08em;text-transform:uppercase;">
                Biomedical Intelligence
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<hr style="border:none;border-top:1px solid #1F2937;margin:0 0 1rem 0;">',
            unsafe_allow_html=True,
        )

        # ── Navigation ────────────────────────────────────────────
        page_options = [
            "🏠  Home",
            "📊  Analysis",
            "🕸️  Knowledge Graph",
            "💡  Hypotheses",
            "🔍  Semantic Search",
            "💬  Chat",
        ]
        # Map display label → canonical page name
        _page_map = {
            "🏠  Home":           "Home",
            "📊  Analysis":       "Analysis",
            "🕸️  Knowledge Graph": "Knowledge Graph",
            "💡  Hypotheses":     "Hypotheses",
            "🔍  Semantic Search": "Semantic Search",
            "💬  Chat":           "Chat",
        }

        selected_display = st.radio(
            "Navigate",
            options=page_options,
            label_visibility="collapsed",
            key="sidebar_nav",
        )
        selected_page = _page_map.get(selected_display, "Home")

        st.markdown(
            '<hr style="border:none;border-top:1px solid #1F2937;margin:1rem 0;">',
            unsafe_allow_html=True,
        )

        # ── Pipeline status badge ─────────────────────────────────
        pipeline_complete = session_state.get("pipeline_complete", False)
        pipeline_status = session_state.get("pipeline_status", "idle")

        if pipeline_complete:
            status_color = "#10B981"
            status_bg = "#064E3B"
            status_text = "Complete"
            status_icon = "✓"
        elif pipeline_status == "running":
            status_color = "#F59E0B"
            status_bg = "#451A03"
            status_text = "Running"
            status_icon = "⟳"
        else:
            status_color = "#9CA3AF"
            status_bg = "#1C2539"
            status_text = "Idle"
            status_icon = "○"

        st.markdown(
            f"""
            <div style="margin-bottom:12px;">
              <div style="font-size:0.68rem;color:#6B7280;text-transform:uppercase;
                         letter-spacing:0.08em;margin-bottom:6px;">Pipeline</div>
              <span style="
                background:{status_bg};
                color:{status_color};
                border-radius:20px;
                padding:4px 12px;
                font-size:0.78rem;
                font-weight:700;
                letter-spacing:0.03em;
              ">{status_icon} {status_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Active query chip ─────────────────────────────────────
        current_query = session_state.get("current_query", "")
        if current_query:
            # Truncate long queries for display
            display_query = (
                current_query[:40] + "…" if len(current_query) > 40 else current_query
            )
            st.markdown(
                f"""
                <div style="margin-bottom:12px;">
                  <div style="font-size:0.68rem;color:#6B7280;text-transform:uppercase;
                             letter-spacing:0.08em;margin-bottom:6px;">Active Query</div>
                  <div style="
                    background:#1C2539;
                    border:1px solid #3B82F6;
                    border-radius:6px;
                    padding:6px 10px;
                    font-size:0.78rem;
                    color:#93C5FD;
                    word-break:break-word;
                    line-height:1.4;
                  ">
                    🔎 {display_query}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Dataset stats ─────────────────────────────────────────
        papers_df = session_state.get("papers_df")
        if papers_df is not None and hasattr(papers_df, "__len__") and len(papers_df) > 0:
            n_papers = len(papers_df)

            # Entity count from knowledge graph if available
            kg = session_state.get("knowledge_graph")
            n_entities = 0
            if kg is not None and hasattr(kg, "number_of_nodes"):
                n_entities = kg.number_of_nodes()

            st.markdown(
                f"""
                <div style="margin-bottom:12px;">
                  <div style="font-size:0.68rem;color:#6B7280;text-transform:uppercase;
                             letter-spacing:0.08em;margin-bottom:8px;">Dataset</div>
                  <div style="display:flex;gap:12px;">
                    <div style="
                      flex:1;background:#1C2539;border-radius:6px;
                      padding:8px 10px;text-align:center;
                    ">
                      <div style="font-size:1.1rem;font-weight:700;color:#3B82F6;">
                        {n_papers:,}
                      </div>
                      <div style="font-size:0.65rem;color:#9CA3AF;
                                 text-transform:uppercase;letter-spacing:0.06em;">
                        Papers
                      </div>
                    </div>
                    <div style="
                      flex:1;background:#1C2539;border-radius:6px;
                      padding:8px 10px;text-align:center;
                    ">
                      <div style="font-size:1.1rem;font-weight:700;color:#10B981;">
                        {n_entities:,}
                      </div>
                      <div style="font-size:0.65rem;color:#9CA3AF;
                                 text-transform:uppercase;letter-spacing:0.06em;">
                        Entities
                      </div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Past sessions list ────────────────────────────────────
        sessions = session_state.get("sessions", [])
        if sessions:
            st.markdown(
                '<div style="font-size:0.68rem;color:#6B7280;text-transform:uppercase;'
                'letter-spacing:0.08em;margin-bottom:8px;">Recent Queries</div>',
                unsafe_allow_html=True,
            )
            # Show up to 5 most recent
            for sess in sessions[-5:][::-1]:
                query_text = sess.get("query_text", "") or sess.get("query", "")
                session_id = sess.get("id", "")
                if not query_text:
                    continue
                short = query_text[:35] + "…" if len(query_text) > 35 else query_text
                btn_key = f"sess_chip_{session_id}_{hash(query_text) & 0xFFFF}"
                if st.button(
                    f"🕐 {short}",
                    key=btn_key,
                    help=query_text,
                    use_container_width=True,
                ):
                    session_state["current_query"] = query_text
                    # Attempt to reload session data from the database
                    try:
                        from database.db_manager import DatabaseManager
                        db = DatabaseManager()
                        import pandas as pd
                        papers = db.get_papers_by_query(query_text)
                        if papers:
                            from pipeline.cleaner import DataCleaner
                            cleaner = DataCleaner()
                            session_state["papers_df"] = cleaner.run_full_pipeline(papers)
                    except Exception:
                        pass
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.65rem;color:#374151;text-align:center;">'
            "BioLitAI-X · Phase 6"
            "</div>",
            unsafe_allow_html=True,
        )

    return selected_page
