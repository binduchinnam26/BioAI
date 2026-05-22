"""
Knowledge Graph page — entity-relationship graph explorer with gap overlay.
Streamlit is imported inside each function.
"""


def render_knowledge_graph_page(session_state):
    """
    Thin wrapper that adds a gap-highlight toggle at the page level, then
    delegates to the main render_knowledge_graph implementation.
    """
    import streamlit as st

    st.markdown(
        '<h2 style="color:#F9FAFB;font-weight:700;margin-bottom:0.25rem;">'
        '🕸️ Knowledge Graph</h2>',
        unsafe_allow_html=True,
    )

    # Gap highlight toggle
    kg = session_state.get("knowledge_graph")
    gap_report = session_state.get("gap_report", [])

    if gap_report:
        highlight = st.checkbox(
            f"Highlight {len(gap_report)} research gap nodes",
            value=True,
            key="kg_gap_highlight_toggle",
            help=(
                "When enabled, nodes involved in detected research gaps "
                "are highlighted with a pulsing yellow border."
            ),
        )
    else:
        highlight = False

    # Store toggle value in session state for the underlying renderer
    session_state["kg_highlight_gaps"] = highlight

    render_knowledge_graph(session_state)


def render_knowledge_graph(session_state):
    """
    Render the Knowledge Graph page.

    Layout
    ------
    Left panel (st.columns): filter controls
    Main area: PyVis directed graph with entity-type colours + gap pulse overlay
    Below: relationship table with sort/filter/pagination/CSV export
    """
    import streamlit as st

    papers_df = session_state.get("papers_df")
    kg_graph = session_state.get("knowledge_graph")

    if papers_df is None or kg_graph is None:
        _no_data_placeholder()
        return

    if kg_graph.number_of_nodes() == 0:
        st.info(
            "The knowledge graph is empty. "
            "This usually means no biomedical entities were extracted — "
            "ensure scispacy and the UMLS linker are installed."
        )
        return

    # ── Page header ────────────────────────────────────────────────────────────
    st.markdown(
        '<h2 style="font-size:1.4rem;font-weight:700;color:#F9FAFB;'
        'margin-bottom:0.25rem;">Knowledge Graph</h2>'
        f'<p style="font-size:0.875rem;color:#9CA3AF;margin-bottom:1.25rem;">'
        f'{kg_graph.number_of_nodes():,} entities · '
        f'{kg_graph.number_of_edges():,} relationships</p>',
        unsafe_allow_html=True,
    )

    gap_report = session_state.get("gap_report", [])
    gap_pairs = [
        (g.get("entity_a", g.get("concept_a", "")),
         g.get("entity_b", g.get("concept_b", "")))
        for g in gap_report
    ]

    try:
        from visualization.graph_viz import render_knowledge_graph as _render_kg
        _render_kg(
            graph=kg_graph,
            papers_df=papers_df,
            highlight_gaps=bool(gap_pairs),
            gap_pairs=gap_pairs,
            key_prefix="kg_main",
        )
    except Exception as exc:
        st.error(f"Knowledge graph rendering failed: {exc}")



def _no_data_placeholder():
    import streamlit as st

    st.markdown(
        """
        <div style="text-align:center;padding:80px 20px;">
          <div style="font-size:3rem;margin-bottom:12px;">🕸️</div>
          <h3 style="color:#F9FAFB;margin-bottom:8px;">Knowledge Graph Not Built</h3>
          <p style="color:#9CA3AF;max-width:440px;margin:0 auto;">
            Run a query on the <b>Home</b> page to extract entities and build
            the knowledge graph. Requires scispacy with the UMLS entity linker.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_gap_summary(gap_report: list):
    import streamlit as st

    with st.expander(f"🔍 Research Gaps ({len(gap_report)})", expanded=False):
        type_colors = {
            "structural":   "#3B82F6",
            "cross_domain": "#8B5CF6",
            "temporal":     "#F59E0B",
        }

        # Sort by score descending
        sorted_gaps = sorted(gap_report, key=lambda g: g.get("score", 0), reverse=True)

        for i, gap in enumerate(sorted_gaps[:20]):
            entity_a = gap.get("entity_a", gap.get("concept_a", ""))
            entity_b = gap.get("entity_b", gap.get("concept_b", ""))
            gap_type = gap.get("type", "structural")
            score = gap.get("score", 0.0)
            color = type_colors.get(gap_type, "#3B82F6")
            type_label = gap_type.replace("_", " ").title()

            st.markdown(
                f"""
                <div style="border-left:3px solid {color};padding:8px 14px;
                            margin-bottom:6px;background:#111827;border-radius:0 6px 6px 0;">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                      <span style="background:{color}20;color:{color};border-radius:10px;
                                  padding:1px 8px;font-size:0.68rem;font-weight:600;
                                  margin-right:8px;">{type_label}</span>
                      <span style="font-size:0.85rem;color:#F9FAFB;font-weight:500;">
                        {entity_a}
                      </span>
                      <span style="color:#6B7280;margin:0 8px;">↔</span>
                      <span style="font-size:0.85rem;color:#F9FAFB;font-weight:500;">
                        {entity_b}
                      </span>
                    </div>
                    <span style="font-size:0.75rem;color:#9CA3AF;">
                      Score: <b style="color:#F9FAFB;">{score:.2f}</b>
                    </span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if len(sorted_gaps) > 20:
            st.markdown(
                f'<p style="font-size:0.78rem;color:#6B7280;text-align:center;">'
                f'Showing top 20 of {len(sorted_gaps)} gaps. '
                f'Go to <b>Hypotheses</b> for AI-generated explanations.</p>',
                unsafe_allow_html=True,
            )
