"""
Analysis page — publication trends, topic evolution, keyword frequency,
author productivity, and bibliometric network visualisations.
Streamlit is imported inside each function.
"""


def render_analysis(session_state):
    """
    Render the Analysis page.

    Tabs
    ----
    1. Trends          – publication trend + topic evolution charts
    2. Keywords        – top-keyword bar chart
    3. Authors         – author productivity chart
    4. Co-authorship   – co-authorship network (PyVis)
    5. Keyword Network – keyword co-occurrence network (PyVis)
    6. Topic Network   – topic similarity network (PyVis)
    """
    import streamlit as st

    papers_df = session_state.get("papers_df")
    if papers_df is None or (hasattr(papers_df, "empty") and papers_df.empty):
        _no_data_placeholder()
        return

    st.markdown(
        '<h2 style="font-size:1.4rem;font-weight:700;color:#F9FAFB;'
        'margin-bottom:0.25rem;">Analysis</h2>'
        '<p style="font-size:0.875rem;color:#9CA3AF;margin-bottom:1.25rem;">'
        f'Corpus: <b style="color:#3B82F6;">'
        f'{session_state.get("current_query","")}</b> '
        f'&nbsp;·&nbsp; {len(papers_df):,} papers</p>',
        unsafe_allow_html=True,
    )

    tab_labels = [
        "📈 Trends",
        "🏷️ Keywords",
        "👤 Authors",
        "🤝 Co-authorship",
        "🔑 Keyword Net",
        "🔵 Topic Net",
    ]
    tabs = st.tabs(tab_labels)

    # ── Tab 0: Trends ─────────────────────────────────────────────────────────
    with tabs[0]:
        _render_trends_tab(session_state, papers_df)

    # ── Tab 1: Keywords ───────────────────────────────────────────────────────
    with tabs[1]:
        _render_keywords_tab(papers_df)

    # ── Tab 2: Authors ────────────────────────────────────────────────────────
    with tabs[2]:
        _render_authors_tab(papers_df)

    # ── Tab 3: Co-authorship Network ──────────────────────────────────────────
    with tabs[3]:
        _render_coauthor_tab(session_state)

    # ── Tab 4: Keyword Co-occurrence Network ──────────────────────────────────
    with tabs[4]:
        _render_keyword_net_tab(session_state, papers_df)

    # ── Tab 5: Topic Network ──────────────────────────────────────────────────
    with tabs[5]:
        _render_topic_net_tab(session_state, papers_df)


def _no_data_placeholder():
    import streamlit as st

    st.markdown(
        """
        <div style="text-align:center;padding:80px 20px;">
          <div style="font-size:3rem;margin-bottom:12px;">📊</div>
          <h3 style="color:#F9FAFB;margin-bottom:8px;">No Data Yet</h3>
          <p style="color:#9CA3AF;max-width:400px;margin:0 auto;">
            Run a query on the <b>Home</b> page to generate analysis data.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_trends_tab(session_state, papers_df):
    import streamlit as st
    from visualization.trend_charts import render_publication_trend, render_topic_evolution

    render_publication_trend(papers_df)

    topics_over_time = None
    topic_results = session_state.get("topic_model_results")
    if isinstance(topic_results, dict):
        topics_over_time = topic_results.get("topics_over_time")

    st.markdown("---")
    if topics_over_time is not None:
        render_topic_evolution(topics_over_time)
    else:
        st.info(
            "Topic evolution data is not available. "
            "Ensure BERTopic is installed and the pipeline has run successfully."
        )


def _render_keywords_tab(papers_df):
    from visualization.trend_charts import render_top_keywords
    render_top_keywords(papers_df)


def _render_authors_tab(papers_df):
    from visualization.trend_charts import render_author_productivity
    render_author_productivity(papers_df)


def _render_coauthor_tab(session_state):
    import streamlit as st

    graph = session_state.get("coauthor_graph")

    if graph is None or graph.number_of_nodes() == 0:
        st.info(
            "Co-authorship network is empty. "
            "This may occur if the corpus has fewer than 2 authors per paper."
        )
        return

    try:
        from visualization.network_viz import render_coauthorship_network
        render_coauthorship_network(graph)
    except Exception as exc:
        st.error(f"Could not render co-authorship network: {exc}")


def _render_keyword_net_tab(session_state, papers_df):
    import streamlit as st

    graph = session_state.get("keyword_graph")

    if graph is None or graph.number_of_nodes() == 0:
        st.info(
            "Keyword network is empty. Papers may not have enough keyword overlap."
        )
        return

    try:
        from visualization.network_viz import render_keyword_network
        render_keyword_network(graph)
    except Exception as exc:
        st.error(f"Could not render keyword network: {exc}")


def _render_topic_net_tab(session_state, papers_df):
    import streamlit as st

    graph = session_state.get("topic_graph")

    if graph is None or graph.number_of_nodes() == 0:
        st.info(
            "Topic network is not available. "
            "Ensure BERTopic is installed and the pipeline has completed topic modelling."
        )
        return

    try:
        from visualization.network_viz import render_topic_network
        render_topic_network(graph)
    except Exception as exc:
        st.error(f"Could not render topic network: {exc}")


def _render_network_controls(title: str, legend: str):
    import streamlit as st

    st.markdown(
        f'<h4 style="font-size:0.95rem;font-weight:600;color:#F9FAFB;'
        f'margin-bottom:4px;">{title}</h4>'
        f'<p style="font-size:0.75rem;color:#6B7280;margin-bottom:12px;">'
        f'{legend}</p>',
        unsafe_allow_html=True,
    )


def _get_network_filters(prefix: str, graph) -> dict:
    import streamlit as st

    with st.expander("⚙️ Filter & Search", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            min_link = st.slider(
                "Min edge weight",
                min_value=1,
                max_value=10,
                value=1,
                key=f"{prefix}_min_link",
            )
        with col2:
            min_size = st.slider(
                "Min node connections",
                min_value=1,
                max_value=20,
                value=1,
                key=f"{prefix}_min_size",
            )
        with col3:
            search = st.text_input(
                "Highlight node",
                placeholder="Search label…",
                key=f"{prefix}_search",
            )

        # Community filter
        communities = sorted(set(
            data.get("community", 0)
            for _, data in graph.nodes(data=True)
        ))
        community_options = ["All"] + [str(c) for c in communities]
        community_filter = st.selectbox(
            "Community",
            community_options,
            key=f"{prefix}_community",
        )
        community_val = None if community_filter == "All" else int(community_filter)

    return {
        "min_link": min_link,
        "min_size": min_size,
        "search": search,
        "community": community_val,
    }


def _render_network_stats(graph):
    import streamlit as st
    import networkx as nx

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    density = nx.density(graph) if n_nodes > 1 else 0.0
    try:
        avg_clustering = nx.average_clustering(graph)
    except Exception:
        avg_clustering = 0.0
    components = nx.number_connected_components(graph) if not graph.is_directed() else "—"

    st.markdown(
        f"""
        <div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:8px;
                    padding:10px 14px;background:#111827;border:1px solid #1F2937;
                    border-radius:8px;">
          <div style="font-size:0.78rem;color:#9CA3AF;">
            <b style="color:#3B82F6;">{n_nodes:,}</b> nodes
          </div>
          <div style="font-size:0.78rem;color:#9CA3AF;">
            <b style="color:#10B981;">{n_edges:,}</b> edges
          </div>
          <div style="font-size:0.78rem;color:#9CA3AF;">
            Density: <b style="color:#F9FAFB;">{density:.4f}</b>
          </div>
          <div style="font-size:0.78rem;color:#9CA3AF;">
            Avg clustering: <b style="color:#F9FAFB;">{avg_clustering:.3f}</b>
          </div>
          <div style="font-size:0.78rem;color:#9CA3AF;">
            Components: <b style="color:#F9FAFB;">{components}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
