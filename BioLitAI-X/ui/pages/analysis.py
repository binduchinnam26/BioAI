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
        with st.spinner("Building topic network…"):
            graph, err = _build_topic_graph_safe(session_state, papers_df)
        if graph is not None and graph.number_of_nodes() > 0:
            session_state["topic_graph"] = graph
        else:
            st.error(f"Topic network could not be built: {err}")
            return

    _render_topic_paper_table(session_state, papers_df, graph)


def _render_topic_paper_table(session_state, papers_df, graph):
    """
    Below the topic network, render one st.expander per topic listing
    up to 5 representative papers (title, year, first author, PubMed link).
    """
    import streamlit as st
    import json

    if graph is None or graph.number_of_nodes() == 0:
        return
    if papers_df is None or (hasattr(papers_df, "empty") and papers_df.empty):
        return

    # Reconstruct topic → pmid list from topic_model_results
    topic_results = session_state.get("topic_model_results")
    if not topic_results or not topic_results.get("topics"):
        return

    topics = topic_results["topics"]
    pmids = papers_df["pmid"].astype(str).tolist()

    # Build pmid → row lookup for fast access
    papers_lookup = {}
    for _, row in papers_df.iterrows():
        pid = str(row.get("pmid", "")).strip()
        if pid:
            papers_lookup[pid] = row

    # Group pmids by topic_id, skip outlier topic -1
    topic_pmids: dict = {}
    for i, tid in enumerate(topics):
        if i >= len(pmids):
            break
        tid = int(tid)
        if tid < 0:
            continue
        topic_pmids.setdefault(tid, []).append(pmids[i])

    if not topic_pmids:
        return

    st.markdown(
        "<hr style='border-color:#1F2937;margin:16px 0 12px 0;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.875rem;font-weight:600;color:#F9FAFB;'
        'margin-bottom:2px;">Papers by Topic</p>'
        '<p style="font-size:0.75rem;color:#6B7280;margin-bottom:12px;">'
        'Top 5 representative papers per discovered research topic</p>',
        unsafe_allow_html=True,
    )

    # Sort topics by paper count descending
    sorted_topics = sorted(
        [(tid, plist) for tid, plist in topic_pmids.items() if graph.has_node(tid)],
        key=lambda x: graph.nodes[x[0]].get("weight", 0),
        reverse=True,
    )

    for tid, pmid_list in sorted_topics:
        node_data = graph.nodes[tid]
        raw_words = node_data.get("top_words", [])
        _raw = [
            w[0] if isinstance(w, (list, tuple)) else str(w)
            for w in raw_words
        ]
        words = []
        for w in _raw:
            wl = w.lower()
            if any(
                wl.startswith(s.lower()) or s.lower().startswith(wl)
                for s in words
            ):
                continue
            words.append(w)
            if len(words) >= 3:
                break
        label = ", ".join(words) if words else f"Topic {tid}"
        paper_count = node_data.get("weight", len(pmid_list))
        color = node_data.get("color_hex", "#3B82F6")

        with st.expander(f"{label}  ·  {paper_count} papers", expanded=False):
            shown = 0
            # Sort by year descending so most recent papers appear first
            def _year_of(pid):
                r = papers_lookup.get(pid)
                if r is None:
                    return 0
                y = r.get("pub_year", 0)
                try:
                    return int(y)
                except Exception:
                    return 0

            for pmid in sorted(pmid_list, key=_year_of, reverse=True):
                if shown >= 5:
                    break
                row = papers_lookup.get(pmid)
                if row is None:
                    continue
                title = str(row.get("title", "")).strip()
                if not title or title.lower() == "nan":
                    continue

                year = row.get("pub_year", "")
                try:
                    year_str = str(int(year))
                except Exception:
                    year_str = "—"

                authors = row.get("authors", [])
                if isinstance(authors, str):
                    try:
                        authors = json.loads(authors)
                    except Exception:
                        authors = []
                first_author = ""
                if isinstance(authors, list) and authors:
                    a = authors[0]
                    first_author = (
                        a.get("name") or a.get("normalized_name") or ""
                        if isinstance(a, dict) else str(a)
                    )
                author_str = f"{first_author} et al." if first_author.strip() else ""
                meta = "&nbsp;·&nbsp;".join(filter(None, [author_str, year_str]))

                st.markdown(
                    f'<div style="background:#1C2539;border-radius:6px;'
                    f'padding:10px 14px;margin-bottom:8px;'
                    f'border-left:3px solid {color};">'
                    f'<div style="font-size:0.875rem;font-weight:600;color:#F9FAFB;'
                    f'margin-bottom:4px;line-height:1.45;">{title}</div>'
                    f'<div style="font-size:0.75rem;color:#9CA3AF;">{meta}'
                    f'{"&nbsp;·&nbsp;" if meta else ""}'
                    f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" '
                    f'target="_blank" style="color:#3B82F6;text-decoration:none;">'
                    f'PMID&nbsp;{pmid}</a></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                shown += 1

            if shown == 0:
                st.markdown(
                    '<p style="font-size:0.8rem;color:#6B7280;'
                    'padding:4px 0;">No paper details available for this topic.</p>',
                    unsafe_allow_html=True,
                )


def _build_topic_graph_safe(session_state, papers_df):
    """
    Build a topic NetworkX graph. Returns (graph, None) on success,
    (None, error_string) on failure. Never raises.
    """
    import importlib.util, pathlib, sys

    base = pathlib.Path(__file__).parent.parent.parent / "BioLitAI-X"
    if not base.exists():
        base = pathlib.Path(__file__).parent.parent  # fallback: project root

    def _direct_import(module_name, rel_path):
        """Import a single .py file without touching the package __init__."""
        if module_name in sys.modules:
            return sys.modules[module_name]
        full = base / rel_path
        spec = importlib.util.spec_from_file_location(module_name, full)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod

    try:
        tm_mod = _direct_import("_tm_standalone", "pipeline/topic_modeler.py")
        nb_mod = _direct_import("_nb_standalone", "pipeline/network_builder.py")
        TopicModeler = tm_mod.TopicModeler
        NetworkBuilder = nb_mod.NetworkBuilder
    except Exception as e:
        return None, f"import error: {e}"

    try:
        topic_results = session_state.get("topic_model_results")

        if (topic_results
                and topic_results.get("topic_info")
                and topic_results.get("topics")):
            topic_info = topic_results["topic_info"]
            topics = topic_results["topics"]
        else:
            abstracts = papers_df["abstract"].fillna("").tolist()
            modeler = TopicModeler()
            topics, _ = modeler.fit_transform(abstracts)
            topic_info = modeler.get_topic_summary()

        if not topic_info:
            return None, "topic modelling returned no topics (all documents assigned to outlier cluster)"

        pmids = papers_df["pmid"].astype(str).tolist()
        paper_assignments = [
            {"pmid": pmids[i], "topic_id": int(topics[i])}
            for i in range(min(len(pmids), len(topics)))
        ]

        graph = NetworkBuilder().build_topic_network({
            "topic_summary": topic_info,
            "paper_assignments": paper_assignments,
        })
        if graph.number_of_nodes() == 0:
            return None, "network builder produced an empty graph"
        return graph, None

    except Exception as e:
        import traceback
        return None, traceback.format_exc(limit=5)


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
