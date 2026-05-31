"""
Semantic Search page — FAISS-powered cosine-similarity search over the corpus.
Streamlit is imported inside each function.
"""


def render_semantic_search(session_state):
    """
    Render the Semantic Search page.

    Layout
    ------
    1. Search input + top-k slider
    2. Results list (paper cards with similarity scores)
    3. Corpus stats sidebar column
    """
    import streamlit as st

    papers_df = session_state.get("papers_df")
    embedder = session_state.get("embedder")

    if papers_df is None or (hasattr(papers_df, "empty") and papers_df.empty):
        _no_data_placeholder()
        return

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        '<h2 style="font-size:1.4rem;font-weight:700;color:#F9FAFB;'
        'margin-bottom:0.25rem;">Semantic Search</h2>'
        '<p style="font-size:0.875rem;color:#9CA3AF;margin-bottom:1.25rem;">'
        'Search the corpus using natural-language meaning, not just keywords. '
        'Powered by PubMedBERT sentence embeddings + FAISS cosine similarity.</p>',
        unsafe_allow_html=True,
    )

    if embedder is None:
        st.warning(
            "The semantic embedding index is not loaded. "
            "Ensure the pipeline completed the Embedding step, "
            "or check that `sentence-transformers` and `faiss-cpu` are installed."
        )
        return

    # ── Search form ───────────────────────────────────────────────────────────
    col_q, col_k = st.columns([5, 1])
    with col_q:
        search_query = st.text_input(
            "Semantic query",
            placeholder="Describe the concept, mechanism, or finding you're looking for…",
            label_visibility="collapsed",
            key="sem_query",
        )
    with col_k:
        top_k = st.number_input(
            "Top K",
            min_value=1,
            max_value=50,
            value=10,
            label_visibility="collapsed",
            key="sem_top_k",
        )

    search_clicked = st.button(
        "🔍 Search",
        type="primary",
        key="sem_search_btn",
    )

    # ── Results ───────────────────────────────────────────────────────────────
    if search_clicked and search_query.strip():
        _run_search(papers_df, embedder, search_query.strip(), int(top_k))
    elif st.session_state.get("_sem_results"):
        _display_results(
            st.session_state["_sem_results"],
            st.session_state.get("_sem_query_text", ""),
        )


def _no_data_placeholder():
    import streamlit as st

    st.markdown(
        """
        <div style="text-align:center;padding:80px 20px;">
          <div style="font-size:3rem;margin-bottom:12px;">🔍</div>
          <h3 style="color:#F9FAFB;margin-bottom:8px;">No Corpus Loaded</h3>
          <p style="color:#9CA3AF;max-width:400px;margin:0 auto;">
            Run a query on the <b>Home</b> page to build the semantic index,
            then return here to search it.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _run_search(papers_df, embedder, query: str, top_k: int):
    import streamlit as st

    with st.spinner("Searching semantic index…"):
        try:
            hits = embedder.semantic_search(query, top_k=top_k)
        except Exception as exc:
            st.error(f"Semantic search failed: {exc}")
            return

    if not hits:
        st.info("No results found. Try rephrasing your query.")
        return

    # ── Relevance gate ────────────────────────────────────────────────────────
    # Cosine similarity on L2-normalised vectors: 1.0 = identical, 0.0 = orthogonal.
    # If the BEST match in the entire corpus scores below 0.50, the query topic
    # is not represented in this dataset — show a clear "not relevant" message
    # instead of misleading low-confidence results.
    _RELEVANCE_THRESHOLD = 0.50
    top_score = max((h.get("score", 0.0) for h in hits), default=0.0)
    if top_score < _RELEVANCE_THRESHOLD:
        st.markdown(
            f"""
            <div style="text-align:center;padding:60px 20px;">
              <div style="font-size:2.5rem;margin-bottom:12px;">🔭</div>
              <h3 style="color:#F9FAFB;margin-bottom:8px;">Topic Not Found in Corpus</h3>
              <p style="color:#9CA3AF;max-width:480px;margin:0 auto 16px auto;line-height:1.6;">
                The current corpus does not contain papers related to
                <b style="color:#3B82F6;">"{query[:80]}{'…' if len(query) > 80 else ''}"</b>.
                The highest similarity score across all {len(papers_df):,} papers
                was <b style="color:#F59E0B;">{int(top_score * 100)}%</b>
                — below the relevance threshold of {int(_RELEVANCE_THRESHOLD * 100)}%.
              </p>
              <p style="color:#6B7280;font-size:0.82rem;max-width:420px;margin:0 auto;">
                Try searching for topics related to your active query, or run a new
                pipeline query on the <b>Home</b> page to load a different corpus.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Map pmid → paper dict
    pmid_to_paper = {str(row.get("pmid", "")): row.to_dict() for _, row in papers_df.iterrows()}

    results = []
    for hit in hits:
        pmid = str(hit.get("pmid", ""))
        paper = pmid_to_paper.get(pmid)
        if paper:
            results.append({
                "paper": paper,
                "score": hit.get("score", 0.0),
                "rank": hit.get("rank", 0),
            })

    import streamlit as st
    st.session_state["_sem_results"] = results
    st.session_state["_sem_query_text"] = query

    _display_results(results, query)


def _build_kg_entity_words(session_state) -> set:
    """
    Extract individual words from all knowledge-graph node names.
    Returns a lower-cased set of biomedical term words (≥4 chars) that can
    be used to filter query tokens before highlighting.
    """
    import re as _re
    kg = session_state.get("knowledge_graph")
    entity_words = set()
    if kg is not None and hasattr(kg, "nodes"):
        for node, data in kg.nodes(data=True):
            name = data.get("name", str(node))
            # Split on whitespace, hyphens, slashes to get individual words
            for word in _re.split(r'[\s\-/]+', name):
                word = word.strip().lower()
                if len(word) >= 4:
                    entity_words.add(word)
    return entity_words


def _display_results(results: list, query: str):
    import streamlit as st

    if not results:
        st.info("No results.")
        return

    st.markdown(
        f'<p style="font-size:0.82rem;color:#9CA3AF;margin-bottom:12px;">'
        f'{len(results)} results for: <b style="color:#3B82F6;">{query}</b></p>',
        unsafe_allow_html=True,
    )

    # Build KG entity word set for smart biomedical-only highlighting
    kg_entities = _build_kg_entity_words(st.session_state)

    try:
        from ui.components.cards import render_paper_card
        for hit in results:
            render_paper_card(
                paper=hit["paper"],
                similarity_score=hit.get("score"),
                query_highlight=query,
                kg_entities=kg_entities if kg_entities else None,
            )
    except Exception:
        # Fallback plain rendering
        for hit in results:
            paper = hit["paper"]
            score = hit.get("score", 0.0)
            title = paper.get("title", "Untitled")
            pmid = paper.get("pmid", "")
            journal = paper.get("journal", "")
            year = paper.get("pub_year", "")

            st.markdown(
                f"""
                <div style="background:#111827;border:1px solid #1F2937;
                            border-radius:8px;padding:12px 16px;margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;">
                    <a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                       target="_blank" rel="noopener"
                       style="font-size:0.9rem;font-weight:600;color:#3B82F6;
                              text-decoration:none;">{title}</a>
                    <span style="font-size:0.72rem;color:#10B981;
                                 font-weight:600;white-space:nowrap;margin-left:8px;">
                      {score:.3f}
                    </span>
                  </div>
                  <div style="font-size:0.75rem;color:#6B7280;margin-top:4px;">
                    {journal}{' · ' if journal and year else ''}{year}
                    {' · PMID: ' + pmid if pmid else ''}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
