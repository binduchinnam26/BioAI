"""
Home page — query input, pipeline execution, and past session chips.
Streamlit is imported inside the function per project conventions.
"""


def render_home(session_state):
    """
    Render the BioLitAI-X home page.

    Layout
    ------
    1. Hero section (centered title + subtitle)
    2. Search section (query input, max-results slider, run button)
    3. Past query chips (from session_state['sessions'])
    4. Pipeline execution with step-by-step stepper animation
    """
    import streamlit as st

    # ── 1. Hero section ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 0 2rem 0;">
          <div style="font-size:3.5rem;font-weight:800;
                     background:linear-gradient(135deg,#3B82F6,#9B72CF);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                     background-clip:text;line-height:1.2;margin-bottom:0.75rem;">
            BioLitAI-X
          </div>
          <div style="font-size:1.05rem;color:#9CA3AF;
                     max-width:560px;margin:0 auto;line-height:1.6;">
            From Literature to Discovery &nbsp;—&nbsp;
            AI-Powered Biomedical Intelligence
          </div>
          <div style="margin-top:0.75rem;font-size:0.78rem;color:#4B5563;
                     letter-spacing:0.06em;text-transform:uppercase;">
            Search · Analyse · Hypothesise · Discover
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 2. Search section ─────────────────────────────────────────────────────
    _, mid_col, _ = st.columns([1, 3, 1])

    with mid_col:
        query = st.text_input(
            "Research Query",
            value=session_state.get("current_query", ""),
            placeholder="Enter any biomedical research query…",
            label_visibility="collapsed",
            key="home_query_input",
        )

        col_slider, col_btn = st.columns([3, 1])
        with col_slider:
            max_results = st.slider(
                "Max Results",
                min_value=100,
                max_value=300,
                value=100,
                step=50,
                help=(
                    "Number of PubMed papers to retrieve. "
                    "100 is recommended for free-tier speed."
                ),
                key="home_max_results",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            run_clicked = st.button(
                "▶ Run Pipeline",
                use_container_width=True,
                key="home_run_btn",
            )

    # ── 3. Past query chips ───────────────────────────────────────────────────
    sessions = session_state.get("sessions", [])
    if sessions:
        st.markdown(
            '<div style="text-align:center;margin-top:0.5rem;margin-bottom:0.25rem;">'
            '<span style="font-size:0.72rem;color:#6B7280;text-transform:uppercase;'
            'letter-spacing:0.08em;">Recent queries</span></div>',
            unsafe_allow_html=True,
        )
        chip_cols = st.columns(min(len(sessions[-5:]), 5))
        for i, sess in enumerate(sessions[-5:][::-1]):
            q = sess.get("query_text", "") or sess.get("query", "")
            if not q:
                continue
            short = q[:28] + "…" if len(q) > 28 else q
            with chip_cols[i % len(chip_cols)]:
                if st.button(
                    short,
                    key=f"chip_q_{i}_{hash(q) & 0xFFFF}",
                    help=q,
                    use_container_width=True,
                ):
                    session_state["current_query"] = q
                    st.rerun()

    # ── Completion banner (persists after pipeline finishes) ──────────────────
    if (
        session_state.get("pipeline_complete")
        and session_state.get("pipeline_status") != "running"
        and not run_clicked
    ):
        _, _bcol, _ = st.columns([1, 3, 1])
        with _bcol:
            _n_p = (
                len(session_state["papers_df"])
                if session_state.get("papers_df") is not None
                else 0
            )
            st.markdown(
                f"""
                <div style="
                  margin-top:1.5rem;
                  padding:22px 26px;
                  background:linear-gradient(135deg,#0B1F3A 0%,#0D2750 100%);
                  border:1px solid #1E4080;
                  border-radius:14px;
                  text-align:center;
                ">
                  <div style="font-size:1.6rem;margin-bottom:10px;">🎉</div>
                  <div style="font-size:1.0rem;font-weight:700;color:#F9FAFB;
                               margin-bottom:8px;">
                    Analysis Ready!
                  </div>
                  <div style="font-size:0.83rem;color:#9CA3AF;line-height:1.7;
                               margin-bottom:16px;">
                    <b style="color:#60A5FA;">{_n_p:,} papers</b> have been processed and
                    indexed. Use the <b style="color:#93C5FD;">sidebar on the left</b>
                    to navigate to any view and explore your results.
                  </div>
                  <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;">
                    <span style="background:#1C2539;color:#60A5FA;
                                 border:1px solid #1E3A5F;border-radius:20px;
                                 padding:5px 15px;font-size:0.75rem;font-weight:600;">
                      Analysis
                    </span>
                    <span style="background:#1C2539;color:#60A5FA;
                                 border:1px solid #1E3A5F;border-radius:20px;
                                 padding:5px 15px;font-size:0.75rem;font-weight:600;">
                      Knowledge Graph
                    </span>
                    <span style="background:#1C2539;color:#60A5FA;
                                 border:1px solid #1E3A5F;border-radius:20px;
                                 padding:5px 15px;font-size:0.75rem;font-weight:600;">
                      Hypotheses
                    </span>
                    <span style="background:#1C2539;color:#60A5FA;
                                 border:1px solid #1E3A5F;border-radius:20px;
                                 padding:5px 15px;font-size:0.75rem;font-weight:600;">
                      Semantic Search
                    </span>
                    <span style="background:#1C2539;color:#60A5FA;
                                 border:1px solid #1E3A5F;border-radius:20px;
                                 padding:5px 15px;font-size:0.75rem;font-weight:600;">
                      Chat
                    </span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── 4. Pipeline execution ─────────────────────────────────────────────────
    if run_clicked:
        if not query or not query.strip():
            st.warning("Please enter a biomedical research query before running the pipeline.")
            return

        _run_pipeline(session_state, query.strip(), max_results)


def _run_pipeline(session_state, query: str, max_results: int):
    """
    Execute the full BioLitAI-X pipeline with live stepper UI.

    Steps
    -----
    0 – Fetching       : PubMedRetriever.fetch_with_progress
    1 – Parsing        : XMLParser.parse_batches
    2 – Cleaning       : DataCleaner.run_full_pipeline + DB storage
    3 – Processing     : NLPProcessor.process_corpus + EmbeddingEngine.embed_corpus
    4 – Embedding      : TopicModeler.fit_transform + NetworkBuilder (all graphs)
    5 – Building Graph : KnowledgeGraph.build_from_entities + GapDetector
    """
    import streamlit as st

    STEPS = [
        "Fetching",
        "Parsing",
        "Cleaning",
        "Processing",
        "Embedding",
        "Building Graph",
    ]

    from ui.components.loaders import show_pipeline_stepper

    stepper_ph = st.empty()
    status_ph = st.empty()
    error_ph = st.empty()

    def _update_stepper(step: int, papers=0, entities=0, rels=0, msg: str = ""):
        with stepper_ph.container():
            show_pipeline_stepper(
                steps=STEPS,
                current_step=step,
                paper_count=papers,
                entity_count=entities,
                rel_count=rels,
            )
        if msg:
            status_ph.markdown(
                f'<div style="text-align:center;font-size:0.82rem;color:#9CA3AF;">'
                f'{msg}</div>',
                unsafe_allow_html=True,
            )

    # ── Initialise DB session ─────────────────────────────────────────────────
    session_state["pipeline_status"] = "running"
    session_state["pipeline_complete"] = False
    session_state["current_query"] = query

    db_session_id = -1
    try:
        from database.db_manager import DatabaseManager
        db = DatabaseManager()
        db_session_id = db.save_query_session(query, max_results)
    except Exception as exc:
        error_ph.warning(f"Database initialisation warning: {exc}")
        db = None

    papers_df = None
    all_entities = []
    all_relationships = []
    embeddings = None
    embedder = None
    topic_results = None
    coauthor_graph = None
    keyword_graph = None
    topic_graph = None
    knowledge_graph = None
    gap_report = []

    # ── Step 0: Fetching ──────────────────────────────────────────────────────
    _update_stepper(0, msg=f"Searching PubMed for: {query!r}…")
    xml_batches = []
    try:
        from pipeline.retrieval import PubMedRetriever
        retriever = PubMedRetriever()

        fetched_count = [0]

        def _fetch_progress(done, total):
            fetched_count[0] = done
            _update_stepper(0, papers=done, msg=f"Fetching {done}/{total} records…")

        xml_batches = retriever.fetch_with_progress(
            query=query,
            max_results=max_results,
            progress_callback=_fetch_progress,
        )
        _update_stepper(
            0,
            papers=fetched_count[0],
            msg=f"Fetched {len(xml_batches)} XML batch(es).",
        )
    except EnvironmentError as exc:
        error_ph.error(
            f"Configuration error: {exc}\n\n"
            "Please add `ENTREZ_EMAIL=your_email@example.com` to your `.env` file."
        )
        session_state["pipeline_status"] = "idle"
        return
    except Exception as exc:
        error_ph.error(f"Step 0 (Fetching) failed: {exc}")
        session_state["pipeline_status"] = "idle"
        return

    if not xml_batches:
        error_ph.warning(
            "No results returned from PubMed for this query. "
            "Try broadening your search terms."
        )
        session_state["pipeline_status"] = "idle"
        return

    # ── Step 1: Parsing ───────────────────────────────────────────────────────
    _update_stepper(1, papers=len(xml_batches) * 100, msg="Parsing XML records…")
    raw_papers = []
    try:
        from pipeline.parser import XMLParser
        parser = XMLParser()
        raw_papers = parser.parse_batches(xml_batches, query_used=query)
        _update_stepper(1, papers=len(raw_papers), msg=f"Parsed {len(raw_papers)} papers.")
    except Exception as exc:
        error_ph.error(f"Step 1 (Parsing) failed: {exc}")
        session_state["pipeline_status"] = "idle"
        return

    # ── Step 2: Cleaning ──────────────────────────────────────────────────────
    _update_stepper(2, papers=len(raw_papers), msg="Cleaning and deduplicating records…")
    try:
        from pipeline.cleaner import DataCleaner
        cleaner = DataCleaner()
        papers_df = cleaner.run_full_pipeline(raw_papers)
        _update_stepper(2, papers=len(papers_df), msg=f"Cleaned: {len(papers_df)} unique papers.")

        # Persist to database
        if db is not None:
            for _, row in papers_df.iterrows():
                try:
                    db.insert_paper(row.to_dict())
                except Exception:
                    pass
            if db_session_id > 0:
                db.update_query_session(
                    db_session_id,
                    papers_fetched=len(papers_df),
                    pipeline_status="cleaning_done",
                )
    except Exception as exc:
        error_ph.error(f"Step 2 (Cleaning) failed: {exc}")
        # Attempt to continue with raw data as a fallback
        import pandas as pd
        papers_df = pd.DataFrame(raw_papers) if raw_papers else pd.DataFrame()

    if papers_df is None or papers_df.empty:
        error_ph.warning("No usable papers after cleaning. Pipeline halted.")
        session_state["pipeline_status"] = "idle"
        return

    # ── Step 3: Processing (NLP + Embeddings) ─────────────────────────────────
    _update_stepper(3, papers=len(papers_df), msg="Running NLP entity extraction…")

    # NLP is optional — gracefully skip if scispacy is not installed
    try:
        from pipeline.nlp_processor import NLPProcessor
        nlp = NLPProcessor()

        nlp_done = [0]

        def _nlp_progress(done, total):
            nlp_done[0] = done
            _update_stepper(
                3,
                papers=len(papers_df),
                entities=done * 3,  # rough estimate
                msg=f"NLP: {done}/{total} abstracts processed…",
            )

        all_entities, all_relationships = nlp.process_corpus(
            papers_df=papers_df,
            db_manager=db,
            progress_callback=_nlp_progress,
        )
        _update_stepper(
            3,
            papers=len(papers_df),
            entities=len(all_entities),
            rels=len(all_relationships),
            msg=f"NLP complete: {len(all_entities)} entities, {len(all_relationships)} relationships.",
        )
    except ImportError:
        status_ph.warning(
            "NLP unavailable — install scispacy to enable entity extraction. "
            "Continuing with structural analysis only."
        )
        all_entities = []
        all_relationships = []
    except Exception as exc:
        status_ph.warning(f"NLP processing encountered an error ({exc}). Continuing.")
        all_entities = []
        all_relationships = []

    # Embeddings
    _update_stepper(
        3,
        papers=len(papers_df),
        entities=len(all_entities),
        rels=len(all_relationships),
        msg="Computing semantic embeddings…",
    )
    try:
        from pipeline.embedder import EmbeddingEngine
        embedder = EmbeddingEngine()

        emb_done = [0]

        def _emb_progress(done, total):
            emb_done[0] = done
            _update_stepper(
                3,
                papers=len(papers_df),
                entities=len(all_entities),
                rels=len(all_relationships),
                msg=f"Embedding: {done}/{total} documents…",
            )

        embeddings = embedder.embed_corpus(
            papers_df=papers_df,
            query=query,
            progress_callback=_emb_progress,
        )
        session_state["embeddings_ready"] = True
        _update_stepper(
            3,
            papers=len(papers_df),
            entities=len(all_entities),
            rels=len(all_relationships),
            msg=f"Embeddings complete: {len(embeddings)} vectors.",
        )
    except ImportError as exc:
        status_ph.warning(
            f"Embedding engine unavailable ({exc}). "
            "Semantic search will be disabled."
        )
        embedder = None
        embeddings = None
    except Exception as exc:
        status_ph.warning(f"Embedding step failed ({exc}). Continuing.")
        embedder = None
        embeddings = None

    # ── Step 4: Topic Modelling + Bibliometric Networks ────────────────────────
    _update_stepper(
        4,
        papers=len(papers_df),
        entities=len(all_entities),
        rels=len(all_relationships),
        msg="Discovering topics and building networks…",
    )

    topic_results = None
    try:
        from pipeline.topic_modeler import TopicModeler
        import numpy as np

        modeler = TopicModeler()
        abstracts = papers_df["abstract"].fillna("").tolist()
        emb_input = embeddings if (embeddings is not None and len(embeddings) > 0) else None
        topics, probs = modeler.fit_transform(abstracts, embeddings=emb_input)
        topic_info = modeler.get_topic_summary()
        topics_over_time = modeler.get_topic_over_time(papers_df)
        topic_results = {
            "topics": topics,
            "probs": probs,
            "topic_info": topic_info,
            "topics_over_time": topics_over_time,
            "modeler": modeler,
        }
        _update_stepper(
            4,
            papers=len(papers_df),
            entities=len(all_entities),
            rels=len(all_relationships),
            msg="Topics discovered. Building bibliometric networks…",
        )
    except ImportError:
        status_ph.warning(
            "BERTopic unavailable — install bertopic, umap-learn, hdbscan "
            "for topic modelling."
        )
    except Exception as exc:
        status_ph.warning(f"Topic modelling failed ({exc}). Continuing.")

    # Bibliometric networks
    try:
        from pipeline.network_builder import NetworkBuilder
        builder = NetworkBuilder()
        coauthor_graph = builder.build_coauthorship_network(papers_df)
        keyword_graph = builder.build_keyword_cooccurrence_network(papers_df)
        if topic_results and "topics" in topic_results:
            # Build paper_assignments list for topic network
            paper_assignments = []
            for i, pmid in enumerate(papers_df["pmid"].astype(str).tolist()):
                if i < len(topic_results["topics"]):
                    paper_assignments.append({
                        "pmid": pmid,
                        "topic_id": int(topic_results["topics"][i]),
                    })
            topic_graph = builder.build_topic_network({
                "topic_summary": topic_results.get("topic_info", []),
                "paper_assignments": paper_assignments,
            })
        else:
            topic_graph = builder.build_topic_network({
                "topic_summary": [],
                "paper_assignments": [],
            })
        _update_stepper(
            4,
            papers=len(papers_df),
            entities=len(all_entities),
            rels=len(all_relationships),
            msg="Bibliometric networks built.",
        )
    except Exception as exc:
        status_ph.warning(f"Network building failed ({exc}). Continuing.")

    # ── Step 5: Knowledge Graph + Gap Detection ────────────────────────────────
    _update_stepper(
        5,
        papers=len(papers_df),
        entities=len(all_entities),
        rels=len(all_relationships),
        msg="Building knowledge graph…",
    )
    try:
        from pipeline.knowledge_graph import KnowledgeGraph
        kg_builder = KnowledgeGraph()
        knowledge_graph = kg_builder.build_from_entities(
            entities_df=all_entities,
            relationships_df=all_relationships,
        )
        # Add semantic similarity edges if embeddings are available
        if embeddings is not None and len(embeddings) > 0 and knowledge_graph.number_of_nodes() > 0:
            try:
                pmid_list = papers_df["pmid"].astype(str).tolist()
                kg_builder.add_semantic_edges(
                    embeddings=embeddings,
                    pmid_list=pmid_list,
                )
            except Exception:
                pass
        _update_stepper(
            5,
            papers=len(papers_df),
            entities=knowledge_graph.number_of_nodes() if knowledge_graph else 0,
            rels=knowledge_graph.number_of_edges() if knowledge_graph else 0,
            msg="Knowledge graph built. Detecting research gaps…",
        )
    except Exception as exc:
        status_ph.warning(f"Knowledge graph construction failed ({exc}). Continuing.")

    try:
        from pipeline.gap_detector import GapDetector
        detector = GapDetector()
        if knowledge_graph is not None and knowledge_graph.number_of_nodes() > 0:
            gap_report = detector.compile_gap_report(
                knowledge_graph=knowledge_graph,
                papers_df=papers_df,
            )
        else:
            gap_report = []
        _update_stepper(
            6,
            papers=len(papers_df),
            entities=knowledge_graph.number_of_nodes() if knowledge_graph else 0,
            rels=knowledge_graph.number_of_edges() if knowledge_graph else 0,
            msg=f"Pipeline complete! {len(gap_report)} research gaps identified.",
        )
    except Exception as exc:
        status_ph.warning(f"Gap detection failed ({exc}). Continuing.")
        gap_report = []

    # ── Update DB session to complete ──────────────────────────────────────────
    if db is not None and db_session_id > 0:
        try:
            db.update_query_session(db_session_id, pipeline_status="complete")
        except Exception:
            pass

    # ── Store all results in session_state ────────────────────────────────────
    session_state["pipeline_complete"] = True
    session_state["pipeline_status"] = "complete"
    session_state["papers_df"] = papers_df
    session_state["knowledge_graph"] = knowledge_graph
    session_state["coauthor_graph"] = coauthor_graph
    session_state["keyword_graph"] = keyword_graph
    session_state["topic_graph"] = topic_graph
    session_state["topic_model_results"] = topic_results
    session_state["gap_report"] = gap_report
    session_state["embedder"] = embedder
    session_state["active_session_id"] = db_session_id

    # Add this session to the sessions list for the sidebar
    existing_sessions = session_state.get("sessions", [])
    existing_sessions.append({
        "id": db_session_id,
        "query_text": query,
        "papers_fetched": len(papers_df),
    })
    session_state["sessions"] = existing_sessions

    # ── Final status update ───────────────────────────────────────────────────
    status_ph.success(
        f"Pipeline complete! Retrieved and analysed **{len(papers_df):,} papers** "
        f"for query: *{query}*. "
        f"Navigate to **Analysis**, **Knowledge Graph**, **Hypotheses**, "
        f"**Semantic Search**, or **Chat** to explore your results."
    )
    st.rerun()
