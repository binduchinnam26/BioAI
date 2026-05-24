"""
Hypotheses page — AI-generated hypotheses from detected research gaps.
Uses Google Gemini (free-tier by default) via HypothesisGenerator.
Streamlit is imported inside each function.
"""


def render_hypotheses(session_state):
    """
    Render the Hypotheses page.

    Layout
    ------
    1. Header with gap/hypothesis counts
    2. Generate button (runs HypothesisGenerator on top-N gaps)
    3. Hypothesis cards (cached from session_state and DB)
    4. Gap browser (all detected gaps)
    """
    import streamlit as st

    papers_df = session_state.get("papers_df")
    gap_report = session_state.get("gap_report", [])

    if papers_df is None or (hasattr(papers_df, "empty") and papers_df.empty):
        _no_data_placeholder()
        return

    # ── Header ────────────────────────────────────────────────────────────────
    hypotheses = session_state.get("hypotheses", [])

    st.markdown(
        '<h2 style="font-size:1.4rem;font-weight:700;color:#F9FAFB;'
        'margin-bottom:0.25rem;">Hypotheses</h2>'
        f'<p style="font-size:0.875rem;color:#9CA3AF;margin-bottom:1rem;">'
        f'{len(gap_report)} research gaps detected · '
        f'{len(hypotheses)} hypotheses generated</p>',
        unsafe_allow_html=True,
    )

    # ── Generate panel ────────────────────────────────────────────────────────
    if not gap_report:
        st.info(
            "No research gaps detected yet. "
            "Run the pipeline on the **Home** page with entity extraction enabled."
        )
        return

    _render_generate_panel(session_state, papers_df, gap_report)

    # ── Tabs: Hypotheses | Gaps ───────────────────────────────────────────────
    tab_hyp, tab_gap = st.tabs([
        f"💡 Hypotheses ({len(hypotheses)})",
        f"🔍 Gaps ({len(gap_report)})",
    ])

    with tab_hyp:
        _render_hypothesis_list(hypotheses)

    with tab_gap:
        _render_gap_list(gap_report)


def _no_data_placeholder():
    import streamlit as st

    st.markdown(
        """
        <div style="text-align:center;padding:80px 20px;">
          <div style="font-size:3rem;margin-bottom:12px;">💡</div>
          <h3 style="color:#F9FAFB;margin-bottom:8px;">No Data Yet</h3>
          <p style="color:#9CA3AF;max-width:400px;margin:0 auto;">
            Run a query on the <b>Home</b> page first to detect research gaps,
            then return here to generate AI hypotheses.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_generate_panel(session_state, papers_df, gap_report):
    import streamlit as st
    from config import HYPOTHESIS_TOP_GAPS

    with st.expander("⚙️ Generation Settings", expanded=True):
        col_n, col_btn = st.columns([3, 1])
        with col_n:
            top_n = st.slider(
                "Number of gaps to process",
                min_value=1,
                max_value=min(len(gap_report), 10),
                value=min(HYPOTHESIS_TOP_GAPS, len(gap_report)),
                help=(
                    "Free-tier Gemini Flash supports ~15 RPM / 1500 RPD. "
                    "Keep ≤5 for safety on the free tier."
                ),
                key="hyp_top_n",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            generate_clicked = st.button(
                "✨ Generate",
                use_container_width=True,
                key="hyp_generate_btn",
                type="primary",
            )

    if generate_clicked:
        _run_generation(session_state, papers_df, gap_report, top_n)


def _run_generation(session_state, papers_df, gap_report, top_n: int):
    import streamlit as st

    progress_ph = st.empty()
    status_ph = st.empty()

    try:
        from pipeline.hypothesis_generator import HypothesisGenerator
        generator = HypothesisGenerator()
        generator.setup()  # raises EnvironmentError if GEMINI_API_KEY missing

        from database.db_manager import DatabaseManager
        db = DatabaseManager()
        embedder = session_state.get("embedder")

        hypotheses = []
        progress_ph.progress(0.0)

        def _progress_cb(done: int, total: int, status: str = ""):
            frac = done / total if total > 0 else 0.0
            progress_ph.progress(frac)
            status_ph.markdown(
                f'<div style="font-size:0.82rem;color:#9CA3AF;text-align:center;">'
                f'Generating hypothesis {done}/{total}…</div>',
                unsafe_allow_html=True,
            )

        query_used = session_state.get("current_query", "")
        hypotheses = generator.generate_batch_hypotheses(
            gap_report=gap_report,
            papers_df=papers_df,
            db_manager=db,
            embedder=embedder,
            query_used=query_used,
            top_n=top_n,
            progress_callback=_progress_cb,
        )

        progress_ph.empty()
        status_ph.empty()

        if hypotheses:
            session_state["hypotheses"] = hypotheses
            st.success(f"Generated {len(hypotheses)} hypotheses.")
            st.rerun()
        else:
            # Count how many gaps actually had two concepts (eligible for generation)
            eligible = sum(
                1 for g in gap_report
                if g.get("concept_a") and g.get("concept_b")
                and g["concept_a"] != g.get("concept_b")
            )
            if eligible == 0:
                st.warning(
                    "All detected gaps are single-concept (temporal) gaps — "
                    "they describe one entity with no direct pair to compare. "
                    "The Gemini model needs two concepts to generate a hypothesis. "
                    "Try running the pipeline on a larger corpus (increase Max Results) "
                    "to detect more structural or cross-domain gaps."
                )
            else:
                st.warning(
                    f"No hypotheses were generated. "
                    f"({eligible} eligible gap pairs found, but the Gemini API returned no results.) "
                    "Check that your GEMINI_API_KEY is valid and not quota-exhausted."
                )

    except EnvironmentError as exc:
        progress_ph.empty()
        status_ph.empty()
        st.error(
            f"Gemini API not configured: {exc}\n\n"
            "Add `GEMINI_API_KEY=your_key` to your `.env` file. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
    except Exception as exc:
        progress_ph.empty()
        status_ph.empty()
        # Check for daily quota exhaustion (DailyQuotaError bubbles up as Exception)
        if "daily quota" in str(exc).lower() or "midnight pacific" in str(exc).lower():
            st.markdown(
                """
                <div style="background:#1C1A14;border:1px solid #92400E;border-radius:10px;
                            padding:20px 24px;margin-top:8px;">
                  <div style="font-size:1.1rem;font-weight:700;color:#F59E0B;margin-bottom:8px;">
                    ⏳ Daily API Quota Exhausted
                  </div>
                  <p style="color:#D1D5DB;font-size:0.875rem;line-height:1.6;margin-bottom:12px;">
                    Both Gemini API keys have used up their free-tier <b>daily request limit (RPD)</b>.
                    This is not an error with your keys or the app — it's a Google free-tier limit.
                  </p>
                  <div style="background:#111827;border-radius:6px;padding:12px 16px;
                              margin-bottom:12px;font-size:0.82rem;color:#9CA3AF;line-height:1.8;">
                    🕛 &nbsp;<b style="color:#F9FAFB;">Quota resets at midnight Pacific Time (00:00 PT)</b><br>
                    🔑 &nbsp;Add keys from more Google accounts to multiply your daily budget<br>
                    💾 &nbsp;Previously generated hypotheses are saved and still accessible below
                  </div>
                  <p style="color:#6B7280;font-size:0.78rem;margin:0;">
                    Get free keys at
                    <a href="https://aistudio.google.com/app/apikey" target="_blank"
                       style="color:#3B82F6;">aistudio.google.com/app/apikey</a>
                    — each Google account gives a separate daily quota.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error(f"Hypothesis generation failed: {exc}")


def _render_hypothesis_list(hypotheses: list):
    import streamlit as st

    if not hypotheses:
        st.markdown(
            """
            <div style="text-align:center;padding:40px 20px;color:#9CA3AF;">
              <div style="font-size:2rem;margin-bottom:8px;">🤖</div>
              Click <b>Generate</b> above to create AI-powered hypotheses
              from the detected research gaps.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Sort options
    col_sort, col_filter = st.columns([2, 2])
    with col_sort:
        sort_by = st.selectbox(
            "Sort by",
            ["Novelty Score (↓)", "Novelty Score (↑)", "Order Generated"],
            key="hyp_sort",
            label_visibility="collapsed",
        )
    with col_filter:
        filter_term = st.text_input(
            "Filter",
            placeholder="Filter hypotheses…",
            key="hyp_filter",
            label_visibility="collapsed",
        )

    # Sort
    sorted_hyps = list(hypotheses)
    if sort_by == "Novelty Score (↓)":
        sorted_hyps.sort(key=lambda h: float(h.get("novelty_score", h.get("confidence_score", 0))), reverse=True)
    elif sort_by == "Novelty Score (↑)":
        sorted_hyps.sort(key=lambda h: float(h.get("novelty_score", h.get("confidence_score", 0))))

    # Filter
    if filter_term:
        ft = filter_term.lower()
        sorted_hyps = [
            h for h in sorted_hyps
            if ft in str(h).lower()
        ]

    if not sorted_hyps:
        st.info("No hypotheses match the current filter.")
        return

    from ui.components.cards import render_hypothesis_card
    for i, hyp in enumerate(sorted_hyps):
        render_hypothesis_card(hyp=hyp, index=i)


def _render_gap_list(gap_report: list):
    import streamlit as st
    from ui.components.cards import render_gap_card

    if not gap_report:
        st.info("No gaps to display.")
        return

    type_colors = {
        "structural":   "#3B82F6",
        "cross_domain": "#8B5CF6",
        "temporal":     "#F59E0B",
    }

    # Filter controls
    col_type, col_search = st.columns([2, 2])
    with col_type:
        gap_types_available = sorted(set(g.get("type", "structural") for g in gap_report))
        type_filter = st.multiselect(
            "Gap type",
            gap_types_available,
            default=gap_types_available,
            key="gap_type_filter",
            label_visibility="collapsed",
        )
    with col_search:
        gap_search = st.text_input(
            "Search gaps",
            placeholder="Search entity names…",
            key="gap_search",
            label_visibility="collapsed",
        )

    filtered = [
        g for g in gap_report
        if g.get("type", "structural") in type_filter
        and (
            not gap_search
            or gap_search.lower() in (g.get("concept_a", "") + " " + g.get("concept_b", "")).lower()
        )
    ]

    sorted_gaps = sorted(filtered, key=lambda g: g.get("score", 0), reverse=True)

    st.markdown(
        f'<p style="font-size:0.78rem;color:#6B7280;margin-bottom:10px;">'
        f'Showing {len(sorted_gaps)} of {len(gap_report)} gaps</p>',
        unsafe_allow_html=True,
    )

    for i, gap in enumerate(sorted_gaps[:50]):
        render_gap_card(gap=gap, index=i)

    if len(sorted_gaps) > 50:
        st.info(f"Showing top 50 gaps. {len(sorted_gaps) - 50} more gaps not shown.")
