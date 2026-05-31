"""
Literature Chat page — Gemini-powered Q&A grounded in the corpus.
Streamlit is imported inside each function.
"""


def render_chat(session_state):
    """
    Render the Literature Chat page.

    Layout
    ------
    1. Header and corpus indicator
    2. Chat history display (user bubbles right, assistant bubbles left)
    3. Input box + send button
    4. Clear conversation button
    """
    import streamlit as st

    papers_df = session_state.get("papers_df")

    if papers_df is None or (hasattr(papers_df, "empty") and papers_df.empty):
        _no_data_placeholder()
        return

    # ── Header ────────────────────────────────────────────────────────────────
    query = session_state.get("current_query", "")
    st.markdown(
        '<h2 style="font-size:1.4rem;font-weight:700;color:#F9FAFB;'
        'margin-bottom:0.25rem;">Literature Chat</h2>'
        f'<p style="font-size:0.875rem;color:#9CA3AF;margin-bottom:1.25rem;">'
        f'Ask questions about: <b style="color:#3B82F6;">{query}</b> · '
        f'{len(papers_df):,} papers · Answers are grounded in your corpus.</p>',
        unsafe_allow_html=True,
    )

    # ── Check Gemini availability (cached — setup() makes a real API call, so
    #    we must NOT call it on every Streamlit rerender) ──────────────────────
    if "chat_generator" not in session_state:
        try:
            from pipeline.hypothesis_generator import HypothesisGenerator
            _gen = HypothesisGenerator()
            _gen.setup()
            session_state["chat_generator"] = _gen
        except EnvironmentError as exc:
            st.error(
                f"Gemini API not configured: {exc}\n\n"
                "Add `GEMINI_API_KEY=your_key` to your `.env` file. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
            return
        except Exception as exc:
            st.error(f"Chat setup failed: {exc}")
            return
    generator = session_state["chat_generator"]

    embedder = session_state.get("embedder")

    # ── Chat history ──────────────────────────────────────────────────────────
    if "chat_history" not in session_state:
        session_state["chat_history"] = []

    chat_history = session_state["chat_history"]

    _render_chat_history(chat_history)

    # ── Input row ─────────────────────────────────────────────────────────────
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Message",
            placeholder="Ask a question about the literature…",
            label_visibility="collapsed",
            key="chat_input",
        )
    with col_send:
        send_clicked = st.button(
            "Send →",
            type="primary",
            use_container_width=True,
            key="chat_send",
        )

    # Clear button
    if st.button("🗑 Clear conversation", key="chat_clear"):
        session_state["chat_history"] = []
        st.rerun()

    # ── Handle starter question injection ────────────────────────────────────
    if st.session_state.get("chat_starter_inject"):
        injected = st.session_state.pop("chat_starter_inject")
        _send_message(
            session_state=session_state,
            generator=generator,
            papers_df=papers_df,
            embedder=embedder,
            user_message=injected,
        )
        return

    # ── Send message ──────────────────────────────────────────────────────────
    if send_clicked and user_input.strip():
        _send_message(
            session_state=session_state,
            generator=generator,
            papers_df=papers_df,
            embedder=embedder,
            user_message=user_input.strip(),
        )


def _no_data_placeholder():
    import streamlit as st

    st.markdown(
        """
        <div style="text-align:center;padding:80px 20px;">
          <div style="font-size:3rem;margin-bottom:12px;">💬</div>
          <h3 style="color:#F9FAFB;margin-bottom:8px;">No Corpus Loaded</h3>
          <p style="color:#9CA3AF;max-width:400px;margin:0 auto;">
            Run a query on the <b>Home</b> page first, then return here
            to ask questions about the literature.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_chat_history(chat_history: list):
    import streamlit as st

    if not chat_history:
        st.markdown(
            """
            <div style="background:#111827;border:1px dashed #1F2937;
                        border-radius:12px;padding:28px 24px;text-align:center;
                        margin-bottom:16px;">
              <div style="font-size:1.5rem;margin-bottom:8px;">🤖</div>
              <p style="font-size:0.875rem;color:#9CA3AF;margin:0 0 16px 0;line-height:1.6;">
                Ask me anything about the papers in your corpus.<br>
                My answers are grounded in the retrieved literature —
                I will not invent facts outside the corpus.
              </p>
              <p style="font-size:0.72rem;color:#6B7280;text-transform:uppercase;
                        letter-spacing:0.08em;margin-bottom:10px;">
                Suggested questions
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Starter question buttons
        starter_questions = [
            "What are the main findings across these papers?",
            "What methodologies are most commonly used in this corpus?",
            "What are the key research gaps identified in the literature?",
            "Which papers provide the strongest evidence for the primary outcomes?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(starter_questions):
            with cols[i % 2]:
                if st.button(q, key=f"starter_q_{i}", use_container_width=True):
                    # Inject as user message — will be picked up by the send flow
                    st.session_state["chat_starter_inject"] = q
                    st.rerun()
        return

    for turn in chat_history:
        role = turn.get("role", "user")
        content = turn.get("content", "")

        if role == "user":
            try:
                from ui.components.cards import render_chat_message
                render_chat_message(role="user", content=content)
            except Exception:
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:flex-end;margin-bottom:8px;">
                      <div style="background:#3B82F6;
                                  border-radius:12px 12px 2px 12px;
                                  padding:10px 14px;max-width:80%;
                                  font-size:0.875rem;color:#FFFFFF;line-height:1.55;">
                        {content}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            # Format assistant response with basic markdown → HTML conversion
            formatted = _format_assistant_response(content)
            source_pmids = turn.get("source_pmids", [])
            source_titles = turn.get("source_titles", [])
            sources = [
                {"pmid": p, "title": t}
                for p, t in zip(source_pmids, source_titles)
            ] if source_pmids else []

            try:
                from ui.components.cards import render_chat_message
                render_chat_message(role="assistant", content=content, sources=sources)
            except Exception:
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:flex-start;margin-bottom:8px;">
                      <div style="background:#111827;border:1px solid #1F2937;
                                  border-radius:12px 12px 12px 2px;
                                  padding:10px 14px;max-width:85%;
                                  font-size:0.875rem;color:#D1D5DB;line-height:1.65;">
                        {formatted}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Scroll anchor
    st.markdown(
        '<div id="chat-bottom"></div>'
        '<script>document.getElementById("chat-bottom").scrollIntoView({behavior:"smooth"});</script>',
        unsafe_allow_html=True,
    )


def _format_assistant_response(text: str) -> str:
    """Convert basic markdown (bold, italic, code) to HTML for inline rendering."""
    import re

    # Escape HTML entities first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Bold **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Italic *text*
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # Inline code `text`
    text = re.sub(
        r'`(.+?)`',
        r'<code style="background:#1C2539;padding:1px 5px;border-radius:3px;'
        r'font-size:0.82em;color:#10B981;">\1</code>',
        text,
    )
    # Newlines → <br>
    text = text.replace("\n", "<br>")

    return text


def _send_message(session_state, generator, papers_df, embedder, user_message: str):
    import streamlit as st

    chat_history = session_state.get("chat_history", [])

    # Add user turn immediately
    chat_history.append({"role": "user", "content": user_message})
    session_state["chat_history"] = chat_history

    # Stream response
    with st.spinner("Thinking…"):
        try:
            result = generator.chat_about_literature(
                user_message=user_message,
                conversation_history=chat_history[:-1],  # history without current turn
                papers_df=papers_df,
                embedder=embedder,
            )
            if isinstance(result, dict):
                response_text = result.get("response_text", "")
                source_pmids = result.get("source_pmids", [])
                source_titles = result.get("source_titles", [])
            else:
                response_text = str(result)
                source_pmids = []
                source_titles = []
        except Exception as exc:
            response_text = f"I encountered an error: {exc}. Please try again."
            source_pmids = []
            source_titles = []

    chat_history.append({
        "role": "assistant",
        "content": response_text,
        "source_pmids": source_pmids,
        "source_titles": source_titles,
    })
    session_state["chat_history"] = chat_history

    st.rerun()
