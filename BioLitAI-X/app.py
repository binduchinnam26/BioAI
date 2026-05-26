"""
BioLitAI-X — main Streamlit entry point.

Run with:
    streamlit run app.py
"""
import sys
from pathlib import Path

# ── Load environment variables from .env file ──────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# Ensure the project root is on sys.path regardless of launch CWD.
_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioLitAI-X",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": "BioLitAI-X — Biomedical Literature Intelligence Platform",
    },
)

# ── Inject global CSS theme ────────────────────────────────────────────────
_CSS_PATH = _ROOT / "ui" / "styles" / "theme.css"
if _CSS_PATH.exists():
    with open(_CSS_PATH, "r", encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ── Initialise session state defaults ──────────────────────────────────────
_DEFAULTS = {
    "current_page": "Home",
    "pipeline_status": "idle",
    "pipeline_complete": False,
    "current_query": "",
    "papers_df": None,
    "knowledge_graph": None,
    "coauthor_graph": None,
    "keyword_graph": None,
    "topic_graph": None,
    "topic_model_results": None,
    "gap_report": [],
    "hypotheses": [],
    "embedder": None,
    "chat_history": [],
    "sessions": [],
    "active_session_id": -1,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Shared database manager ────────────────────────────────────────────────
@st.cache_resource
def _get_db_manager():
    from database.db_manager import DatabaseManager
    return DatabaseManager()

try:
    _db = _get_db_manager()
except Exception as _exc:
    st.error(f"Database initialisation failed: {_exc}")
    st.stop()

# ── Load past sessions from DB into session state ──────────────────────────
if not st.session_state.get("_sessions_loaded"):
    try:
        _rows = _db.get_all_sessions()[:10]
        if _rows:
            st.session_state["sessions"] = [
                {"id": r.get("id", -1), "query_text": r.get("query_text", "")}
                for r in _rows
                if r.get("query_text")
            ]
    except Exception:
        pass
    st.session_state["_sessions_loaded"] = True

# ── Sidebar navigation ─────────────────────────────────────────────────────
# If the Run Pipeline button was just clicked with a valid query, mark the
# pipeline as "running" NOW — before render_sidebar() reads the status —
# so the sidebar badge reflects the correct state immediately.
if (st.session_state.get("home_run_btn", False)
        and st.session_state.get("home_query_input", "").strip()):
    st.session_state["pipeline_status"] = "running"

from ui.components.sidebar import render_sidebar

selected_page = render_sidebar(st.session_state)

# ── Route to page ─────────────────────────────────────────────────────────
if selected_page == "Home":
    from ui.pages.home import render_home
    render_home(st.session_state)

elif selected_page == "Analysis":
    from ui.pages.analysis import render_analysis
    render_analysis(st.session_state)

elif selected_page == "Knowledge Graph":
    from ui.pages.knowledge_graph import render_knowledge_graph_page
    render_knowledge_graph_page(st.session_state)

elif selected_page == "Hypotheses":
    from ui.pages.hypotheses import render_hypotheses
    render_hypotheses(st.session_state)

elif selected_page == "Semantic Search":
    from ui.pages.semantic_search import render_semantic_search
    render_semantic_search(st.session_state)

elif selected_page == "Chat":
    from ui.pages.chat import render_chat
    render_chat(st.session_state)

else:
    from ui.pages.home import render_home
    render_home(st.session_state)
