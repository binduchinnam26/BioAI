"""
Animated loading components for BioLitAI-X.
All functions use st.markdown with inline HTML/CSS for rendering.
Streamlit is imported inside each function per project conventions.
"""


def show_skeleton_cards(n: int = 4, height: str = "120px"):
    """
    Render n animated shimmer skeleton card placeholders.
    Injects shimmer CSS once, then renders n shimmer divs.
    """
    import streamlit as st

    shimmer_css = """
    <style>
    @keyframes _shimmer {
      0%   { background-position: -1200px 0; }
      100% { background-position:  1200px 0; }
    }
    .sk-card {
      background: linear-gradient(90deg, #1C2539 25%, #2A3550 50%, #1C2539 75%);
      background-size: 2400px 100%;
      animation: _shimmer 1.8s infinite linear;
      border-radius: 8px;
      margin-bottom: 12px;
    }
    </style>
    """

    cards_html = shimmer_css
    cols_per_row = min(n, 4)
    col_width = 100 / cols_per_row
    cards_html += f'<div style="display:flex;gap:12px;flex-wrap:wrap;">'
    for _ in range(n):
        cards_html += (
            f'<div class="sk-card" '
            f'style="width:calc({col_width}% - 12px);height:{height};"></div>'
        )
    cards_html += "</div>"

    st.markdown(cards_html, unsafe_allow_html=True)


def show_skeleton_graph(height: str = "750px"):
    """
    Render a single large shimmer block representing a graph canvas placeholder.
    """
    import streamlit as st

    html = f"""
    <style>
    @keyframes _shimmer_g {{
      0%   {{ background-position: -1200px 0; }}
      100% {{ background-position:  1200px 0; }}
    }}
    .sk-graph {{
      background: linear-gradient(90deg, #111827 25%, #1C2539 50%, #111827 75%);
      background-size: 2400px 100%;
      animation: _shimmer_g 2s infinite linear;
      border-radius: 10px;
      border: 1px solid #1F2937;
      width: 100%;
      height: {height};
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .sk-graph-inner {{
      color: #374151;
      font-size: 1rem;
      font-family: 'Open Sans', Arial, sans-serif;
      letter-spacing: 0.05em;
      user-select: none;
    }}
    </style>
    <div class="sk-graph">
      <div class="sk-graph-inner">Loading graph data…</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def show_skeleton_results(n: int = 5):
    """
    Render n skeleton result-card outlines suitable for a search results list.
    Each card has a wide title bar, two narrow lines for authors/journal,
    and a body block for the abstract excerpt.
    """
    import streamlit as st

    html = """
    <style>
    @keyframes _shimmer_r {
      0%   { background-position: -1200px 0; }
      100% { background-position:  1200px 0; }
    }
    .sk-result-card {
      background: #111827;
      border: 1px solid #1F2937;
      border-radius: 8px;
      padding: 16px 20px;
      margin-bottom: 12px;
    }
    .sk-line {
      background: linear-gradient(90deg, #1C2539 25%, #2A3550 50%, #1C2539 75%);
      background-size: 2400px 100%;
      animation: _shimmer_r 1.8s infinite linear;
      border-radius: 4px;
      margin-bottom: 10px;
    }
    </style>
    """

    for _ in range(n):
        html += """
        <div class="sk-result-card">
          <div class="sk-line" style="width:75%;height:18px;"></div>
          <div class="sk-line" style="width:45%;height:12px;margin-bottom:14px;"></div>
          <div class="sk-line" style="width:100%;height:10px;"></div>
          <div class="sk-line" style="width:95%;height:10px;"></div>
          <div class="sk-line" style="width:80%;height:10px;margin-bottom:0;"></div>
        </div>
        """

    st.markdown(html, unsafe_allow_html=True)


def show_skeleton_hypothesis_cards(n: int = 4):
    """
    Render a 2x2 grid of skeleton hypothesis card placeholders.
    """
    import streamlit as st

    html = """
    <style>
    @keyframes _shimmer_h {
      0%   { background-position: -1200px 0; }
      100% { background-position:  1200px 0; }
    }
    .sk-hyp-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }
    .sk-hyp-card {
      background: #111827;
      border: 1px solid #1F2937;
      border-radius: 8px;
      padding: 20px;
      min-height: 180px;
    }
    .sk-hyp-line {
      background: linear-gradient(90deg, #1C2539 25%, #2A3550 50%, #1C2539 75%);
      background-size: 2400px 100%;
      animation: _shimmer_h 1.8s infinite linear;
      border-radius: 4px;
      margin-bottom: 12px;
    }
    </style>
    <div class="sk-hyp-grid">
    """

    for _ in range(n):
        html += """
        <div class="sk-hyp-card">
          <div style="display:flex;justify-content:space-between;margin-bottom:14px;">
            <div class="sk-hyp-line" style="width:60%;height:16px;margin:0;"></div>
            <div class="sk-hyp-line" style="width:18%;height:16px;margin:0;border-radius:12px;"></div>
          </div>
          <div class="sk-hyp-line" style="width:100%;height:10px;"></div>
          <div class="sk-hyp-line" style="width:95%;height:10px;"></div>
          <div class="sk-hyp-line" style="width:85%;height:10px;"></div>
          <div class="sk-hyp-line" style="width:40%;height:10px;margin-top:4px;"></div>
        </div>
        """

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def show_pipeline_stepper(
    steps,
    current_step: int,
    paper_count: int = 0,
    entity_count: int = 0,
    rel_count: int = 0,
):
    """
    Render a 6-step horizontal pipeline stepper with live counter bar.

    Parameters
    ----------
    steps : list[str]
        Step labels, e.g. ['Fetching','Parsing','Cleaning',
                           'Processing','Embedding','Building Graph']
    current_step : int
        Index of the active step (0-5).  -1 = none started.  6 = all complete.
    paper_count : int
        Live paper count shown in the counter bar.
    entity_count : int
        Live entity count shown in the counter bar.
    rel_count : int
        Live relationship count shown in the counter bar.
    """
    import streamlit as st

    n = len(steps)

    def _step_circle(idx: int) -> str:
        if current_step == -1:
            # Nothing started
            state = "waiting"
        elif idx < current_step:
            state = "complete"
        elif idx == current_step:
            state = "active"
        else:
            state = "waiting"

        if state == "complete":
            bg = "#10B981"
            border = "#059669"
            icon = "✓"
            pulse = ""
        elif state == "active":
            bg = "#3B82F6"
            border = "#2563EB"
            icon = (
                '<svg width="18" height="18" viewBox="0 0 24 24" '
                'style="animation:spin 0.9s linear infinite;display:block;">'
                '<circle cx="12" cy="12" r="10" stroke="#FFFFFF" '
                'stroke-width="3" fill="none" stroke-dasharray="30 10"/>'
                "</svg>"
            )
            pulse = "animation:pulse_step 1.4s infinite;"
        else:
            bg = "#1C2539"
            border = "#374151"
            icon = str(idx + 1)
            pulse = ""

        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'flex:1;min-width:0;">'
            # Connector line left
            + (
                f'<div style="position:relative;display:flex;align-items:center;'
                f'width:100%;justify-content:center;">'
                if True
                else ""
            )
            + f'<div style="width:44px;height:44px;border-radius:50%;'
            f'background:{bg};border:2px solid {border};'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-weight:700;font-size:0.9rem;color:#FFFFFF;{pulse}'
            f'flex-shrink:0;">{icon}</div>'
            + "</div>"
            + f'<div style="margin-top:8px;font-size:0.72rem;color:#9CA3AF;'
            f'text-align:center;white-space:nowrap;overflow:hidden;'
            f'text-overflow:ellipsis;max-width:90px;">{steps[idx]}</div>'
            + "</div>"
        )

    # Build connector between steps
    def _connector(idx: int) -> str:
        if idx < current_step:
            color = "#10B981"
        elif idx == current_step - 1:
            color = "#3B82F6"
        else:
            color = "#374151"
        return (
            f'<div style="flex:1;height:2px;background:{color};'
            f'margin-top:-22px;z-index:0;"></div>'
        )

    stepper_html = """
    <style>
    @keyframes spin {
      from { transform: rotate(0deg); }
      to   { transform: rotate(360deg); }
    }
    @keyframes pulse_step {
      0%   { box-shadow: 0 0 0 0 rgba(59,130,246,0.5); }
      70%  { box-shadow: 0 0 0 10px rgba(59,130,246,0); }
      100% { box-shadow: 0 0 0 0 rgba(59,130,246,0); }
    }
    </style>
    <div style="
      background:#111827;
      border:1px solid #1F2937;
      border-radius:12px;
      padding:24px 28px 20px 28px;
      margin-bottom:20px;
    ">
      <div style="display:flex;align-items:flex-start;position:relative;">
    """

    for i in range(n):
        stepper_html += _step_circle(i)
        if i < n - 1:
            stepper_html += _connector(i)

    stepper_html += "</div>"

    # Live counter bar
    stepper_html += f"""
      <div style="
        display:flex;
        gap:32px;
        margin-top:22px;
        padding-top:16px;
        border-top:1px solid #1F2937;
      ">
        <div style="display:flex;flex-direction:column;align-items:center;">
          <span style="font-size:1.4rem;font-weight:700;color:#3B82F6;">
            {paper_count:,}
          </span>
          <span style="font-size:0.72rem;color:#9CA3AF;margin-top:2px;">
            Papers
          </span>
        </div>
        <div style="display:flex;flex-direction:column;align-items:center;">
          <span style="font-size:1.4rem;font-weight:700;color:#10B981;">
            {entity_count:,}
          </span>
          <span style="font-size:0.72rem;color:#9CA3AF;margin-top:2px;">
            Entities
          </span>
        </div>
        <div style="display:flex;flex-direction:column;align-items:center;">
          <span style="font-size:1.4rem;font-weight:700;color:#9B72CF;">
            {rel_count:,}
          </span>
          <span style="font-size:0.72rem;color:#9CA3AF;margin-top:2px;">
            Relationships
          </span>
        </div>
      </div>
    </div>
    """

    st.markdown(stepper_html, unsafe_allow_html=True)
