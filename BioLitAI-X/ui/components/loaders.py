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
    Render pipeline progress as a vertical list of step bars.

    Visual states
    -------------
    waiting  — dark bar, dim text; step not yet started.
    active   — blue left-border, blue label, 5-dot pulse animation on the right.
    complete — green left-border, green label, gradient fill-line + "Done" tag.

    Parameters
    ----------
    steps        : list[str]  Step labels.
    current_step : int        0 … len(steps)-1 = active step index.
                              Pass len(steps) when all steps are done.
    paper_count, entity_count, rel_count : int  Live counters shown below steps.
    """
    import streamlit as st

    n = len(steps)
    all_complete = current_step >= n

    # ── CSS — plain string, NOT f-string, so CSS braces stay literal ─────────
    css = """
    <style>
    @keyframes _blx_pulse {
      0%   { transform:scale(0.8); background-color:#b3d4fc;
             box-shadow:0 0 0 0 rgba(178,212,252,0.7); }
      50%  { transform:scale(1.2); background-color:#6793fb;
             box-shadow:0 0 0 8px rgba(178,212,252,0); }
      100% { transform:scale(0.8); background-color:#b3d4fc;
             box-shadow:0 0 0 0 rgba(178,212,252,0.7); }
    }
    .blx-dot {
      height:8px; width:8px; border-radius:50%;
      background-color:#b3d4fc; flex-shrink:0;
      animation:_blx_pulse 1.5s infinite ease-in-out;
    }
    .blx-dot:nth-child(1) { animation-delay:-0.30s; }
    .blx-dot:nth-child(2) { animation-delay:-0.10s; }
    .blx-dot:nth-child(3) { animation-delay: 0.10s; }
    .blx-dot:nth-child(4) { animation-delay: 0.25s; }
    .blx-dot:nth-child(5) { animation-delay: 0.40s; }
    </style>
    """

    html = css

    # ── Outer card ────────────────────────────────────────────────────────────
    html += (
        '<div style="background:#0A1220;border:1px solid #1F2937;'
        'border-radius:14px;padding:22px 24px 18px 24px;margin-bottom:20px;">'
    )

    # ── Step bars ─────────────────────────────────────────────────────────────
    for i, step in enumerate(steps):
        if all_complete or i < current_step:
            # ── Complete ──────────────────────────────────────────────────────
            html += (
                '<div style="'
                'background:linear-gradient(135deg,#071320 0%,#091A2B 100%);'
                'border:1px solid #1A3350;border-left:3px solid #10B981;'
                'border-radius:8px;padding:12px 18px;margin-bottom:8px;'
                'display:flex;align-items:center;gap:12px;'
                'box-shadow:0 1px 6px rgba(0,0,0,0.25),'
                'inset 0 1px 0 rgba(16,185,129,0.08);">'
                f'<span style="font-size:0.84rem;font-weight:600;color:#34D399;'
                f'letter-spacing:0.01em;flex:1;">{step}</span>'
                '<div style="flex:2;height:1px;'
                'background:linear-gradient(to right,rgba(16,185,129,0.35),transparent);'
                'border-radius:1px;"></div>'
                '<span style="font-size:0.68rem;color:#059669;font-weight:600;'
                'letter-spacing:0.07em;text-transform:uppercase;white-space:nowrap;">'
                'Done</span>'
                '</div>'
            )
        elif i == current_step:
            # ── Active (animated dots) ────────────────────────────────────────
            html += (
                '<div style="'
                'background:linear-gradient(135deg,#0D1B2A 0%,#0F2240 100%);'
                'border:1px solid #1E3A5F;border-left:3px solid #3B82F6;'
                'border-radius:8px;padding:12px 18px;margin-bottom:8px;'
                'display:flex;align-items:center;justify-content:space-between;'
                'box-shadow:0 0 18px rgba(59,130,246,0.10),'
                'inset 0 1px 0 rgba(59,130,246,0.08);">'
                f'<span style="font-size:0.84rem;font-weight:600;color:#93C5FD;'
                f'letter-spacing:0.01em;">{step}</span>'
                '<div style="display:flex;align-items:center;gap:7px;padding-right:2px;">'
                '<div class="blx-dot"></div>'
                '<div class="blx-dot"></div>'
                '<div class="blx-dot"></div>'
                '<div class="blx-dot"></div>'
                '<div class="blx-dot"></div>'
                '</div>'
                '</div>'
            )
        else:
            # ── Waiting ───────────────────────────────────────────────────────
            html += (
                '<div style="'
                'background:#060D1A;border:1px solid #111827;'
                'border-left:3px solid #1C2539;'
                'border-radius:8px;padding:12px 18px;margin-bottom:8px;'
                'display:flex;align-items:center;">'
                f'<span style="font-size:0.84rem;font-weight:500;color:#2D3748;'
                f'letter-spacing:0.01em;">{step}</span>'
                '</div>'
            )

    # ── Live stats bar ────────────────────────────────────────────────────────
    html += (
        '<div style="display:flex;gap:28px;margin-top:16px;padding-top:14px;'
        'border-top:1px solid #1F2937;">'
        # Papers
        '<div style="display:flex;flex-direction:column;align-items:center;">'
        f'<span style="font-size:1.3rem;font-weight:700;color:#3B82F6;">{paper_count:,}</span>'
        '<span style="font-size:0.67rem;color:#6B7280;margin-top:2px;'
        'letter-spacing:0.05em;text-transform:uppercase;">Papers</span>'
        '</div>'
        # Entities
        '<div style="display:flex;flex-direction:column;align-items:center;">'
        f'<span style="font-size:1.3rem;font-weight:700;color:#10B981;">{entity_count:,}</span>'
        '<span style="font-size:0.67rem;color:#6B7280;margin-top:2px;'
        'letter-spacing:0.05em;text-transform:uppercase;">Entities</span>'
        '</div>'
        # Relationships
        '<div style="display:flex;flex-direction:column;align-items:center;">'
        f'<span style="font-size:1.3rem;font-weight:700;color:#9B72CF;">{rel_count:,}</span>'
        '<span style="font-size:0.67rem;color:#6B7280;margin-top:2px;'
        'letter-spacing:0.05em;text-transform:uppercase;">Relationships</span>'
        '</div>'
        '</div>'
        '</div>'   # close outer card
    )

    st.markdown(html, unsafe_allow_html=True)
