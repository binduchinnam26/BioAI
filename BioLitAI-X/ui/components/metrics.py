"""
Metric / KPI display components for BioLitAI-X.
All rendering uses st.markdown with inline HTML.
Streamlit is imported inside each function per project conventions.
"""


def render_kpi_row(
    total_papers: int,
    unique_authors: int,
    unique_keywords: int,
    clusters: int,
    year_range: tuple,
):
    """
    Render a row of 5 KPI cards using st.columns(5).

    Parameters
    ----------
    total_papers    : total number of papers in the corpus
    unique_authors  : number of unique first authors
    unique_keywords : number of unique keywords (all types)
    clusters        : number of detected topic/community clusters
    year_range      : (min_year, max_year) tuple; both may be None
    """
    import streamlit as st

    yr_min, yr_max = year_range if year_range else (None, None)
    if yr_min and yr_max:
        yr_label = f"{yr_min} – {yr_max}"
    elif yr_min:
        yr_label = str(yr_min)
    elif yr_max:
        yr_label = str(yr_max)
    else:
        yr_label = "N/A"

    cards = [
        {
            "label": "Total Papers",
            "value": f"{total_papers:,}",
            "icon": "📄",
            "color": "#3B82F6",
        },
        {
            "label": "Unique Authors",
            "value": f"{unique_authors:,}",
            "icon": "👤",
            "color": "#10B981",
        },
        {
            "label": "Unique Keywords",
            "value": f"{unique_keywords:,}",
            "icon": "🏷️",
            "color": "#9B72CF",
        },
        {
            "label": "Topic Clusters",
            "value": f"{clusters:,}",
            "icon": "🔵",
            "color": "#F5A623",
        },
        {
            "label": "Year Range",
            "value": yr_label,
            "icon": "📅",
            "color": "#E85D5D",
        },
    ]

    cols = st.columns(5)
    for col, card in zip(cols, cards):
        with col:
            render_stat_card(
                label=card["label"],
                value=card["value"],
                color=card["color"],
                icon=card["icon"],
            )


def render_stat_card(
    label: str,
    value,
    color: str = None,
    icon: str = None,
):
    """
    Render a single KPI card as a styled HTML div via st.markdown.

    Parameters
    ----------
    label : card title / metric name
    value : metric value (will be converted to str)
    color : accent colour for the value text (defaults to #3B82F6)
    icon  : optional emoji or Unicode character displayed above the value
    """
    import streamlit as st

    accent = color or "#3B82F6"
    icon_html = (
        f'<div style="font-size:1.5rem;margin-bottom:6px;">{icon}</div>'
        if icon
        else ""
    )

    html = f"""
    <div style="
      background:#1C2539;
      border-radius:8px;
      padding:1.25rem 1rem;
      box-shadow:0 4px 6px rgba(0,0,0,0.3);
      text-align:center;
      transition:transform 200ms ease,box-shadow 200ms ease;
      height:100%;
    "
    onmouseover="this.style.transform='translateY(-2px)';this.style.boxShadow='0 8px 16px rgba(0,0,0,0.4)';"
    onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='0 4px 6px rgba(0,0,0,0.3)';"
    >
      {icon_html}
      <div style="
        font-size:1.7rem;
        font-weight:700;
        color:{accent};
        line-height:1.2;
        margin-bottom:4px;
      ">{value}</div>
      <div style="
        font-size:0.72rem;
        font-weight:600;
        color:#9CA3AF;
        text-transform:uppercase;
        letter-spacing:0.06em;
      ">{label}</div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


def render_mini_stats(stats_dict: dict):
    """
    Render a compact horizontal grid of small labelled badges.

    Parameters
    ----------
    stats_dict : dict mapping label → value, e.g.
                 {"Journals": 42, "With Abstract": "94%", "Languages": 3}
    """
    import streamlit as st

    if not stats_dict:
        return

    badges_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;margin:8px 0;">'

    for label, value in stats_dict.items():
        badges_html += f"""
        <div style="
          background:#111827;
          border:1px solid #1F2937;
          border-radius:20px;
          padding:5px 14px;
          display:inline-flex;
          align-items:center;
          gap:8px;
        ">
          <span style="
            font-size:1rem;
            font-weight:700;
            color:#3B82F6;
          ">{value}</span>
          <span style="
            font-size:0.72rem;
            color:#9CA3AF;
            font-weight:600;
            text-transform:uppercase;
            letter-spacing:0.04em;
          ">{label}</span>
        </div>
        """

    badges_html += "</div>"
    st.markdown(badges_html, unsafe_allow_html=True)
