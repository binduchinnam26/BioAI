"""
Plotly chart renderers for publication trends, topic evolution, keyword
frequency, and author productivity.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional
from config import (
    COLOR_BACKGROUND, COLOR_SURFACE, COLOR_PRIMARY, COLOR_SUCCESS,
    COLOR_WARNING, COLOR_SECONDARY, COLOR_DANGER, COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY, COMMUNITY_COLORS,
)

_DARK_LAYOUT = dict(
    plot_bgcolor=COLOR_BACKGROUND,
    paper_bgcolor=COLOR_SURFACE,
    font=dict(color=COLOR_TEXT_PRIMARY, family="Open Sans, Arial, sans-serif"),
    xaxis=dict(gridcolor="#1F2937", zerolinecolor="#1F2937"),
    yaxis=dict(gridcolor="#1F2937", zerolinecolor="#1F2937"),
    margin=dict(l=40, r=20, t=40, b=40),
)
_NO_MODEBAR = {"displayModeBar": False}

_KW_TYPE_COLORS = {
    "author_keyword":   "#4E9AF1",
    "mesh_descriptor":  "#34C78A",
    "mesh_qualifier":   "#9B72CF",
    "chemical":         "#F5A623",
    "publication_type": "#E85D5D",
}


def render_publication_trend(papers_df):
    """
    Render a dual-panel publication trend chart: bars for annual counts and
    a spline line for cumulative totals. Shared x-axis via subplot layout.
    """
    import streamlit as st

    if papers_df is None or (hasattr(papers_df, "empty") and papers_df.empty):
        st.info("No publication data available. Run the pipeline first.")
        return

    df = papers_df.copy()
    if "pub_year" not in df.columns:
        st.info("Publication year data is not available.")
        return

    year_counts = (
        df[df["pub_year"].notna()]
        .groupby("pub_year")
        .size()
        .reset_index(name="count")
        .sort_values("pub_year")
    )

    if year_counts.empty:
        st.info("No publication year data to display.")
        return

    year_counts["pub_year"] = year_counts["pub_year"].astype(int)
    year_counts["cumulative"] = year_counts["count"].cumsum()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=("Annual Publications", "Cumulative Publications"),
    )

    # Bar trace — annual counts
    fig.add_trace(
        go.Bar(
            x=year_counts["pub_year"],
            y=year_counts["count"],
            name="Papers / Year",
            marker_color="#3B82F6",
            marker_line_color="#1D4ED8",
            marker_line_width=0.5,
            opacity=0.85,
            hovertemplate=(
                "<b>Year:</b> %{x}<br>"
                "<b>Publications:</b> %{y:,}<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    # Line trace — cumulative (spline)
    fig.add_trace(
        go.Scatter(
            x=year_counts["pub_year"],
            y=year_counts["cumulative"],
            name="Cumulative",
            mode="lines+markers",
            line=dict(color="#10B981", shape="spline", smoothing=1.3, width=2.5),
            marker=dict(size=5, color="#10B981"),
            fill="tozeroy",
            fillcolor="rgba(16,185,129,0.08)",
            hovertemplate=(
                "<b>Year:</b> %{x}<br>"
                "<b>Cumulative:</b> %{y:,}<extra></extra>"
            ),
        ),
        row=2, col=1,
    )

    layout = dict(
        **_DARK_LAYOUT,
        height=480,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_TEXT_SECONDARY, size=12),
        ),
        hovermode="x unified",
        xaxis2=dict(
            gridcolor="#1F2937",
            zerolinecolor="#1F2937",
            tickformat="d",
        ),
        yaxis=dict(gridcolor="#1F2937", zerolinecolor="#1F2937"),
        yaxis2=dict(gridcolor="#1F2937", zerolinecolor="#1F2937"),
    )
    fig.update_layout(**layout)

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.update(font=dict(color=COLOR_TEXT_SECONDARY, size=12))

    st.plotly_chart(fig, use_container_width=True, config=_NO_MODEBAR)


def render_topic_evolution(topics_over_time_df):
    """
    Render a stacked area chart of topic frequency evolution over time.
    Accepts a BERTopic topics_over_time DataFrame with columns:
    Topic, Words, Frequency, Year (or Timestamp).
    """
    import streamlit as st

    if topics_over_time_df is None or (
        hasattr(topics_over_time_df, "empty") and topics_over_time_df.empty
    ):
        st.info("Topic evolution data is not available. Run the pipeline with topic modelling enabled.")
        return

    df = topics_over_time_df.copy()

    # Normalise column names — BERTopic uses 'Timestamp' in some versions
    if "Timestamp" in df.columns and "Year" not in df.columns:
        df["Year"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.year
    if "Year" not in df.columns:
        st.info("Year column not found in topic evolution data.")
        return

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year", "Frequency", "Topic"])
    df["Year"] = df["Year"].astype(int)

    # Exclude outlier topic -1
    df = df[df["Topic"] >= 0]

    if df.empty:
        st.info("No topic evolution data to display after filtering.")
        return

    # Build pivot: rows=Year, columns=Topic label, values=Frequency
    df["label"] = df.apply(
        lambda r: (
            str(r["Words"])[:30] if "Words" in df.columns and str(r["Words"]) != "nan"
            else f"Topic {int(r['Topic'])}"
        ),
        axis=1,
    )
    pivot = df.pivot_table(
        index="Year", columns="label", values="Frequency", aggfunc="sum"
    ).fillna(0)

    # Normalise each year to 0-100 %
    pivot_pct = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) * 100

    fig = go.Figure()
    topic_labels = list(pivot_pct.columns)

    for i, topic_label in enumerate(topic_labels):
        color = COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=pivot_pct.index.tolist(),
                y=pivot_pct[topic_label].tolist(),
                name=topic_label,
                mode="lines",
                stackgroup="one",
                line=dict(width=0.5, color=color, shape="spline"),
                fillcolor=color.replace("#", "rgba(")
                    if not color.startswith("rgba") else color,
                opacity=0.8,
                hovertemplate=(
                    f"<b>{topic_label}</b><br>"
                    "Year: %{x}<br>"
                    "Share: %{y:.1f}%<extra></extra>"
                ),
            )
        )

    layout = dict(
        **_DARK_LAYOUT,
        height=400,
        title=dict(
            text="Topic Landscape Over Time",
            font=dict(color=COLOR_TEXT_PRIMARY, size=14),
            x=0,
        ),
        yaxis=dict(
            title="Relative Frequency (%)",
            gridcolor="#1F2937",
            zerolinecolor="#1F2937",
            ticksuffix="%",
        ),
        xaxis=dict(
            title="Publication Year",
            gridcolor="#1F2937",
            zerolinecolor="#1F2937",
            tickformat="d",
        ),
        legend=dict(
            orientation="v",
            x=1.01,
            y=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_TEXT_SECONDARY, size=10),
            itemwidth=30,
        ),
        hovermode="x unified",
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config=_NO_MODEBAR)


def render_top_keywords(papers_df):
    """
    Render a horizontal bar chart of the top 20 keywords across author keywords,
    MeSH descriptors, MeSH qualifiers, chemical names, and publication types.
    Each keyword type is rendered in a distinct colour with a legend.
    """
    import streamlit as st
    import json

    if papers_df is None or (hasattr(papers_df, "empty") and papers_df.empty):
        st.info("No keyword data available. Run the pipeline first.")
        return

    from collections import Counter
    kw_freq: Counter = Counter()
    kw_type: dict = {}

    def _safe_list(val, fallback_type="list"):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []

    for _, row in papers_df.iterrows():
        # Author keywords
        for kw in _safe_list(row.get("keywords")):
            if isinstance(kw, str) and kw.strip():
                k = kw.strip().lower()
                kw_freq[k] += 1
                kw_type.setdefault(k, "author_keyword")

        # MeSH descriptors
        for m in _safe_list(row.get("mesh_terms")):
            if isinstance(m, dict):
                d = m.get("descriptor", "")
                if d:
                    k = d.strip().lower()
                    kw_freq[k] += 1
                    kw_type.setdefault(k, "mesh_descriptor")

        # MeSH qualifiers
        for q in _safe_list(row.get("mesh_qualifiers")):
            if isinstance(q, str) and q.strip():
                k = q.strip().lower()
                kw_freq[k] += 1
                kw_type.setdefault(k, "mesh_qualifier")

        # Chemical terms
        for c in _safe_list(row.get("chemical_terms")):
            if isinstance(c, dict):
                n = c.get("name", "")
                if n:
                    k = n.strip().lower()
                    kw_freq[k] += 1
                    kw_type.setdefault(k, "chemical")

        # Publication types
        for pt in _safe_list(row.get("publication_types")):
            if isinstance(pt, str) and pt.strip():
                k = pt.strip().lower()
                kw_freq[k] += 1
                kw_type.setdefault(k, "publication_type")

    if not kw_freq:
        st.info("No keyword frequency data could be extracted from the corpus.")
        return

    top_20 = kw_freq.most_common(20)
    if not top_20:
        st.info("No keywords found.")
        return

    labels = [kw for kw, _ in top_20]
    counts = [cnt for _, cnt in top_20]
    types_ = [kw_type.get(kw, "author_keyword") for kw in labels]
    colors = [_KW_TYPE_COLORS.get(t, "#4E9AF1") for t in types_]

    # Build one trace per keyword type for legend grouping
    legend_traces: dict = {}
    for ktype, color in _KW_TYPE_COLORS.items():
        legend_traces[ktype] = go.Bar(
            x=[],
            y=[],
            name=ktype.replace("_", " ").title(),
            orientation="h",
            marker_color=color,
            showlegend=True,
        )

    bar_traces = []
    for label, count, ktype, color in zip(labels, counts, types_, colors):
        shown_in_legend = ktype not in [t.name.lower().replace(" ", "_") for t in bar_traces]
        bar_traces.append(
            go.Bar(
                x=[count],
                y=[label],
                name=ktype.replace("_", " ").title(),
                orientation="h",
                marker_color=color,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"Type: {ktype.replace('_', ' ').title()}<br>"
                    "Frequency: %{x:,}<extra></extra>"
                ),
                legendgroup=ktype,
                showlegend=True,
            )
        )

    # Deduplicate legend entries
    seen_legend = set()
    for tr in bar_traces:
        if tr.name in seen_legend:
            tr.showlegend = False
        else:
            seen_legend.add(tr.name)

    fig = go.Figure(data=bar_traces)
    fig.update_layout(
        **_DARK_LAYOUT,
        height=520,
        barmode="overlay",
        title=dict(
            text="Top 20 Keywords by Frequency",
            font=dict(color=COLOR_TEXT_PRIMARY, size=14),
            x=0,
        ),
        xaxis=dict(
            title="Frequency (papers)",
            gridcolor="#1F2937",
            zerolinecolor="#1F2937",
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=11),
            gridcolor="#1F2937",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_TEXT_SECONDARY, size=11),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config=_NO_MODEBAR)


def render_author_productivity(papers_df):
    """
    Render a horizontal bar chart of the top 15 most productive authors,
    using a gradient fill from dark blue (least) to bright blue (most).
    """
    import streamlit as st
    import json

    if papers_df is None or (hasattr(papers_df, "empty") and papers_df.empty):
        st.info("No author data available. Run the pipeline first.")
        return

    from collections import Counter
    author_counts: Counter = Counter()

    for _, row in papers_df.iterrows():
        authors = row.get("authors")
        if isinstance(authors, str):
            try:
                authors = json.loads(authors)
            except Exception:
                authors = []
        if isinstance(authors, list) and authors:
            first = authors[0]
            if isinstance(first, dict):
                name = first.get("name") or first.get("normalized_name") or ""
            else:
                name = str(first)
            if name.strip():
                author_counts[name.strip()] += 1
        else:
            # Fallback: first_author column
            fa = row.get("first_author")
            if fa and isinstance(fa, str) and fa.strip():
                author_counts[fa.strip()] += 1

    if not author_counts:
        st.info("No author data found in the corpus.")
        return

    top_15 = author_counts.most_common(15)
    if not top_15:
        st.info("No authors found.")
        return

    names = [a for a, _ in reversed(top_15)]
    counts = [c for _, c in reversed(top_15)]

    # Linear colour gradient: most papers=#3B82F6, fewest=#1E3A5F
    max_c = max(counts)
    min_c = min(counts)

    def _lerp_color(count):
        if max_c == min_c:
            return "#3B82F6"
        t = (count - min_c) / (max_c - min_c)
        r = int(0x1E + t * (0x3B - 0x1E))
        g = int(0x3A + t * (0x82 - 0x3A))
        b = int(0x5F + t * (0xF6 - 0x5F))
        return f"#{r:02X}{g:02X}{b:02X}"

    bar_colors = [_lerp_color(c) for c in counts]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=names,
            orientation="h",
            marker_color=bar_colors,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Publications: %{x:,}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **_DARK_LAYOUT,
        height=460,
        title=dict(
            text="Top 15 Authors by Publication Count",
            font=dict(color=COLOR_TEXT_PRIMARY, size=14),
            x=0,
        ),
        xaxis=dict(
            title="Number of Publications",
            gridcolor="#1F2937",
            zerolinecolor="#1F2937",
        ),
        yaxis=dict(
            tickfont=dict(size=11),
            gridcolor="#1F2937",
        ),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config=_NO_MODEBAR)
