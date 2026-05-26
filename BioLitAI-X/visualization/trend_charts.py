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
from utils.helpers import hex_to_rgba

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
    Render a combined publication trend chart: bars for annual counts on the
    left Y-axis and a spline line for cumulative totals on the right Y-axis.

    Bar colours indicate year-over-year growth direction:
      green  (#10B981) = more papers than the previous year
      red    (#EF4444) = fewer papers than the previous year
      blue   (#3B82F6) = first year or unchanged
    A dashed amber line marks the corpus average papers/year.
    X-axis ticks are pinned to actual data years to prevent duplicates.
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

    years  = year_counts["pub_year"].tolist()
    counts = year_counts["count"].tolist()

    # ── Growth-direction bar colours ──────────────────────────────────────────
    bar_colors = []
    for i, c in enumerate(counts):
        if i == 0:
            bar_colors.append("#3B82F6")        # first year — neutral blue
        elif c > counts[i - 1]:
            bar_colors.append("#10B981")        # growth — green
        elif c < counts[i - 1]:
            bar_colors.append("#EF4444")        # decline — red
        else:
            bar_colors.append("#3B82F6")        # unchanged — neutral blue

    avg_per_year = year_counts["count"].mean()

    fig = go.Figure()

    # Bar trace — annual counts (left Y-axis)
    fig.add_trace(
        go.Bar(
            x=years,
            y=counts,
            name="Papers / Year",
            marker_color=bar_colors,
            marker_line_color="rgba(0,0,0,0.15)",
            marker_line_width=0.5,
            opacity=0.88,
            yaxis="y1",
            hovertemplate=(
                "<b>Year:</b> %{x}<br>"
                "<b>Publications:</b> %{y:,}<extra></extra>"
            ),
        )
    )

    # Average reference line (dashed amber, left Y-axis)
    fig.add_trace(
        go.Scatter(
            x=[years[0], years[-1]],
            y=[avg_per_year, avg_per_year],
            name=f"Avg {avg_per_year:.1f} / yr",
            mode="lines",
            line=dict(color="#F59E0B", width=1.5, dash="dash"),
            yaxis="y1",
            hovertemplate=(
                f"Average: {avg_per_year:.1f} papers/yr<extra></extra>"
            ),
        )
    )

    # Cumulative line (right Y-axis, spline)
    fig.add_trace(
        go.Scatter(
            x=years,
            y=year_counts["cumulative"].tolist(),
            name="Cumulative",
            mode="lines+markers",
            line=dict(color="#10B981", shape="spline", smoothing=1.3, width=2.5),
            marker=dict(size=6, color="#10B981"),
            yaxis="y2",
            hovertemplate=(
                "<b>Year:</b> %{x}<br>"
                "<b>Cumulative:</b> %{y:,}<extra></extra>"
            ),
        )
    )

    fig.update_layout(**{
        **_DARK_LAYOUT,
        "height": 420,
        "hovermode": "x unified",
        "showlegend": True,
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_TEXT_SECONDARY, size=12),
        ),
        "xaxis": dict(
            gridcolor="#1F2937",
            zerolinecolor="#1F2937",
            tickmode="array",
            tickvals=years,
            tickformat="d",
        ),
        "yaxis": dict(
            title=dict(text="Papers / Year", font=dict(color="#3B82F6")),
            tickfont=dict(color="#3B82F6"),
            gridcolor="#1F2937",
            zerolinecolor="#1F2937",
        ),
        "yaxis2": dict(
            title=dict(text="Cumulative", font=dict(color="#10B981")),
            tickfont=dict(color="#10B981"),
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            zerolinecolor="#1F2937",
        ),
    })

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
        ts = pd.to_numeric(df["Timestamp"], errors="coerce")
        # If values are already plain years (1900–2200), use them directly.
        # Passing raw integers to pd.to_datetime interprets them as nanoseconds
        # → 1970, which is wrong.
        if ts.notna().all() and ts.between(1900, 2200).all():
            df["Year"] = ts.astype(int)
        else:
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
                fillcolor=hex_to_rgba(color, 0.6) if color.startswith("#") else color,
                opacity=0.8,
                hovertemplate=(
                    f"<b>{topic_label}</b><br>"
                    "Year: %{x}<br>"
                    "Share: %{y:.1f}%<extra></extra>"
                ),
            )
        )

    layout = {
        **_DARK_LAYOUT,
        "height": 400,
        "title": dict(
            text="Topic Landscape Over Time",
            font=dict(color=COLOR_TEXT_PRIMARY, size=14),
            x=0,
        ),
        "yaxis": dict(
            title="Relative Frequency (%)",
            gridcolor="#1F2937",
            zerolinecolor="#1F2937",
            ticksuffix="%",
        ),
        "xaxis": dict(
            title="Publication Year",
            gridcolor="#1F2937",
            zerolinecolor="#1F2937",
            tickformat="d",
        ),
        "legend": dict(
            orientation="v",
            x=1.01,
            y=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_TEXT_SECONDARY, size=10),
            itemwidth=30,
        ),
        "hovermode": "x unified",
    }
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
    fig.update_layout(**{
        **_DARK_LAYOUT,
        "height": 520,
        "barmode": "overlay",
        "title": dict(
            text="Top 20 Keywords by Frequency",
            font=dict(color=COLOR_TEXT_PRIMARY, size=14),
            x=0,
        ),
        "xaxis": dict(
            title="Frequency (papers)",
            gridcolor="#1F2937",
            zerolinecolor="#1F2937",
        ),
        "yaxis": dict(
            autorange="reversed",
            tickfont=dict(size=11),
            gridcolor="#1F2937",
        ),
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_TEXT_SECONDARY, size=11),
        ),
    })
    st.plotly_chart(fig, use_container_width=True, config=_NO_MODEBAR)


def render_author_productivity(papers_df):
    """
    Render a horizontal stacked bar chart of the top 15 most productive authors,
    split into first-author (blue) vs co-author (purple) contributions.
    """
    import streamlit as st
    import json

    if papers_df is None or (hasattr(papers_df, "empty") and papers_df.empty):
        st.info("No author data available. Run the pipeline first.")
        return

    from collections import Counter
    first_author_counts: Counter = Counter()
    coauthor_counts: Counter = Counter()

    for _, row in papers_df.iterrows():
        authors = row.get("authors")
        if isinstance(authors, str):
            try:
                authors = json.loads(authors)
            except Exception:
                authors = []
        if isinstance(authors, list) and authors:
            for i, author in enumerate(authors):
                if isinstance(author, dict):
                    name = author.get("name") or author.get("normalized_name") or ""
                else:
                    name = str(author)
                if name.strip():
                    if i == 0:
                        first_author_counts[name.strip()] += 1
                    else:
                        coauthor_counts[name.strip()] += 1
        else:
            # Fallback: first_author column
            fa = row.get("first_author")
            if fa and isinstance(fa, str) and fa.strip():
                first_author_counts[fa.strip()] += 1

    all_authors = set(first_author_counts.keys()) | set(coauthor_counts.keys())
    if not all_authors:
        st.info("No author data found in the corpus.")
        return

    total_counts = {
        a: first_author_counts.get(a, 0) + coauthor_counts.get(a, 0)
        for a in all_authors
    }

    top_15 = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    if not top_15:
        st.info("No authors found.")
        return

    # Reverse so highest total is at the top in a horizontal bar chart
    top_15_rev = list(reversed(top_15))
    names = [a for a, _ in top_15_rev]
    first_counts = [first_author_counts.get(a, 0) for a in names]
    co_counts = [coauthor_counts.get(a, 0) for a in names]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=first_counts,
        y=names,
        name="First Author",
        orientation="h",
        marker_color="#3B82F6",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "First-author papers: %{x:,}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Bar(
        x=co_counts,
        y=names,
        name="Co-author",
        orientation="h",
        marker_color="#8B5CF6",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Co-author papers: %{x:,}<extra></extra>"
        ),
    ))

    fig.update_layout(**{
        **_DARK_LAYOUT,
        "height": 460,
        "barmode": "stack",
        "title": dict(
            text="Top 15 Authors by Publication Count",
            font=dict(color=COLOR_TEXT_PRIMARY, size=14),
            x=0,
        ),
        "xaxis": dict(
            title="Number of Publications",
            gridcolor="#1F2937",
            zerolinecolor="#1F2937",
            tickformat="d",
            dtick=1,
        ),
        "yaxis": dict(
            tickfont=dict(size=11),
            gridcolor="#1F2937",
        ),
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLOR_TEXT_SECONDARY, size=11),
        ),
        "showlegend": True,
    })
    st.plotly_chart(fig, use_container_width=True, config=_NO_MODEBAR)
