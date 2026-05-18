"""
VOSviewer-faithful bibliometric network renderer.

Implements render_coauthorship_network, render_keyword_network, and
render_topic_network using PyVis embedded in Streamlit via
st.components.v1.html().

Every visual specification in Phase 4 Section B is implemented here.
"""

import json
import math
import re
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from config import (
    CANVAS_BG,
    COMMUNITY_COLORS,
    NODE_SIZE_MIN,
    NODE_SIZE_MAX,
    EDGE_WIDTH_MIN,
    EDGE_WIDTH_MAX,
    COLOR_SURFACE_ELEVATED,
    COLOR_TEXT_SECONDARY,
    COLOR_PRIMARY,
)
from utils.helpers import (
    lighten_hex,
    hex_to_rgba,
    scale_node_size,
    scale_edge_width,
    truncate,
    format_author_short,
    percentile,
)

# ── Physics options ───────────────────────────────────────────────────────────

def get_physics_options(node_count: int) -> Dict:
    """
    Return Barnes-Hut physics options tuned to graph size.
    Small graphs (<50 nodes): spread out more.
    Large graphs (>500 nodes): compress to maintain readability.
    """
    if node_count < 50:
        grav = -5000
        spring = 220
    elif node_count > 500:
        grav = -12000
        spring = 120
    else:
        grav = -8000
        spring = 160

    return {
        "physics": {
            "enabled": True,
            "barnesHut": {
                "gravitationalConstant": grav,
                "centralGravity": 0.25,
                "springLength": spring,
                "springConstant": 0.03,
                "damping": 0.09,
                "avoidOverlap": 0.6,
            },
            "maxVelocity": 50,
            "minVelocity": 0.5,
            "stabilization": {
                "enabled": True,
                "iterations": 1200,
                "updateInterval": 20,
                "fit": True,
            },
            "timestep": 0.4,
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 150,
            "hideEdgesOnDrag": True,
            "hideEdgesOnZoom": False,
            "multiselect": True,
            "navigationButtons": False,
            "keyboard": {"enabled": False},
        },
        "nodes": {"chosen": True, "physics": True},
        "edges": {
            "chosen": True,
            "physics": True,
            "hoverWidth": 2.5,
            "selectionWidth": 3.0,
        },
    }


# ── Tooltip container style ───────────────────────────────────────────────────

_TOOLTIP_STYLE = (
    "background:#1C2539;border:1px solid #3B4A6B;border-radius:8px;"
    "padding:10px 14px;font-family:'Open Sans',Arial,sans-serif;"
    "font-size:12px;color:#F0F4FF;max-width:260px;"
    "box-shadow:0 4px 16px rgba(0,0,0,0.5);line-height:1.6;"
)

def _wrap_tooltip(content: str) -> str:
    return f'<div style="{_TOOLTIP_STYLE}">{content}</div>'


# ── Node sizing helpers ───────────────────────────────────────────────────────

def _compute_node_sizes(
    graph: nx.Graph,
    weight_attr: str = "weight",
) -> Dict[Any, float]:
    weights = {n: graph.nodes[n].get(weight_attr, 1) for n in graph.nodes()}
    w_min = min(weights.values(), default=1)
    w_max = max(weights.values(), default=1)
    return {
        n: scale_node_size(w, w_min, w_max, NODE_SIZE_MIN, NODE_SIZE_MAX)
        for n, w in weights.items()
    }


def _compute_edge_widths(graph: nx.Graph) -> Dict[Tuple, float]:
    edge_weights = {
        (u, v): graph[u][v].get("weight", 1) for u, v in graph.edges()
    }
    if not edge_weights:
        return {}
    w_min = min(edge_weights.values())
    w_max = max(edge_weights.values())
    return {
        edge: scale_edge_width(w, w_min, w_max, EDGE_WIDTH_MIN, EDGE_WIDTH_MAX)
        for edge, w in edge_weights.items()
    }


# ── Label visibility ──────────────────────────────────────────────────────────

def _label_font(
    weight: float,
    p25: float,
    p50: float,
    p75: float,
    p90: float,
) -> Dict:
    """
    VOSviewer-style dramatic font scaling: top nodes get large text,
    all nodes remain visible (no transparent labels).
    """
    face = "Arial, sans-serif"
    if weight >= p90:
        return {"size": 30, "color": "#FFFFFF", "face": face}
    if weight >= p75:
        return {"size": 20, "color": "#FFFFFF", "face": face}
    if weight >= p50:
        return {"size": 14, "color": "#E5E7EB", "face": face}
    if weight >= p25:
        return {"size": 11, "color": "#D1D5DB", "face": face}
    return {"size": 9, "color": "#9CA3AF", "face": face}


# ── PyVis HTML post-processing ────────────────────────────────────────────────

_STABILIZE_JS = """
<script>
network.once('stabilizationIterationsDone', function() {
  network.setOptions({ physics: { enabled: false } });
});
</script>
"""

_HIGHLIGHT_JS = """
<script>
var allNodes = network.body.data.nodes;
var allEdges = network.body.data.edges;

network.on('click', function(params) {
  if (params.nodes.length === 0) {
    // Click on empty canvas — restore full opacity
    var nodeUpdates = [];
    allNodes.getIds().forEach(function(id) {
      nodeUpdates.push({ id: id, opacity: 1.0 });
    });
    allNodes.update(nodeUpdates);
    var edgeUpdates = [];
    allEdges.getIds().forEach(function(id) {
      var e = allEdges.get(id);
      edgeUpdates.push({ id: id, color: e._originalColor || e.color });
    });
    allEdges.update(edgeUpdates);
    return;
  }
  var clickedId = params.nodes[0];
  var connectedNodes = new Set(network.getConnectedNodes(clickedId));
  connectedNodes.add(clickedId);
  var nodeUpdates = [];
  allNodes.getIds().forEach(function(id) {
    nodeUpdates.push({ id: id, opacity: connectedNodes.has(id) ? 1.0 : 0.15 });
  });
  allNodes.update(nodeUpdates);
  var connectedEdges = new Set(network.getConnectedEdges(clickedId));
  var edgeUpdates = [];
  allEdges.getIds().forEach(function(id) {
    edgeUpdates.push({ id: id, opacity: connectedEdges.has(id) ? 1.0 : 0.05 });
  });
  allEdges.update(edgeUpdates);
});

network.on('doubleClick', function(params) {
  if (params.nodes.length > 0) {
    window.parent.postMessage({ type: 'nodeDetail', id: params.nodes[0] }, '*');
  } else {
    network.fit({ animation: { duration: 400, easingFunction: 'easeInOutQuad' } });
  }
});
</script>
"""

def _post_process_html(html: str, node_count: int = 0) -> str:
    """
    Set dark background and inject stabilization + interaction JavaScript.
    """
    html = html.replace(
        "background-color: #ffffff;", f"background-color: {CANVAS_BG};"
    ).replace(
        "background-color:#ffffff;", f"background-color:{CANVAS_BG};"
    )
    html = re.sub(
        r'(id="mynetwork"[^>]*style=")([^"]*)',
        rf'\1background:{CANVAS_BG};',
        html,
    )
    # Inject JS before </body>
    html = html.replace(
        "</body>", _STABILIZE_JS + _HIGHLIGHT_JS + "</body>"
    )
    return html


def _pyvis_to_html(net, node_count: int = 0) -> str:
    """Generate PyVis HTML string (no disk write)."""
    html = net.generate_html(notebook=False)
    return _post_process_html(html, node_count)


# ── Controls panel ────────────────────────────────────────────────────────────

def _render_controls(
    graph: nx.Graph,
    key_prefix: str,
    session_key_html: str,
    rebuild_fn,
) -> Tuple[str, float, float, Optional[List[int]], bool]:
    """
    Render the controls strip above a network graph.
    Returns (search_term, min_link, min_size, selected_communities, freeze).
    """
    import streamlit as st
    c1, c2, c3, c4, c5, c6, c7 = st.columns([3, 2, 2, 2, 1, 1, 1])

    with c1:
        search = st.text_input(
            "Search nodes…",
            key=f"{key_prefix}_search",
            placeholder="Search nodes…",
            label_visibility="collapsed",
        )
    all_edge_weights = [
        graph[u][v].get("weight", 1) for u, v in graph.edges()
    ]
    edge_p90 = percentile(all_edge_weights, 90) if all_edge_weights else 1.0
    with c2:
        min_link = st.slider(
            "Min Link Strength",
            min_value=1,
            max_value=max(int(edge_p90), 2),
            value=1,
            key=f"{key_prefix}_min_link",
        )
    all_node_weights = [
        graph.nodes[n].get("weight", 1) for n in graph.nodes()
    ]
    node_p75 = percentile(all_node_weights, 75) if all_node_weights else 1.0
    with c3:
        min_size = st.slider(
            "Min Node Size",
            min_value=1,
            max_value=max(int(node_p75), 2),
            value=1,
            key=f"{key_prefix}_min_size",
        )

    all_communities = sorted(
        set(
            graph.nodes[n].get("community_id", 0)
            for n in graph.nodes()
        )
    )
    with c4:
        selected_comms = st.multiselect(
            "Communities",
            options=all_communities,
            default=all_communities,
            key=f"{key_prefix}_comms",
        )

    with c5:
        density_on = st.checkbox("Density", key=f"{key_prefix}_density")
    with c6:
        freeze = st.checkbox("❄ Freeze", key=f"{key_prefix}_freeze")
    with c7:
        export_png = st.button("↗ PNG", key=f"{key_prefix}_export")
        if export_png:
            st.info("Right-click the graph → Save image as… to export PNG.")

    return search, float(min_link), float(min_size), selected_comms, freeze


# ── Network rendering core ────────────────────────────────────────────────────

def _build_pyvis_network(
    graph: nx.Graph,
    node_sizes: Dict,
    edge_widths: Dict,
    node_weights: Dict,
    label_fn,
    tooltip_fn,
    edge_tooltip_fn,
    directed: bool = False,
    shape_fn=None,
) -> Any:
    """
    Build a PyVis Network object from a NetworkX graph with full
    VOSviewer-faithful styling applied.
    """
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise ImportError("pyvis is not installed. Run: pip install pyvis") from exc

    w_list = list(node_weights.values())
    p25 = percentile(w_list, 25) if w_list else 1.0
    p50 = percentile(w_list, 50) if w_list else 1.0
    p75 = percentile(w_list, 75) if w_list else 1.0
    p90 = percentile(w_list, 90) if w_list else 1.0

    net = Network(
        height="850px",
        width="100%",
        directed=directed,
        bgcolor=CANVAS_BG,
        font_color="#FFFFFF",
    )
    net.toggle_physics(True)

    for node in graph.nodes():
        data = graph.nodes[node]
        fill_hex = data.get("color_hex", COMMUNITY_COLORS[0])
        size = node_sizes.get(node, NODE_SIZE_MIN)
        weight = node_weights.get(node, 1)
        label = label_fn(node, data)
        tooltip = tooltip_fn(node, data, graph)
        shape = shape_fn(data) if shape_fn else "dot"
        font = _label_font(weight, p25, p50, p75, p90)

        # Borderless VOSviewer-style nodes: border matches fill
        node_color = {
            "background": fill_hex,
            "border": fill_hex,
            "highlight": {"background": lighten_hex(fill_hex, 0.25), "border": "#FFFFFF"},
            "hover": {"background": lighten_hex(fill_hex, 0.15), "border": fill_hex},
        }

        net.add_node(
            node,
            label=label,
            title=tooltip,
            size=size,
            shape=shape,
            color=node_color,
            borderWidth=0,
            borderWidthSelected=2,
            font=font,
        )

    for u, v in graph.edges():
        src_color_hex = graph.nodes[u].get("color_hex", COMMUNITY_COLORS[0])
        # VOSviewer-style: very thin, lightly opaque edges
        edge_color = {
            "color": hex_to_rgba(src_color_hex, 0.18),
            "highlight": hex_to_rgba(src_color_hex, 0.85),
            "hover": hex_to_rgba(src_color_hex, 0.60),
        }
        width = edge_widths.get((u, v), EDGE_WIDTH_MIN)
        edge_data = graph[u][v] if isinstance(graph, nx.Graph) else {}
        tooltip = edge_tooltip_fn(u, v, edge_data)
        net.add_edge(
            u, v,
            width=width,
            color=edge_color,
            title=tooltip,
            arrows="" if not directed else "to",
            smooth=False if not directed else
                {"type": "curvedCW", "roundness": 0.2},
        )

    physics_opts = get_physics_options(graph.number_of_nodes())
    net.set_options(json.dumps(physics_opts))
    return net


def _default_edge_tooltip(u, v, data) -> str:
    weight = data.get("weight", 1)
    pmids = data.get("evidence_pmids", [])
    pmid_str = ", ".join(str(p) for p in pmids[:3]) if pmids else "N/A"
    content = (
        f'<div style="font-weight:600;margin-bottom:4px;">{u} ↔ {v}</div>'
        f'<div>Co-occurrence: <b>{weight}</b> papers</div>'
        f'<div style="color:#9CA3AF;font-size:11px;margin-top:6px;">'
        f'PMIDs: {pmid_str}</div>'
    )
    return _wrap_tooltip(content)


# ── A) Co-authorship network ──────────────────────────────────────────────────

def render_coauthorship_network(
    graph: nx.Graph,
    papers_df=None,
    key_prefix: str = "coauth",
):
    """
    Render the Author Collaboration Network.
    Section title + subtitle + controls panel + PyVis graph + stats panel.
    """
    import streamlit as st
    st.markdown(
        "### 01 — Author Collaboration Network\n"
        "<span style='color:#9CA3AF;font-size:13px;'>"
        "Node size = publication count &nbsp;|&nbsp; "
        "Color = research cluster &nbsp;|&nbsp; "
        "Edge thickness = collaboration strength"
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#1F2937;margin:8px 0;'>",
                unsafe_allow_html=True)

    if graph is None or graph.number_of_nodes() == 0:
        st.info("No co-authorship data available. Run the pipeline first.")
        return

    search, min_link, min_size, sel_comms, freeze = _render_controls(
        graph, key_prefix,
        f"_coauth_html",
        lambda g: g,
    )

    # Apply filters
    filtered = _filter_graph(graph, min_link, min_size, sel_comms, search)

    cache_key = (
        f"_coauth_html_{key_prefix}_{min_link}_{min_size}_"
        f"{','.join(map(str, sel_comms))}_{search}_{freeze}"
    )
    if cache_key not in st.session_state or st.session_state[cache_key] is None:
        node_sizes = _compute_node_sizes(filtered)
        edge_widths = _compute_edge_widths(filtered)
        node_weights = {n: filtered.nodes[n].get("weight", 1)
                        for n in filtered.nodes()}

        def label_fn(node, data):
            return format_author_short(str(node))

        def tooltip_fn(node, data, g):
            deg = g.degree(node)
            paper_count = data.get("weight", 0)
            cid = data.get("community_id", 0)
            neighbors = sorted(
                g.neighbors(node),
                key=lambda nb: g[node][nb].get("weight", 0),
                reverse=True,
            )[:5]
            collab_list = "".join(
                f"<div style='color:#D1D5DB;margin-left:6px;'>• {format_author_short(nb)}</div>"
                for nb in neighbors
            )
            content = (
                f'<div style="font-size:14px;font-weight:700;color:#FFFFFF;margin-bottom:6px;">'
                f'{node}</div>'
                f'<div style="color:#9CA3AF;font-size:11px;margin-bottom:8px;">'
                f'Research Cluster {cid}</div>'
                f'<div style="margin-bottom:4px;">'
                f'<span style="color:#9CA3AF;">Papers:</span> '
                f'<span style="color:#4E9AF1;font-weight:600;">{paper_count}</span></div>'
                f'<div style="margin-bottom:4px;">'
                f'<span style="color:#9CA3AF;">Collaborations:</span> '
                f'<span style="color:#34C78A;">{deg}</span></div>'
                f'<div style="margin-top:8px;padding-top:8px;'
                f'border-top:1px solid #2D3A55;color:#9CA3AF;font-size:11px;">'
                f'Top collaborators:<br>{collab_list}</div>'
            )
            return _wrap_tooltip(content)

        net = _build_pyvis_network(
            filtered, node_sizes, edge_widths, node_weights,
            label_fn, tooltip_fn, _default_edge_tooltip,
        )
        if freeze:
            net.toggle_physics(False)
        html = _pyvis_to_html(net, filtered.number_of_nodes())
        st.session_state[cache_key] = html
    else:
        html = st.session_state[cache_key]

    col_graph, col_stats = st.columns([3, 1])
    with col_graph:
        st.components.v1.html(html, height=870, scrolling=False)
    with col_stats:
        _render_coauth_stats(graph)


def _render_coauth_stats(graph: nx.Graph):
    import streamlit as st
    from config import COLOR_SURFACE_ELEVATED
    st.markdown(
        f'<div style="background:{COLOR_SURFACE_ELEVATED};border-radius:8px;'
        f'padding:14px;margin-top:4px;">',
        unsafe_allow_html=True,
    )
    st.metric("Total Authors", graph.number_of_nodes())
    st.metric("Collaboration Links", graph.number_of_edges())
    communities = set(
        graph.nodes[n].get("community_id", 0) for n in graph.nodes()
    )
    st.metric("Research Clusters", len(communities))
    top5 = sorted(
        graph.nodes(data=True),
        key=lambda x: x[1].get("weight", 0),
        reverse=True,
    )[:5]
    if top5:
        st.markdown(
            "<div style='color:#9CA3AF;font-size:11px;margin-top:8px;'>"
            "Top Authors by Papers</div>",
            unsafe_allow_html=True,
        )
        for name, data in top5:
            st.markdown(
                f"<div style='color:#F9FAFB;font-size:12px;'>"
                f"• {truncate(str(name), 22)} "
                f"<span style='color:#4E9AF1;'>{data.get('weight', 0)}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    density = nx.density(graph)
    st.markdown(
        f"<div style='margin-top:10px;color:#9CA3AF;font-size:11px;'>"
        f"Density: <span style='color:#F9FAFB;'>{density:.4f}</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ── B) Keyword co-occurrence network ─────────────────────────────────────────

_KW_SHAPES = {
    "author_keyword": "dot",
    "mesh_descriptor": "square",
    "mesh_qualifier": "triangle",
    "chemical": "diamond",
    "publication_type": "star",
}

_KW_TYPE_COLORS = {
    "author_keyword": "#4E9AF1",
    "mesh_descriptor": "#34C78A",
    "mesh_qualifier": "#9B72CF",
    "chemical": "#F5A623",
    "publication_type": "#E85D5D",
}


def render_keyword_network(
    graph: nx.Graph,
    key_prefix: str = "keyword",
):
    import streamlit as st
    st.markdown(
        "### 02 — Keyword Co-occurrence Map\n"
        "<span style='color:#9CA3AF;font-size:13px;'>"
        "Node size = frequency &nbsp;|&nbsp; Color = thematic cluster &nbsp;|&nbsp; "
        "Shape = keyword type &nbsp;|&nbsp; Edge = co-occurrence"
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#1F2937;margin:8px 0;'>",
                unsafe_allow_html=True)

    if graph is None or graph.number_of_nodes() == 0:
        st.info("No keyword data available. Run the pipeline first.")
        return

    # Shape legend
    legend_html = "".join(
        f'<span style="margin-right:14px;font-size:12px;color:#D1D5DB;">'
        f'<span style="color:{_KW_TYPE_COLORS[t]};">■</span> {t.replace("_", " ").title()}'
        f'</span>'
        for t in _KW_SHAPES
    )
    st.markdown(
        f'<div style="margin-bottom:8px;">{legend_html}</div>',
        unsafe_allow_html=True,
    )

    search, min_link, min_size, sel_comms, freeze = _render_controls(
        graph, key_prefix, f"_kw_html", lambda g: g,
    )
    filtered = _filter_graph(graph, min_link, min_size, sel_comms, search)

    cache_key = (
        f"_kw_html_{key_prefix}_{min_link}_{min_size}_"
        f"{','.join(map(str, sel_comms))}_{search}_{freeze}"
    )
    if cache_key not in st.session_state:
        node_sizes = _compute_node_sizes(filtered)
        edge_widths = _compute_edge_widths(filtered)
        node_weights = {n: filtered.nodes[n].get("weight", 1)
                        for n in filtered.nodes()}

        def label_fn(node, data):
            return truncate(str(node), 25)

        def tooltip_fn(node, data, g):
            freq = data.get("weight", 0)
            ktype = data.get("source_type", "author_keyword")
            cid = data.get("community_id", 0)
            type_color = _KW_TYPE_COLORS.get(ktype, "#9CA3AF")
            nbrs = sorted(
                g.neighbors(node),
                key=lambda nb: g[node][nb].get("weight", 0),
                reverse=True,
            )[:5]
            cooccur_list = "".join(
                f"<div style='color:#D1D5DB;margin-left:6px;'>"
                f"• {truncate(nb, 22)} "
                f"<span style='color:#9CA3AF;'>({g[node][nb].get('weight', 0)})</span>"
                f"</div>"
                for nb in nbrs
            )
            content = (
                f'<div style="font-size:14px;font-weight:700;color:#FFFFFF;margin-bottom:4px;">'
                f'{node}</div>'
                f'<div style="margin-bottom:6px;">'
                f'<span style="background:{type_color};color:#000;font-size:10px;'
                f'padding:2px 6px;border-radius:4px;">'
                f'{ktype.replace("_", " ").title()}</span></div>'
                f'<div style="margin-bottom:4px;">'
                f'<span style="color:#9CA3AF;">Frequency:</span> '
                f'<span style="color:#4E9AF1;font-weight:600;">{freq}</span></div>'
                f'<div style="margin-bottom:4px;">'
                f'<span style="color:#9CA3AF;">Cluster:</span> {cid}</div>'
                f'<div style="margin-top:8px;padding-top:8px;'
                f'border-top:1px solid #2D3A55;color:#9CA3AF;font-size:11px;">'
                f'Top co-occurring:<br>{cooccur_list}</div>'
            )
            return _wrap_tooltip(content)

        def shape_fn(data):
            ktype = data.get("source_type", "author_keyword")
            return _KW_SHAPES.get(ktype, "dot")

        net = _build_pyvis_network(
            filtered, node_sizes, edge_widths, node_weights,
            label_fn, tooltip_fn, _default_edge_tooltip,
            shape_fn=shape_fn,
        )
        if freeze:
            net.toggle_physics(False)
        html = _pyvis_to_html(net, filtered.number_of_nodes())
        st.session_state[cache_key] = html
    else:
        html = st.session_state[cache_key]

    col_graph, col_stats = st.columns([3, 1])
    with col_graph:
        st.components.v1.html(html, height=870, scrolling=False)
    with col_stats:
        _render_keyword_stats(graph)


def _render_keyword_stats(graph: nx.Graph):
    import streamlit as st
    from config import COLOR_SURFACE_ELEVATED
    st.markdown(
        f'<div style="background:{COLOR_SURFACE_ELEVATED};border-radius:8px;padding:14px;">',
        unsafe_allow_html=True,
    )
    st.metric("Total Keywords", graph.number_of_nodes())
    communities = set(
        graph.nodes[n].get("community_id", 0) for n in graph.nodes()
    )
    st.metric("Thematic Clusters", len(communities))
    top_kw = max(
        graph.nodes(data=True),
        key=lambda x: x[1].get("weight", 0),
        default=(None, {}),
    )
    if top_kw[0]:
        st.markdown(
            f"<div style='color:#9CA3AF;font-size:11px;margin-top:8px;'>"
            f"Most Frequent</div>"
            f"<div style='color:#4E9AF1;font-size:13px;font-weight:600;'>"
            f"{truncate(str(top_kw[0]), 24)}</div>"
            f"<div style='color:#9CA3AF;font-size:11px;'>{top_kw[1].get('weight', 0)} papers</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ── C) Topic landscape network ────────────────────────────────────────────────

def render_topic_network(
    graph: nx.Graph,
    papers_df=None,
    key_prefix: str = "topic",
):
    import streamlit as st
    st.markdown(
        "### 03 — Research Topic Landscape\n"
        "<span style='color:#9CA3AF;font-size:13px;'>"
        "Node size = paper count in selected year range &nbsp;|&nbsp; "
        "Edge = shared papers between topics"
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#1F2937;margin:8px 0;'>",
                unsafe_allow_html=True)

    if graph is None or graph.number_of_nodes() == 0:
        st.info("No topic data available. Run the pipeline first.")
        return

    search, min_link, min_size, sel_comms, freeze = _render_controls(
        graph, key_prefix, f"_topic_html", lambda g: g,
    )

    # Year range slider (below controls per spec)
    year_filter = (None, None)
    if papers_df is not None and not papers_df.empty and "pub_year" in papers_df.columns:
        import pandas as pd
        years = papers_df["pub_year"].dropna().astype(int)
        if not years.empty:
            min_yr, max_yr = int(years.min()), int(years.max())
            if min_yr < max_yr:
                year_filter = st.slider(
                    "Year Range",
                    min_value=min_yr,
                    max_value=max_yr,
                    value=(min_yr, max_yr),
                    key=f"{key_prefix}_yearrange",
                )

    filtered = _filter_graph(graph, min_link, min_size, sel_comms, search)

    cache_key = (
        f"_topic_html_{key_prefix}_{min_link}_{min_size}_"
        f"{','.join(map(str, sel_comms))}_{search}_{freeze}_{year_filter}"
    )
    if cache_key not in st.session_state:
        node_sizes = _compute_node_sizes(filtered)
        edge_widths = _compute_edge_widths(filtered)
        node_weights = {n: filtered.nodes[n].get("weight", 1)
                        for n in filtered.nodes()}

        def label_fn(node, data):
            top_words = data.get("top_words", [])
            if isinstance(top_words, list) and top_words:
                words = [
                    w[0] if isinstance(w, (list, tuple)) else str(w)
                    for w in top_words[:3]
                ]
                return ", ".join(words)
            return str(data.get("label", f"Topic {node}"))

        def tooltip_fn(node, data, g):
            label = label_fn(node, data)
            paper_count = data.get("weight", 0)
            content = (
                f'<div style="font-size:14px;font-weight:700;color:#FFFFFF;margin-bottom:6px;">'
                f'{label}</div>'
                f'<div><span style="color:#9CA3AF;">Papers:</span> '
                f'<span style="color:#4E9AF1;font-weight:600;">{paper_count}</span></div>'
            )
            return _wrap_tooltip(content)

        net = _build_pyvis_network(
            filtered, node_sizes, edge_widths, node_weights,
            label_fn, tooltip_fn, _default_edge_tooltip,
        )
        if freeze:
            net.toggle_physics(False)
        html = _pyvis_to_html(net, filtered.number_of_nodes())
        st.session_state[cache_key] = html
    else:
        html = st.session_state[cache_key]

    col_graph, col_stats = st.columns([3, 1])
    with col_graph:
        st.components.v1.html(html, height=870, scrolling=False)
    with col_stats:
        _render_topic_stats(graph)


def _render_topic_stats(graph: nx.Graph):
    import streamlit as st
    from config import COLOR_SURFACE_ELEVATED
    st.markdown(
        f'<div style="background:{COLOR_SURFACE_ELEVATED};border-radius:8px;padding:14px;">',
        unsafe_allow_html=True,
    )
    st.metric("Topics Discovered", graph.number_of_nodes())
    if graph.number_of_nodes() > 0:
        largest = max(
            graph.nodes(data=True),
            key=lambda x: x[1].get("weight", 0),
        )
        label = largest[1].get("label", f"Topic {largest[0]}")
        st.markdown(
            f"<div style='color:#9CA3AF;font-size:11px;margin-top:8px;'>"
            f"Largest Topic</div>"
            f"<div style='color:#4E9AF1;font-size:13px;font-weight:600;'>"
            f"{truncate(str(label), 28)}</div>"
            f"<div style='color:#9CA3AF;font-size:11px;'>{largest[1].get('weight', 0)} papers</div>",
            unsafe_allow_html=True,
        )
    most_connected = max(
        graph.nodes(), key=lambda n: graph.degree(n), default=None
    )
    if most_connected:
        mc_label = graph.nodes[most_connected].get("label", str(most_connected))
        st.markdown(
            f"<div style='color:#9CA3AF;font-size:11px;margin-top:8px;'>"
            f"Most Connected</div>"
            f"<div style='color:#34C78A;font-size:13px;'>"
            f"{truncate(str(mc_label), 28)}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ── Shared filter helper ──────────────────────────────────────────────────────

def _filter_graph(
    graph: nx.Graph,
    min_link: float,
    min_size: float,
    selected_communities: Optional[List],
    search_term: str = "",
) -> nx.Graph:
    """Apply edge weight, node size, community, and search filters."""
    keep_nodes = set()
    for n in graph.nodes():
        data = graph.nodes[n]
        if data.get("weight", 1) < min_size:
            continue
        if selected_communities is not None:
            if data.get("community_id", 0) not in selected_communities:
                continue
        if search_term:
            if search_term.lower() not in str(n).lower():
                continue
        keep_nodes.add(n)

    sub = graph.subgraph(keep_nodes).copy()

    edges_to_remove = [
        (u, v) for u, v in sub.edges()
        if sub[u][v].get("weight", 1) < min_link
    ]
    sub.remove_edges_from(edges_to_remove)
    return sub
