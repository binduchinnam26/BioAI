"""
Knowledge graph renderer — VOSviewer-faithful directed graph with entity-type
coloring, curved arrows, gap highlighting, left control panel, right detail
panel, and relationship evidence table.
"""

import hashlib as _hashlib
import json
import os as _os
import re
from typing import Any, Dict, List, Optional

import networkx as nx

# Own-file version — uses mtime so it refreshes every time the file is saved,
# guaranteeing the Streamlit session_state cache is busted on every deploy.
try:
    _KG_VERSION = "kgmt" + str(int(_os.path.getmtime(__file__)))
except Exception:
    try:
        with open(__file__, "rb") as _f:
            _KG_VERSION = "kg" + _hashlib.md5(_f.read()).hexdigest()[:10]
    except Exception:
        _KG_VERSION = "kg1"

from config import (
    CANVAS_BG,
    ENTITY_TYPE_COLORS,
    COMMUNITY_COLORS,
    NODE_SIZE_MIN,
    NODE_SIZE_MAX,
    EDGE_WIDTH_MIN,
    EDGE_WIDTH_MAX,
    COLOR_SURFACE_ELEVATED,
    COLOR_SURFACE,
    COLOR_TEXT_SECONDARY,
    COLOR_PRIMARY,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
)
from utils.helpers import (
    lighten_hex,
    hex_to_rgba,
    scale_node_size,
    scale_edge_width,
    truncate,
    percentile,
)
from visualization.network_viz import (
    get_physics_options,
    _TOOLTIP_STYLE,
    _wrap_tooltip,
    _STABILIZE_JS,
    _CONTROLS_JS,
    _post_process_html,
    _default_edge_tooltip,
    _label_font,
    _VIZ_VERSION,
)

# ── Pulse animation for gap nodes ─────────────────────────────────────────────

_PULSE_CSS = """
<style>
@keyframes pulse {
  0%   { border-color: #FBBF24; box-shadow: 0 0 0 0 rgba(251,191,36,0.6); }
  70%  { border-color: #FBBF24; box-shadow: 0 0 0 8px rgba(251,191,36,0); }
  100% { border-color: #FBBF24; box-shadow: 0 0 0 0 rgba(251,191,36,0); }
}
.gap-node { animation: pulse 1.8s infinite; }
/* Remove vis.js default tooltip shell (white border + beige bg + nowrap) */
.vis-tooltip {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  border-radius: 0 !important;
  white-space: normal !important;
  max-width: 300px !important;
}
</style>
"""

# KG-specific stabilise: disable physics only — NO fit() calls.
# _STABILIZE_JS (from network_viz) also calls fit() at 300/800/1500/2500ms
# which overrides the custom bounding-box zoom in _KG_FIT_JS.  We inject
# this script instead of _STABILIZE_JS so the KG fit is never clobbered.
_KG_STABILIZE_JS = """
<script>
(function() {
  network.once('stabilizationIterationsDone', function() {
    network.setOptions({ physics: { enabled: false } });
    network.stopSimulation();
  });
})();
</script>
"""

# Convert string titles → DOM elements AFTER stabilization completes.
# vis.js v7+ uses textContent (not innerHTML) for string titles, so any HTML
# in a string title is displayed as raw escaped text. DOM element titles are
# appended directly (as nodes), so they render correctly as styled HTML.
# We bulk-convert AFTER stabilizationIterationsDone (physics disabled) so
# DataSet.update() does NOT trigger O(n²) avoidOverlap recalculation.
_KG_TOOLTIP_FIX_JS = """
<script>
(function() {
  function _convertTitles() {
    if (typeof network === 'undefined' || !network.body) return;
    var nodeUpdates = [];
    network.body.data.nodes.forEach(function(n) {
      if (typeof n.title === 'string' && n.title.length > 0) {
        var d = document.createElement('div');
        d.innerHTML = n.title;
        nodeUpdates.push({ id: n.id, title: d });
      }
    });
    if (nodeUpdates.length > 0) network.body.data.nodes.update(nodeUpdates);

    var edgeUpdates = [];
    network.body.data.edges.forEach(function(e) {
      if (typeof e.title === 'string' && e.title.length > 0) {
        var d = document.createElement('div');
        d.innerHTML = e.title;
        edgeUpdates.push({ id: e.id, title: d });
      }
    });
    if (edgeUpdates.length > 0) network.body.data.edges.update(edgeUpdates);
  }
  // Run after physics stops — safe to bulk-update DataSet with no avoidOverlap penalty.
  // Small delay so _applyNodeSizes (also on stabilizationIterationsDone) runs first.
  network.once('stabilizationIterationsDone', function() {
    setTimeout(_convertTitles, 200);
  });
  setTimeout(_convertTitles, 6000); // fallback if event never fires
})();
</script>
"""

_GAP_HIGHLIGHT_JS = """
<script>
(function() {
  var gapNodes = __GAP_NODES_JSON__;
  if (!gapNodes || gapNodes.length === 0) return;
  network.on('afterDrawing', function(ctx) {
    var now = Date.now();
    gapNodes.forEach(function(nodeId) {
      if (!network.body.nodes[nodeId]) return;
      var pos = network.getPositions([nodeId])[nodeId];
      // Draw pulsing yellow ring via canvas arc
      var phase = (Math.sin(now / 400) + 1) / 2;
      ctx.save();
      ctx.strokeStyle = 'rgba(251,191,36,' + (0.4 + 0.6 * phase) + ')';
      ctx.lineWidth = 3 + 2 * phase;
      ctx.beginPath();
      var size = (network.body.nodes[nodeId].options.size || 20) + 6 + 4 * phase;
      ctx.arc(pos.x, pos.y, size, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.restore();
    });
    // No requestAnimationFrame here — animation is driven by the
    // setInterval below at ~10 fps.  The previous pattern of calling
    // requestAnimationFrame(network.redraw) inside afterDrawing created
    // an infinite 60 fps redraw loop that blocked the main thread on
    // large graphs and triggered the browser "Page Unresponsive" dialog.
  });
  // Drive the pulse at ~10 fps — smooth enough to be visible, cheap
  // enough not to block the UI.
  setInterval(function() { network.redraw(); }, 100);
})();
</script>
"""

_KG_HIGHLIGHT_JS = """
<script>
var allNodes = network.body.data.nodes;
var allEdges = network.body.data.edges;

network.on('click', function(params) {
  if (params.nodes.length === 0) {
    // Batch into a single DataSet.update() call — calling update() once
    // per node inside forEach caused N separate redraws and froze the tab
    // on large graphs.
    allNodes.update(allNodes.getIds().map(function(id) { return {id:id,opacity:1.0}; }));
    allEdges.update(allEdges.getIds().map(function(id) { return {id:id,opacity:1.0}; }));
    return;
  }
  var clickedId = params.nodes[0];
  var connected = new Set(network.getConnectedNodes(clickedId));
  connected.add(clickedId);
  var nodeUp = [];
  allNodes.getIds().forEach(function(id) {
    nodeUp.push({ id: id, opacity: connected.has(id) ? 1.0 : 0.15 });
  });
  allNodes.update(nodeUp);
  var connEdges = new Set(network.getConnectedEdges(clickedId));
  var edgeUp = [];
  allEdges.getIds().forEach(function(id) {
    edgeUp.push({ id: id, opacity: connEdges.has(id) ? 1.0 : 0.05 });
  });
  allEdges.update(edgeUp);
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


# Robust post-stabilisation fit for the KG.
# Retries at three increasing delays so at least one call lands after
_KG_FIT_JS = """
<script>
(function() {
  network.once('stabilizationIterationsDone', function() {
    // Fire at 500ms; retry at 1500ms and 3000ms if not done yet.
    // No canvas-dimension guard — fit() works regardless of offsetWidth.
    var _done = false;
    [500, 1500, 3000].forEach(function(ms) {
      setTimeout(function() {
        if (_done) return;
        _done = true;
        var positions = network.getPositions();
        var ids = Object.keys(positions);
        var coreIds;
        if (ids.length >= 4) {
          // Centroid of all nodes
          var cx0 = 0, cy0 = 0;
          ids.forEach(function(id) { cx0 += positions[id].x; cy0 += positions[id].y; });
          cx0 /= ids.length; cy0 /= ids.length;
          // Distance of every node from centroid
          var dists = ids.map(function(id) {
            var dx = positions[id].x - cx0, dy = positions[id].y - cy0;
            return { id: id, d: Math.sqrt(dx * dx + dy * dy) };
          });
          // Keep nodes within mean + 1.5*stddev (cuts isolated outlier components)
          var mean = dists.reduce(function(s, x) { return s + x.d; }, 0) / dists.length;
          var vari = dists.reduce(function(s, x) { return s + (x.d - mean) * (x.d - mean); }, 0) / dists.length;
          var thr = mean + 1.5 * Math.sqrt(vari);
          var core = dists.filter(function(x) { return x.d <= thr; });
          if (core.length < 3) {
            dists.sort(function(a, b) { return a.d - b.d; });
            core = dists.slice(0, Math.ceil(dists.length * 0.6));
          }
          coreIds = core.map(function(x) { return x.id; });
        }
        // Step 1: instant fit (no animation) so we can read the resulting scale
        if (coreIds && coreIds.length > 0) {
          network.fit({ nodes: coreIds, animation: false });
        } else {
          network.fit({ animation: false });
        }
        // Step 2: single smooth zoom-in — one animation, no flicker
        setTimeout(function() {
          network.moveTo({
            scale: network.getScale() * 1.8,
            animation: { duration: 700, easingFunction: 'easeInOutQuad' }
          });
        }, 50);
      }, ms);
    });
  });
})();
</script>
"""


def _post_process_kg_html(
    html: str, gap_nodes: Optional[List] = None
) -> str:
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
    gap_json = json.dumps(
        [str(n) for n in (gap_nodes or [])]
    )
    gap_js = _GAP_HIGHLIGHT_JS.replace("__GAP_NODES_JSON__", gap_json)
    html = html.replace(
        "</body>",
        _PULSE_CSS + _KG_STABILIZE_JS + _KG_HIGHLIGHT_JS + gap_js
        + _CONTROLS_JS + _KG_FIT_JS + _KG_TOOLTIP_FIX_JS + "</body>",
    )
    return html


# ── Entity type legend ────────────────────────────────────────────────────────

def _render_entity_legend():
    import streamlit as st
    st.markdown(
        "<div style='font-size:11px;color:#9CA3AF;margin-top:16px;"
        "font-weight:600;'>ENTITY TYPES</div>",
        unsafe_allow_html=True,
    )
    for etype, color in ENTITY_TYPE_COLORS.items():
        st.markdown(
            f"<div style='margin:3px 0;font-size:12px;color:#D1D5DB;'>"
            f"<span style='color:{color};font-size:14px;'>●</span> "
            f"{etype.replace('_', ' ').title()}</div>",
            unsafe_allow_html=True,
        )


# ── Main render function ──────────────────────────────────────────────────────

def render_knowledge_graph(
    graph: nx.MultiDiGraph,
    papers_df=None,
    highlight_gaps: bool = False,
    gap_pairs: Optional[List] = None,
    key_prefix: str = "kg",
):
    """
    Full-width directed knowledge graph renderer.

    Layout: 280px left control panel + remaining canvas + 320px right detail
    panel (slides in on double-click via session_state).
    """
    import streamlit as st
    st.markdown(
        "### Biomedical Knowledge Graph\n"
        "<span style='color:#9CA3AF;font-size:13px;'>"
        "Node size = evidence strength &nbsp;|&nbsp; "
        "Color = entity type &nbsp;|&nbsp; "
        "Arrows = relationship direction"
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#1F2937;margin:8px 0;'>",
                unsafe_allow_html=True)

    if graph is None or graph.number_of_nodes() == 0:
        st.info("No knowledge graph available. Run the pipeline first.")
        return

    # ── Left control panel + graph canvas ────────────────────────────────────
    col_ctrl, col_graph = st.columns([1, 4])

    with col_ctrl:
        st.markdown(
            "<div style='font-size:11px;color:#9CA3AF;font-weight:600;"
            "margin-bottom:8px;'>ENTITY TYPES</div>",
            unsafe_allow_html=True,
        )

        all_etypes = sorted(
            set(data.get("entity_type", "UNKNOWN")
                for _, data in graph.nodes(data=True))
        )
        selected_etypes = []
        for etype in all_etypes:
            color = ENTITY_TYPE_COLORS.get(etype, "#9CA3AF")
            checked = st.checkbox(
                f"● {etype.replace('_', ' ').title()}",
                value=True,
                key=f"{key_prefix}_etype_{etype}",
            )
            if checked:
                selected_etypes.append(etype)

        st.markdown(
            "<div style='font-size:11px;color:#9CA3AF;font-weight:600;"
            "margin-top:14px;margin-bottom:6px;'>RELATIONSHIP TYPES</div>",
            unsafe_allow_html=True,
        )
        all_rel_types = sorted(
            set(
                data.get("relationship_type", "unknown")
                for _, _, data in graph.edges(data=True)
            )
        )
        selected_rels = st.multiselect(
            "Filter relationships",
            options=all_rel_types,
            default=all_rel_types,
            key=f"{key_prefix}_rels",
            label_visibility="collapsed",
        )

        st.markdown(
            "<div style='font-size:11px;color:#9CA3AF;font-weight:600;"
            "margin-top:14px;margin-bottom:6px;'>DEPTH</div>",
            unsafe_allow_html=True,
        )
        depth = st.slider(
            "Neighbourhood Depth",
            min_value=1,
            max_value=3,
            value=2,
            key=f"{key_prefix}_depth",
            label_visibility="collapsed",
        )

        entity_search = st.text_input(
            "Search entities…",
            key=f"{key_prefix}_entity_search",
            placeholder="Search entities…",
            label_visibility="collapsed",
        )

        gap_highlight = st.checkbox(
            "Highlight Research Gaps",
            value=highlight_gaps,
            key=f"{key_prefix}_gap_toggle",
        )

        evidence_threshold = st.slider(
            "Min Evidence Papers",
            min_value=1,
            max_value=10,
            value=1,
            key=f"{key_prefix}_evidence_thresh",
        )

        st.markdown("<div style='margin-top:14px;'>", unsafe_allow_html=True)
        if st.button("Export PNG", key=f"{key_prefix}_export_png"):
            st.info("Right-click the graph → Save image as…")
        if st.button("Export JSON", key=f"{key_prefix}_export_json"):
            from pipeline.knowledge_graph import KnowledgeGraph
            kg_obj = KnowledgeGraph()
            kg_obj.graph = graph
            data = kg_obj.export_to_json()
            st.download_button(
                "Download JSON",
                data=json.dumps(data, indent=2),
                file_name="knowledge_graph.json",
                mime="application/json",
                key=f"{key_prefix}_dl_json",
            )
        st.markdown("</div>", unsafe_allow_html=True)

        _render_entity_legend()

    with col_graph:
        # Apply filters
        filtered = _filter_kg(
            graph,
            selected_etypes,
            selected_rels,
            entity_search,
            evidence_threshold,
        )

        gap_node_list = []
        if gap_highlight and gap_pairs:
            for pair in gap_pairs:
                if isinstance(pair, dict):
                    gap_node_list.extend([pair.get("concept_a"), pair.get("concept_b")])
                elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    gap_node_list.extend([pair[0], pair[1]])
            gap_node_list = [n for n in gap_node_list if n and filtered.has_node(n)]

        cache_key = (
            f"_kg_html_{_VIZ_VERSION}_{_KG_VERSION}_{key_prefix}_"
            f"{','.join(sorted(selected_etypes))}_"
            f"{','.join(sorted(selected_rels))}_{entity_search}_"
            f"{evidence_threshold}_{gap_highlight}"
        )
        if cache_key not in st.session_state:
            html = _build_kg_html(filtered, gap_node_list)
            st.session_state[cache_key] = html
        else:
            html = st.session_state[cache_key]

        st.components.v1.html(html, height=870, scrolling=False)

    # ── Right detail panel (triggered by double-click) ────────────────────────
    selected_entity = st.session_state.get(f"{key_prefix}_selected_entity")
    if selected_entity and graph.has_node(selected_entity):
        _render_entity_detail_panel(graph, selected_entity, papers_df, key_prefix)

    # ── Relationship evidence table ───────────────────────────────────────────
    st.markdown(
        "<hr style='border-color:#1F2937;margin:24px 0 12px;'>",
        unsafe_allow_html=True,
    )
    _render_relationship_table(graph, key_prefix)


def _build_kg_html(
    graph: nx.MultiDiGraph,
    gap_nodes: Optional[List] = None,
) -> str:
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise ImportError("pyvis is not installed. Run: pip install pyvis") from exc

    weights = {n: graph.nodes[n].get("weight", 1) for n in graph.nodes()}

    edge_weights = {}
    for u, v, data in graph.edges(data=True):
        key = (u, v)
        edge_weights[key] = max(edge_weights.get(key, 0),
                                data.get("weight", 1))
    ew_min = min(edge_weights.values(), default=1)
    ew_max = max(edge_weights.values(), default=1)

    # Size by total degree (in+out connections) — mirrors VOSviewer where hub
    # nodes (most connections) render largest.  Weight alone collapses to
    # size_min when all nodes share the same weight (common in KGs), so degree
    # is used as the primary sizing metric for natural visual hierarchy.
    degrees = {n: graph.in_degree(n) + graph.out_degree(n) for n in graph.nodes()}
    d_min = min(degrees.values(), default=1)
    d_max = max(degrees.values(), default=1)
    node_sizes = {
        n: scale_node_size(degrees.get(n, 1), d_min, d_max, 10, 40)
        for n in graph.nodes()
    }

    net = Network(
        height="850px",
        width="100%",
        directed=True,
        bgcolor=CANVAS_BG,
        font_color="#000000",      # black labels — matches keyword network
    )
    net.toggle_physics(True)

    for node in graph.nodes():
        data = graph.nodes[node]
        etype = data.get("entity_type", "UNKNOWN")
        fill_hex = ENTITY_TYPE_COLORS.get(etype, "#9CA3AF")
        label = truncate(str(node), 20)
        tooltip = _kg_node_tooltip(node, data, graph)

        # Font must be large in vis.js units so it stays above vis.js's
        # ~4px hide-threshold at typical zoom levels (0.2–0.4 after fit).
        # At zoom=0.25: font=40 → 10px screen (readable), font=14 → 3.5px (hidden).
        node_size_val = node_sizes.get(node, 10)
        font_px = max(14, min(20, int(node_size_val * 0.65)))
        font = {
            "size": font_px,
            "color": "#000000",
            "face": "arial",
            "strokeWidth": 2,
            "strokeColor": "#FFFFFF",
        }

        # Full color dict — matches keyword network node color structure
        node_color = {
            "background": hex_to_rgba(fill_hex, 0.78),
            "border":     hex_to_rgba(fill_hex, 0.90),
            "highlight": {"background": hex_to_rgba(fill_hex, 0.95), "border": "#000000"},
            "hover":     {"background": hex_to_rgba(fill_hex, 0.88), "border": "#333333"},
        }

        net.add_node(
            str(node),
            label=label,
            title=tooltip,
            size=node_size_val,        # explicit size, not value=
            shape="dot",
            color=node_color,
            font=font,
            borderWidth=0,
            borderWidthSelected=0,
        )

    seen_edges = set()
    for u, v, key, data in graph.edges(data=True, keys=True):
        edge_key = (str(u), str(v))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        etype_u = graph.nodes[u].get("entity_type", "UNKNOWN")
        fill_hex = ENTITY_TYPE_COLORS.get(etype_u, "#9CA3AF")
        w = edge_weights.get((u, v), 1)
        width = scale_edge_width(w, ew_min, ew_max, EDGE_WIDTH_MIN, EDGE_WIDTH_MAX)
        # Full color dict with hover/highlight states — matches keyword network
        edge_color = {
            "color":     hex_to_rgba(fill_hex, 0.45),
            "highlight": hex_to_rgba(fill_hex, 0.90),
            "hover":     hex_to_rgba(fill_hex, 0.70),
            "inherit":   False,
        }
        rel_type = data.get("relationship_type", "")
        pmids = data.get("evidence_pmids", [])
        pmid_str = ", ".join(str(p) for p in pmids[:3])
        tooltip_content = (
            f'<div style="font-weight:600;color:#FFF;margin-bottom:4px;">'
            f'{u} → {v}</div>'
            f'<div style="color:#9CA3AF;margin-bottom:2px;">{rel_type}</div>'
            f'<div>Evidence: <b style="color:#4E9AF1;">{w}</b> papers</div>'
            + (f'<div style="color:#9CA3AF;font-size:11px;margin-top:4px;">'
               f'PMIDs: {pmid_str}</div>' if pmid_str else "")
        )
        net.add_edge(
            str(u), str(v),
            width=width,
            color=edge_color,
            title=_wrap_tooltip(tooltip_content),
            arrows="to",
            smooth={"type": "curvedCW", "roundness": 0.2},
            arrowStrikethrough=False,
        )

    # barnesHut with keyword-network settings: strong repulsion spreads
    # clusters wide across the canvas; centralGravity=0 means topology
    # alone drives placement (no ring force), matching the keyword network
    # arrangement while keeping all directed edges, colours and tooltips.
    opts = {
        "physics": {
            "enabled": True,
            "solver": "barnesHut",
            "barnesHut": {
                "gravitationalConstant": -55000,
                "centralGravity": 0.0,
                "springLength": 220,
                "springConstant": 0.05,
                "damping": 0.10,
                "avoidOverlap": 1.0,
            },
            "maxVelocity": 100,
            "minVelocity": 0.10,
            "stabilization": {
                "enabled": True,
                "iterations": 2000,
                "updateInterval": 10,
                "fit": False,
            },
            "timestep": 0.20,
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 150,
            "hideEdgesOnDrag": True,
            "hideEdgesOnZoom": False,
            "multiselect": True,
            "navigationButtons": False,
            "keyboard": {"enabled": False},
            "zoomView": True,
        },
        "nodes": {
            "chosen": True,
            "physics": True,
            "font": {
                "size": 14,
                "color": "#000000",
                "strokeWidth": 2,
                "strokeColor": "#FFFFFF",
            },
            "borderWidth": 0,
            "borderWidthSelected": 0,
        },
        "edges": {
            "chosen": True,
            "physics": True,
            "hoverWidth": 2.5,
            "selectionWidth": 3.0,
            "smooth": {"type": "curvedCW", "roundness": 0.2},
        },
    }
    net.set_options(json.dumps(opts))
    html = net.generate_html(notebook=False)
    html = _post_process_kg_html(html, gap_nodes)

    # Directly override node sizes AND font sizes via vis.js DataSet API.
    # Font must be large (40-60 vis.js units) so it exceeds vis.js's ~4px
    # hide-threshold at typical post-fit zoom levels of 0.2–0.4.
    _sizes_json = json.dumps({str(n): round(node_sizes[n]) for n in graph.nodes()})
    _node_size_js = f"""<script>
(function() {{
  var _sz = {_sizes_json};
  function _applyNodeSizes() {{
    if (typeof network === 'undefined' || !network.body) return;
    var updates = Object.keys(_sz).map(function(id) {{
      var s = _sz[id];
      var f = Math.max(10, Math.min(24, Math.round(s * 0.65)));
      return {{
        id: id,
        size: s,
        borderWidth: 0,
        borderWidthSelected: 0,
        font: {{ size: f, color: '#000000', face: 'arial', strokeWidth: 2, strokeColor: '#FFFFFF' }}
      }};
    }});
    network.body.data.nodes.update(updates);
  }}
  // Apply ONLY after stabilization completes — calling DataSet.update()
  // while avoidOverlap physics is running triggers O(n²) force recalculation
  // for every updated node, blocking the main thread and causing "Page Unresponsive".
  network.once('stabilizationIterationsDone', _applyNodeSizes);
  setTimeout(_applyNodeSizes, 5000); // fallback if event never fires
}})();
</script>"""
    html = html.replace("</body>", _node_size_js + "\n</body>")
    return html


def _kg_node_tooltip(node: str, data: Dict, graph: nx.MultiDiGraph) -> str:
    etype = data.get("entity_type", "UNKNOWN")
    color = ENTITY_TYPE_COLORS.get(etype, "#9CA3AF")
    umls = data.get("umls_id") or "—"
    paper_count = data.get("paper_count", data.get("weight", 0))

    # Relationship type counts
    out_rels = {}
    for _, tgt, edata in graph.out_edges(node, data=True):
        rt = edata.get("relationship_type", "unknown")
        out_rels[rt] = out_rels.get(rt, 0) + 1
    rel_str = " | ".join(
        f'<span style="color:#4E9AF1;">{rt}</span> → {cnt}'
        for rt, cnt in sorted(out_rels.items(), key=lambda x: x[1], reverse=True)[:3]
    )

    # Top connected by edge weight
    degree_map = {}
    for _, nb, edata in graph.out_edges(node, data=True):
        degree_map[nb] = degree_map.get(nb, 0) + edata.get("weight", 1)
    for src, _, edata in graph.in_edges(node, data=True):
        degree_map[src] = degree_map.get(src, 0) + edata.get("weight", 1)
    top_connected = sorted(degree_map, key=degree_map.get, reverse=True)[:3]
    connected_html = "".join(
        f"<div style='color:#D1D5DB;margin-left:6px;'>• {truncate(n, 22)}</div>"
        for n in top_connected
    )

    content = (
        f'<div style="font-size:14px;font-weight:700;color:#FFFFFF;margin-bottom:4px;'
        f'word-wrap:break-word;overflow-wrap:break-word;white-space:normal;">'
        f'{node}</div>'
        f'<div style="margin-bottom:6px;">'
        f'<span style="background:{color};color:#000;font-size:10px;'
        f'padding:2px 6px;border-radius:4px;">'
        f'{etype.replace("_", " ").title()}</span></div>'
        f'<div style="color:#9CA3AF;font-family:monospace;font-size:11px;margin-bottom:4px;">'
        f'UMLS: {umls}</div>'
        f'<div style="margin-bottom:4px;">'
        f'Found in <b style="color:#4E9AF1;">{paper_count}</b> papers</div>'
        + (f'<div style="margin-bottom:4px;font-size:11px;">{rel_str}</div>'
           if rel_str else "")
        + (f'<div style="margin-top:8px;padding-top:8px;'
           f'border-top:1px solid #2D3A55;color:#9CA3AF;font-size:11px;">'
           f'Top connected:<br>{connected_html}</div>'
           if connected_html else "")
    )
    return _wrap_tooltip(content)


# ── Entity detail panel ───────────────────────────────────────────────────────

def _render_entity_detail_panel(
    graph: nx.MultiDiGraph,
    entity_name: str,
    papers_df,
    key_prefix: str,
):
    import streamlit as st
    data = graph.nodes.get(entity_name, {})
    etype = data.get("entity_type", "UNKNOWN")
    color = ENTITY_TYPE_COLORS.get(etype, "#9CA3AF")
    umls = data.get("umls_id") or "—"

    with st.expander(f"Entity Detail: {entity_name}", expanded=True):
        st.markdown(
            f'<span style="background:{color};color:#000;font-size:12px;'
            f'padding:2px 8px;border-radius:4px;">{etype}</span> '
            f'<code style="color:#9CA3AF;font-size:11px;">{umls}</code>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Papers:** {data.get('paper_count', data.get('weight', 0))}"
        )

        # Outgoing relationships
        out_rels = []
        for _, tgt, edata in graph.out_edges(entity_name, data=True):
            out_rels.append(
                {
                    "Target": tgt,
                    "Relationship": edata.get("relationship_type", "—"),
                    "Confidence": round(edata.get("confidence_score", 0.0), 3),
                    "Evidence Papers": len(edata.get("evidence_pmids", [])),
                }
            )
        if out_rels:
            import pandas as pd
            st.markdown("**Outgoing relationships:**")
            st.dataframe(pd.DataFrame(out_rels), use_container_width=True,
                         hide_index=True)

        if st.button("Close", key=f"{key_prefix}_close_detail"):
            st.session_state[f"{key_prefix}_selected_entity"] = None
            st.rerun()


# ── Relationship evidence table ───────────────────────────────────────────────

def _render_relationship_table(
    graph: nx.MultiDiGraph,
    key_prefix: str,
):
    import streamlit as st
    st.markdown(
        "#### Relationship Evidence Table",
        unsafe_allow_html=True,
    )

    rows = []
    for src, tgt, data in graph.edges(data=True):
        pmids = data.get("evidence_pmids", [])
        pmid_links = " ".join(
            f'<a href="https://pubmed.ncbi.nlm.nih.gov/{p}/" target="_blank"'
            f' style="color:#4E9AF1;">{p}</a>'
            for p in pmids[:3]
        )
        src_type = graph.nodes.get(src, {}).get("entity_type", "—")
        tgt_type = graph.nodes.get(tgt, {}).get("entity_type", "—")
        rows.append(
            {
                "Source Entity": truncate(str(src), 30),
                "Source Type": src_type,
                "Relationship": data.get("relationship_type", "—"),
                "Target Entity": truncate(str(tgt), 30),
                "Target Type": tgt_type,
                "Confidence": round(data.get("confidence_score", 0.0), 3),
                "Evidence PMIDs": ", ".join(str(p) for p in pmids[:3]),
            }
        )

    if not rows:
        st.info("No relationship data available.")
        return

    import pandas as pd
    df = pd.DataFrame(rows)

    # Filtering controls
    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        src_type_filter = st.multiselect(
            "Source Type",
            options=sorted(df["Source Type"].unique()),
            default=sorted(df["Source Type"].unique()),
            key=f"{key_prefix}_table_src_type",
        )
    with cf2:
        tgt_type_filter = st.multiselect(
            "Target Type",
            options=sorted(df["Target Type"].unique()),
            default=sorted(df["Target Type"].unique()),
            key=f"{key_prefix}_table_tgt_type",
        )
    with cf3:
        rel_filter = st.multiselect(
            "Relationship Type",
            options=sorted(df["Relationship"].unique()),
            default=sorted(df["Relationship"].unique()),
            key=f"{key_prefix}_table_rel",
        )

    df_filtered = df[
        df["Source Type"].isin(src_type_filter)
        & df["Target Type"].isin(tgt_type_filter)
        & df["Relationship"].isin(rel_filter)
    ]

    sort_col = st.selectbox(
        "Sort by",
        options=["Confidence", "Source Entity", "Relationship"],
        key=f"{key_prefix}_table_sort",
        label_visibility="collapsed",
    )
    df_filtered = df_filtered.sort_values(sort_col, ascending=False)

    # Pagination
    page_size = 25
    total_rows = len(df_filtered)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    page = st.number_input(
        f"Page (1–{total_pages})",
        min_value=1,
        max_value=total_pages,
        value=1,
        key=f"{key_prefix}_table_page",
    )
    start = (page - 1) * page_size
    end = start + page_size

    st.dataframe(
        df_filtered.iloc[start:end].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Showing {start + 1}–{min(end, total_rows)} of {total_rows} relationships")

    # CSV export
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        "Export CSV",
        data=csv,
        file_name="relationships.csv",
        mime="text/csv",
        key=f"{key_prefix}_export_csv",
    )


# ── KG filter helper ──────────────────────────────────────────────────────────

def _filter_kg(
    graph: nx.MultiDiGraph,
    selected_etypes: List[str],
    selected_rels: List[str],
    search_term: str,
    min_evidence: int,
) -> nx.MultiDiGraph:
    keep_nodes = set()
    for n, data in graph.nodes(data=True):
        etype = data.get("entity_type", "UNKNOWN")
        if etype not in selected_etypes:
            continue
        if search_term and search_term.lower() not in str(n).lower():
            continue
        keep_nodes.add(n)

    sub = graph.subgraph(keep_nodes).copy()

    edges_to_remove = []
    for u, v, key, data in sub.edges(data=True, keys=True):
        if data.get("relationship_type", "") not in selected_rels:
            edges_to_remove.append((u, v, key))
        elif len(data.get("evidence_pmids", [])) < min_evidence \
                and data.get("weight", 0) < min_evidence:
            edges_to_remove.append((u, v, key))
    for u, v, k in edges_to_remove:
        if sub.has_edge(u, v, k):
            sub.remove_edge(u, v, k)

    return sub
