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

import hashlib as _hashlib
try:
    with open(__file__, "rb") as _f:
        _VIZ_VERSION = "v" + _hashlib.md5(_f.read()).hexdigest()[:10]
except Exception:
    _VIZ_VERSION = "v1"

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

def get_physics_options(node_count: int, network_type: str = "default") -> Dict:
    if node_count < 50:
        grav = -3000
        spring = 200
        overlap = 0.8
    elif node_count > 500:
        grav = -8000
        spring = 100
        overlap = 1.0
    else:
        grav = -5000
        spring = 150
        overlap = 0.9

    # Co-authorship network: use forceAtlas2Based solver instead of barnesHut.
    # barnesHut applies global pairwise repulsion between ALL nodes — this pushes
    # intra-cluster nodes into polygon rings regardless of avoidOverlap settings.
    # forceAtlas2Based uses hub-weighted repulsion: high-degree nodes repel more,
    # which naturally separates communities while keeping cluster members in
    # tight organic blobs (not rings). centralGravity=0.01 keeps all clusters
    # in one compact mass without scattering them across the canvas.
    if network_type == "coauthorship":
        return {
            "layout": {
                "improvedLayout": False,
            },
            "physics": {
                "enabled": True,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    # springLength=1 collapses connected nodes toward the same
                    # point; avoidOverlap=1.0 then stops them at exact touching
                    # distance (sum of radii). This creates a unique BLOB
                    # equilibrium — nodes pack at contact — rather than the ring
                    # equilibrium produced by large springLength where nodes
                    # settle equidistant around a hub. Since node sizes vary
                    # (paper-count scaled 18-85), touching distances differ per
                    # pair, naturally breaking symmetry.
                    "gravitationalConstant": -20,
                    "centralGravity": 0.02,
                    "springLength": 1,
                    "springConstant": 0.08,
                    "damping": 0.4,
                    "avoidOverlap": 1.0,
                },
                "maxVelocity": 80,
                "minVelocity": 0.3,
                "stabilization": {
                    "enabled": True,
                    "iterations": 3000,
                    "updateInterval": 25,
                    "fit": True,
                },
                "timestep": 0.25,
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
                "shadow": False,
                "font": {
                    "size": 20,
                    "color": "#000000",
                    "strokeWidth": 2,
                    "strokeColor": "#FFFFFF",
                    "vadjust": 0,
                },
            },
            "edges": {
                "chosen": True,
                "physics": True,
                "hoverWidth": 2.5,
                "selectionWidth": 3.0,
                "smooth": {"type": "continuous", "roundness": 0.1},
            },
        }

    # Keyword networks: zero centralGravity + balanced spring/repulsion.
    # centralGravity=0 prevents circular ring. Stronger repulsion + longer softer
    # springs spread clusters wide across canvas without intra-cluster overlap.
    elif network_type == "keyword":
        grav = -55000        # strong repulsion separates clusters across canvas
        central_grav = 0.0   # zero — topology drives placement, no ring force
        spring = 220         # longer: gives nodes within each cluster room
        spring_const = 0.05  # soft: cluster structure without squashing nodes
        damping = 0.10
        overlap = 1.0
        iterations = 6000
        timestep = 0.20
        max_vel = 100
        min_vel = 0.10
    else:
        central_grav = 0.15
        spring_const = 0.04
        damping = 0.12
        iterations = 2000
        timestep = 0.35
        max_vel = 60
        min_vel = 0.3

    return {
        "physics": {
            "enabled": True,
            "barnesHut": {
                "gravitationalConstant": grav,
                "centralGravity": central_grav,
                "springLength": spring,
                "springConstant": spring_const,
                "damping": damping,
                "avoidOverlap": overlap,
            },
            "maxVelocity": max_vel,
            "minVelocity": min_vel,
            "stabilization": {
                "enabled": True,
                "iterations": iterations,
                "updateInterval": 25,
                "fit": True,
            },
            "timestep": timestep,
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
            "shadow": False,
            "font": {
                "size": 22 if network_type == "keyword" else 14,
                "color": "#000000",
                "strokeWidth": 2,
                "strokeColor": "#FFFFFF",
                "vadjust": -40 if network_type == "keyword" else 0,
            },
        },
        "edges": {
            "chosen": True,
            "physics": True,
            "hoverWidth": 2.5,
            "selectionWidth": 3.0,
            "smooth": {"type": "continuous", "roundness": 0.1},
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
    size_min: Optional[float] = None,
    size_max: Optional[float] = None,
) -> Dict[Any, float]:
    if size_min is None:
        size_min = NODE_SIZE_MIN
    if size_max is None:
        size_max = NODE_SIZE_MAX
    weights = {n: graph.nodes[n].get(weight_attr, 1) for n in graph.nodes()}
    w_min = min(weights.values(), default=1)
    w_max = max(weights.values(), default=1)
    return {
        n: scale_node_size(w, w_min, w_max, size_min, size_max)
        for n, w in weights.items()
    }


def _compute_edge_widths(
    graph: nx.Graph,
    width_min: Optional[float] = None,
    width_max: Optional[float] = None,
) -> Dict[Tuple, float]:
    if width_min is None:
        width_min = EDGE_WIDTH_MIN
    if width_max is None:
        width_max = EDGE_WIDTH_MAX
    edge_weights = {
        (u, v): graph[u][v].get("weight", 1) for u, v in graph.edges()
    }
    if not edge_weights:
        return {}
    w_min = min(edge_weights.values())
    w_max = max(edge_weights.values())
    return {
        edge: scale_edge_width(w, w_min, w_max, width_min, width_max)
        for edge, w in edge_weights.items()
    }


# ── Label visibility ──────────────────────────────────────────────────────────

def _label_font(weight: float, p50: float, p75: float) -> Dict:
    if weight >= p75:
        return {"size": 16, "color": "#000000", "face": "arial", "strokeWidth": 3, "strokeColor": "#FFFFFF"}
    if weight >= p50:
        return {"size": 13, "color": "#000000", "face": "arial", "strokeWidth": 2, "strokeColor": "#FFFFFF"}
    return {"size": 11, "color": "#000000", "face": "arial", "strokeWidth": 1, "strokeColor": "#FFFFFF"}


# ── PyVis HTML post-processing ────────────────────────────────────────────────

_STABILIZE_JS = """
<script>
(function() {
  network.once('stabilizationIterationsDone', function() {
    network.setOptions({ physics: { enabled: false } });
    [300, 800, 1500, 2500].forEach(function(ms) {
      setTimeout(function() {
        var el = document.getElementById('mynetwork');
        if (el && el.offsetWidth > 50 && el.offsetHeight > 50) {
          network.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
        }
      }, ms);
    });
  });
})();
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

_CONTROLS_JS = """
<script>
(function() {
  var style = document.createElement('style');
  style.textContent = `
    #vos-controls {
      position: absolute;
      top: 10px;
      right: 10px;
      display: flex;
      flex-direction: column;
      gap: 2px;
      z-index: 999;
    }
    #vos-controls button {
      width: 28px;
      height: 28px;
      background: #fff;
      border: 1px solid #ccc;
      border-radius: 3px;
      cursor: pointer;
      font-size: 16px;
      line-height: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0;
      color: #333;
      box-shadow: 0 1px 3px rgba(0,0,0,0.15);
    }
    #vos-controls button:hover { background: #f0f0f0; }
    #vos-controls .sep { height: 6px; }
  `;
  document.head.appendChild(style);

  var container = document.createElement('div');
  container.id = 'vos-controls';

  function makeBtn(label, title, fn) {
    var b = document.createElement('button');
    b.innerHTML = label;
    b.title = title;
    b.addEventListener('click', fn);
    return b;
  }

  container.appendChild(makeBtn('+', 'Zoom in', function() {
    var s = network.getScale();
    network.moveTo({ scale: s * 1.3, animation: { duration: 200 } });
  }));
  container.appendChild(makeBtn('−', 'Zoom out', function() {
    var s = network.getScale();
    network.moveTo({ scale: s / 1.3, animation: { duration: 200 } });
  }));
  var sep = document.createElement('div'); sep.className = 'sep';
  container.appendChild(sep);
  container.appendChild(makeBtn('⊡', 'Fit to screen', function() {
    network.fit({ animation: { duration: 400, easingFunction: 'easeInOutQuad' } });
  }));
  container.appendChild(makeBtn('📷', 'Save screenshot', function() {
    var canvas = document.querySelector('#mynetwork canvas');
    if (!canvas) return;
    var link = document.createElement('a');
    link.download = 'network.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
  }));

  var target = document.getElementById('mynetwork');
  if (target) {
    target.style.position = 'relative';
    target.appendChild(container);
  }
})();
</script>
"""

_LABEL_OVERLAP_JS = """
<script>
(function() {
  // VOSviewer-style label overlap avoidance.
  // Greedy algorithm: process nodes largest→smallest; keep the label if it
  // doesn't overlap any already-accepted label, otherwise hide it.
  // Hidden labels reappear on hover and re-evaluation runs on every zoom.
  var _origLabels  = {};
  var _hiddenSet   = new Set();
  var CHAR_W       = 0.55;   // approximate char-width / font-size ratio (Arial)
  var PAD          = 4;      // extra px padding around each label bbox

  function _bbox(nodeId) {
    var node = allNodes.get(nodeId);
    if (!node) return null;
    var lbl = _origLabels[nodeId];
    if (!lbl) return null;
    var fs    = (node.font && node.font.size) ? node.font.size : 14;
    var scale = network.getScale();
    var dp    = network.canvasToDOM(network.getPosition(nodeId));
    var w     = lbl.length * fs * CHAR_W * scale;
    var h     = fs * 1.3 * scale;
    return { l: dp.x - w/2 - PAD, r: dp.x + w/2 + PAD,
             t: dp.y - h/2 - PAD, b: dp.y + h/2 + PAD };
  }

  function _hit(a, b) {
    return !(a.r < b.l || a.l > b.r || a.b < b.t || a.t > b.b);
  }

  function _run() {
    // Snapshot original labels once
    allNodes.getIds().forEach(function(id) {
      if (!(id in _origLabels)) _origLabels[id] = allNodes.get(id).label || '';
    });

    // Sort largest node first (most important)
    var sorted = allNodes.getIds().map(function(id) {
      return { id: id, sz: allNodes.get(id).size || 10 };
    }).sort(function(a, b) { return b.sz - a.sz; });

    var kept    = [];
    var hidden  = new Set();
    var updates = [];

    sorted.forEach(function(item) {
      var id  = item.id;
      var lbl = _origLabels[id];
      if (!lbl) return;
      var box = _bbox(id);
      if (!box) return;

      if (kept.some(function(k) { return _hit(box, k); })) {
        hidden.add(id);
        if (allNodes.get(id).label !== '') updates.push({ id: id, label: '' });
      } else {
        kept.push(box);
        if (allNodes.get(id).label !== lbl) updates.push({ id: id, label: lbl });
      }
    });

    _hiddenSet = hidden;
    if (updates.length) allNodes.update(updates);
  }

  // Run after stabilisation + 600ms fit animation
  network.once('stabilizationIterationsDone', function() {
    // Keyword network: re-fit after _STABILIZE_JS's 600ms animation finishes.
    // Streamlit's iframe may not have its final dimensions on the first fit(),
    // so this backup call (700ms later, when the container is fully sized)
    // ensures the network fills the canvas properly.
    setTimeout(function() {
      network.fit({ animation: { duration: 400, easingFunction: 'easeInOutQuad' } });
    }, 700);
    // Label overlap runs after the backup fit settles (700 + 400 + 50 buffer)
    setTimeout(_run, 1200);
  });

  // Re-evaluate on zoom (zoom-in reveals more labels, zoom-out hides more)
  var _zt = null;
  network.on('zoom', function() {
    clearTimeout(_zt);
    _zt = setTimeout(_run, 200);
  });

  // ── Custom HTML tooltip ───────────────────────────────────────────────────
  // PyVis HTML-encodes the title string so vis.js shows raw escaped HTML as
  // plain text. Fix: hide vis.js built-in tooltip; create a custom div that
  // decodes HTML entities then sets innerHTML so it renders properly.

  // Suppress the vis.js built-in tooltip
  var _ttCss = document.createElement('style');
  _ttCss.textContent = '.vis-tooltip { display: none !important; }';
  document.head.appendChild(_ttCss);

  // Custom tooltip div inside the network container
  var _ttEl  = document.createElement('div');
  _ttEl.style.cssText = 'position:absolute;z-index:9999;pointer-events:none;display:none;max-width:280px;';
  var _netEl = document.getElementById('mynetwork');
  _netEl.style.position = 'relative';
  _netEl.appendChild(_ttEl);

  // Track real mouse position (relative to _netEl) so tooltip always
  // appears beside the cursor — never on top of the node itself.
  var _mouseX = 0, _mouseY = 0;
  _netEl.addEventListener('mousemove', function(e) {
    var rect = _netEl.getBoundingClientRect();
    _mouseX = e.clientX - rect.left;
    _mouseY = e.clientY - rect.top;
  });

  function _decodeHtml(s) {
    var ta = document.createElement('textarea');
    ta.innerHTML = s;
    return ta.value;
  }

  function _showTooltip(nodeId) {
    var node = allNodes.get(nodeId);
    if (!node || !node.title) return;
    _ttEl.innerHTML = _decodeHtml(node.title);
    _ttEl.style.display = 'block';
    var gap  = 18;
    var ttW  = _ttEl.offsetWidth  || 280;
    var ttH  = _ttEl.offsetHeight || 150;
    var cW   = _netEl.offsetWidth;
    var cH   = _netEl.offsetHeight;
    // Place tooltip to the right of cursor; flip left if it would overflow
    var left = _mouseX + gap;
    if (left + ttW > cW) left = _mouseX - ttW - gap;
    var top  = _mouseY - 10;
    if (top + ttH > cH) top = cH - ttH - 10;
    _ttEl.style.left = Math.max(0, left) + 'px';
    _ttEl.style.top  = Math.max(0, top)  + 'px';
  }

  function _hideTooltip() { _ttEl.style.display = 'none'; }

  // Show hidden label + custom tooltip on hover
  network.on('hoverNode', function(p) {
    if (_hiddenSet.has(p.node) && _origLabels[p.node])
      allNodes.update([{ id: p.node, label: _origLabels[p.node] }]);
    _showTooltip(p.node);
  });

  // Re-hide label + tooltip on blur
  network.on('blurNode', function(p) {
    if (_hiddenSet.has(p.node))
      allNodes.update([{ id: p.node, label: '' }]);
    _hideTooltip();
  });

  network.on('dragStart', _hideTooltip);

})();
</script>
"""


def _post_process_html(html: str, node_count: int = 0, network_type: str = "default") -> str:
    for old in [
        'background-color: #ffffff;',
        'background-color:#ffffff;',
        'background-color: white;',
        'background: white;',
        'background:#ffffff;',
        'background: #ffffff;',
    ]:
        html = html.replace(old, f'background:{CANVAS_BG};')
        html = html.replace(old.upper(), f'background:{CANVAS_BG};')

    html = re.sub(
        r'(#mynetwork\s*\{[^}]*)',
        rf'\1background:{CANVAS_BG} !important;',
        html,
        flags=re.DOTALL,
    )
    overlap_js = _LABEL_OVERLAP_JS if network_type == "keyword" else ""
    html = html.replace(
        "</body>", _STABILIZE_JS + _HIGHLIGHT_JS + _CONTROLS_JS + overlap_js + "</body>"
    )
    return html


def _pyvis_to_html(net, node_count: int = 0, network_type: str = "default") -> str:
    """Generate PyVis HTML string (no disk write)."""
    html = net.generate_html(notebook=False)
    return _post_process_html(html, node_count, network_type)


# ── Controls panel ────────────────────────────────────────────────────────────

def _render_controls(
    graph: nx.Graph,
    key_prefix: str,
    session_key_html: str,
    rebuild_fn,
    show_search: bool = True,
    show_density: bool = True,
    show_freeze: bool = True,
    show_png: bool = True,
) -> Tuple[str, float, float, Optional[List[int]], bool]:
    """
    Render the controls strip above a network graph.
    Returns (search_term, min_link, min_size, selected_communities, freeze).
    show_search / show_density / show_freeze let callers hide specific controls;
    hidden controls return their default values (empty string / False).
    """
    import streamlit as st

    # Build column widths dynamically based on which controls are visible.
    col_widths = []
    if show_search:
        col_widths.append(3)   # search
    col_widths.extend([2, 2, 2])  # min_link, min_size, communities always shown
    if show_density:
        col_widths.append(1)
    if show_freeze:
        col_widths.append(1)
    if show_png:
        col_widths.append(1)       # PNG (optional)

    cols = st.columns(col_widths)
    ci = 0  # column index cursor

    if show_search:
        with cols[ci]:
            search = st.text_input(
                "Search nodes…",
                key=f"{key_prefix}_search",
                placeholder="Search nodes…",
                label_visibility="collapsed",
            )
        ci += 1
    else:
        search = ""

    all_edge_weights = [
        graph[u][v].get("weight", 1) for u, v in graph.edges()
    ]
    edge_p90 = percentile(all_edge_weights, 90) if all_edge_weights else 1.0
    with cols[ci]:
        min_link = st.slider(
            "Min Link Strength",
            min_value=1,
            max_value=max(int(edge_p90), 2),
            value=1,
            key=f"{key_prefix}_min_link",
        )
    ci += 1

    all_node_weights = [
        graph.nodes[n].get("weight", 1) for n in graph.nodes()
    ]
    node_p75 = percentile(all_node_weights, 75) if all_node_weights else 1.0
    with cols[ci]:
        min_size = st.slider(
            "Min Node Size",
            min_value=1,
            max_value=max(int(node_p75), 2),
            value=1,
            key=f"{key_prefix}_min_size",
        )
    ci += 1

    all_communities = sorted(
        set(
            graph.nodes[n].get("community_id", 0)
            for n in graph.nodes()
        )
    )
    with cols[ci]:
        selected_comms = st.multiselect(
            "Communities",
            options=all_communities,
            default=all_communities,
            key=f"{key_prefix}_comms",
        )
    ci += 1

    if show_density:
        with cols[ci]:
            density_on = st.checkbox("Density", key=f"{key_prefix}_density")
        ci += 1

    if show_freeze:
        with cols[ci]:
            freeze = st.checkbox("❄ Freeze", key=f"{key_prefix}_freeze")
        ci += 1
    else:
        freeze = False

    if show_png:
        with cols[ci]:
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
    edge_alpha: float = 0.55,
    edge_roundness: float = 0.10,
    font_size_boost: int = 0,
    node_opacity: float = 1.0,
    network_type: str = "default",
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
    p50 = percentile(w_list, 50) if w_list else 1.0
    p75 = percentile(w_list, 75) if w_list else 1.0

    net = Network(
        height="850px",
        width="100%",
        directed=directed,
        bgcolor=CANVAS_BG,
        font_color="#000000",
    )
    net.toggle_physics(True)

    for node in graph.nodes():
        data = graph.nodes[node]
        fill_hex = data.get("color_hex", COMMUNITY_COLORS[0])
        size = node_sizes.get(node, NODE_SIZE_MIN)
        weight = node_weights.get(node, 1)
        label = label_fn(node, data)
        # Note: keyword network label overlap is handled by _LABEL_OVERLAP_JS
        # injected into the HTML — no static hiding here.
        tooltip = tooltip_fn(node, data, graph)
        shape = shape_fn(data) if shape_fn else "dot"
        if network_type == "keyword":
            # VOSviewer-style: font size scales with visual node size.
            # Hub nodes keep large prominent labels; smaller nodes get
            # proportionally smaller fonts that don't bleed onto neighbours.
            node_size_val = node_sizes.get(node, NODE_SIZE_MIN)
            font_px = max(10, min(30, int(node_size_val * 0.33)))
            stroke_w = 3 if font_px >= 20 else (2 if font_px >= 14 else 1)
            font = {
                "size": font_px,
                "color": "#000000",
                "face": "arial",
                "strokeWidth": stroke_w,
                "strokeColor": "#FFFFFF",
            }
        else:
            font = _label_font(weight, p50, p75)
            if font_size_boost:
                font = {**font, "size": font["size"] + font_size_boost}

        bg = hex_to_rgba(fill_hex, node_opacity) if node_opacity < 1.0 else fill_hex
        node_color = {
            "background": bg,
            "border": bg,
            "highlight": {"background": bg, "border": "#000000"},
            "hover": {"background": bg, "border": "#333333"},
        }

        net.add_node(
            node,
            label=label,
            title=tooltip,
            size=size,
            shape=shape,
            color=node_color,
            borderWidth=0,
            borderWidthSelected=0,
            font=font,
        )

    for u, v in graph.edges():
        src_community = graph.nodes[u].get("community_id", 0)
        src_color_hex = graph.nodes[u].get("color_hex", COMMUNITY_COLORS[0])
        edge_color = {
            "color": hex_to_rgba(src_color_hex, edge_alpha),
            "highlight": hex_to_rgba(src_color_hex, min(1.0, edge_alpha + 0.45)),
            "hover": hex_to_rgba(src_color_hex, min(1.0, edge_alpha + 0.30)),
            "inherit": False,
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
            smooth={"type": "continuous", "roundness": edge_roundness} if not directed else
                {"type": "curvedCW", "roundness": edge_roundness},
        )

    physics_opts = get_physics_options(graph.number_of_nodes(), network_type)
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
        show_search=False,
        show_density=False,
        show_freeze=False,
        show_png=False,
    )

    # Apply filters
    filtered = _filter_graph(graph, min_link, min_size, sel_comms, search)

    cache_key = (
        f"_coauth_html_{_VIZ_VERSION}_{key_prefix}_{min_link}_{min_size}_"
        f"{','.join(map(str, sel_comms))}_{search}_{freeze}_coauth"
    )
    if cache_key not in st.session_state or st.session_state[cache_key] is None:
        node_sizes = _compute_node_sizes(filtered, size_min=18, size_max=85)
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
            label_fn, tooltip_fn, lambda u, v, d: "",
            network_type="coauthorship",
            node_opacity=0.78,
            font_size_boost=8,
        )
        if freeze:
            net.toggle_physics(False)

        # Embed random x,y directly into each PyVis node dict BEFORE
        # generate_html() serialises them into the JS DataSet.
        # vis.js reads x/y from the DataSet at construction time — when
        # coordinates are already present it skips its circular seeding
        # entirely and starts physics from these positions.
        # This is more reliable than post-hoc JS injection because it
        # happens before `new vis.Network()` is even called.
        import random as _rnd
        _spread = 600
        for _node in net.nodes:
            _node['x'] = _rnd.uniform(-_spread, _spread)
            _node['y'] = _rnd.uniform(-_spread, _spread)

        html = _pyvis_to_html(net, filtered.number_of_nodes())

        # Hover effect: enlarge node + increase label size on hover.
        # vis.js chosen.node / chosen.label callbacks must be real JS
        # functions (not JSON), so they are injected via setOptions after
        # the network is initialised.
        # chosen.label checks window._coauthHiddenSet: nodes whose label
        # was hidden get no size multiplier (just shown at normal size);
        # nodes whose label was already visible get the 10x enlargement.
        _hover_js = """<script>
(function() {
  network.setOptions({
    nodes: {
      chosen: {
        node: function(values, id, selected, hovering) {
          if (hovering) {
            values.size = values.size * 1.35;
          }
        },
        label: function(values, id, selected, hovering) {
          if (hovering) {
            var isHidden = window._coauthHiddenSet && window._coauthHiddenSet.has(id);
            values.size        = values.size * (isHidden ? 1.5 : 2.0);
            values.color       = '#000000';
            values.strokeWidth = 3;
            values.strokeColor = '#ffffff';
          }
        }
      }
    }
  });
})();
</script>"""
        html = html.replace("</body>", _hover_js + "\n</body>")

        # Label overlap avoidance: hide labels that overlap a larger neighbour;
        # reveal the hidden label when the user hovers that node.
        # _hiddenSet is exposed as window._coauthHiddenSet so the chosen.label
        # callback above can skip the 10x multiplier for revealed-hidden labels.
        _coauth_overlap_js = """<style>
/* Strip vis.js default tooltip container so only our blue card shows */
.vis-tooltip {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  border-radius: 0 !important;
}
</style>
<script>
(function() {
  // Convert string node titles to DOM elements so vis.js renders them as
  // styled HTML cards instead of raw text (vis.js uses textContent for strings).
  allNodes.getIds().forEach(function(id) {
    var node = allNodes.get(id);
    if (node && typeof node.title === 'string' && node.title.trim()) {
      var el = document.createElement('div');
      el.innerHTML = node.title;
      allNodes.update([{ id: id, title: el }]);
    }
  });

  var _origLabels = {};
  window._coauthHiddenSet = new Set();
  var CHAR_W = 0.55;
  var PAD    = 4;

  function _bbox(nodeId) {
    var node = allNodes.get(nodeId);
    if (!node) return null;
    var lbl = _origLabels[nodeId];
    if (!lbl) return null;
    var fs    = (node.font && node.font.size) ? node.font.size : 20;
    var scale = network.getScale();
    var dp    = network.canvasToDOM(network.getPosition(nodeId));
    var w     = lbl.length * fs * CHAR_W * scale;
    var h     = fs * 1.3 * scale;
    return { l: dp.x - w/2 - PAD, r: dp.x + w/2 + PAD,
             t: dp.y - h/2 - PAD, b: dp.y + h/2 + PAD };
  }

  function _hit(a, b) {
    return !(a.r < b.l || a.l > b.r || a.b < b.t || a.t > b.b);
  }

  function _run() {
    allNodes.getIds().forEach(function(id) {
      if (!(id in _origLabels)) _origLabels[id] = allNodes.get(id).label || '';
    });
    var sorted = allNodes.getIds().map(function(id) {
      return { id: id, sz: allNodes.get(id).size || 10 };
    }).sort(function(a, b) { return b.sz - a.sz; });

    var kept = [], hidden = new Set(), updates = [];
    sorted.forEach(function(item) {
      var id = item.id, lbl = _origLabels[id];
      if (!lbl) return;
      var box = _bbox(id);
      if (!box) return;
      if (kept.some(function(k) { return _hit(box, k); })) {
        hidden.add(id);
        if (allNodes.get(id).label !== '') updates.push({ id: id, label: '' });
      } else {
        kept.push(box);
        if (allNodes.get(id).label !== lbl) updates.push({ id: id, label: lbl });
      }
    });
    window._coauthHiddenSet = hidden;
    if (updates.length) allNodes.update(updates);
  }
  // Run after _STABILIZE_JS finishes its last fit() at ~2500ms
  network.once('stabilizationIterationsDone', function() {
    setTimeout(_run, 3500);
  });

  // Re-evaluate on zoom
  var _zt = null;
  network.on('zoom', function() { clearTimeout(_zt); _zt = setTimeout(_run, 200); });

  // Reveal hidden label on hover; re-hide on blur
  network.on('hoverNode', function(p) {
    if (window._coauthHiddenSet.has(p.node) && _origLabels[p.node])
      allNodes.update([{ id: p.node, label: _origLabels[p.node] }]);
  });
  network.on('blurNode', function(p) {
    if (window._coauthHiddenSet.has(p.node))
      allNodes.update([{ id: p.node, label: '' }]);
  });
})();
</script>"""
        html = html.replace("</body>", _coauth_overlap_js + "\n</body>")

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
        "Edge = co-occurrence"
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
        f"_kw_html_{_VIZ_VERSION}_{key_prefix}_{min_link}_{min_size}_"
        f"{','.join(map(str, sel_comms))}_{search}_{freeze}"
    )
    if cache_key not in st.session_state:
        # Keyword network: wide size range for VOSviewer-style dramatic scaling;
        # thin, faint edges so cluster structure reads clearly.
        node_sizes = _compute_node_sizes(filtered, size_min=5, size_max=90)
        edge_widths = _compute_edge_widths(filtered, width_min=0.5, width_max=2.5)
        node_weights = {n: filtered.nodes[n].get("weight", 1)
                        for n in filtered.nodes()}

        def label_fn(node, data):
            return truncate(str(node), 30)

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

        net = _build_pyvis_network(
            filtered, node_sizes, edge_widths, node_weights,
            label_fn, tooltip_fn, _default_edge_tooltip,
            edge_alpha=0.18, edge_roundness=0.30,
            font_size_boost=14, node_opacity=0.78, network_type="keyword",
        )
        if freeze:
            net.toggle_physics(False)
        html = _pyvis_to_html(net, filtered.number_of_nodes(), network_type="keyword")
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
        f"_topic_html_{_VIZ_VERSION}_{key_prefix}_{min_link}_{min_size}_"
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
