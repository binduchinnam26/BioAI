"""
Shared utility functions used across the BioLitAI-X pipeline.
"""

import colorsys
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple


# ── Color utilities ───────────────────────────────────────────────────────────

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert #RRGGBB to (R, G, B) integers."""
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def lighten_hex(hex_color: str, amount: float = 0.30) -> str:
    """
    Lighten a hex color by *amount* (0–1) in HSL lightness space.
    Used for node border colors to match VOSviewer's inner-glow effect.
    """
    r, g, b = hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    l = min(1.0, l + amount)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return rgb_to_hex(int(r2 * 255), int(g2 * 255), int(b2 * 255))


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB + alpha to rgba(R, G, B, alpha) CSS string."""
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha})"


# ── Node / edge scaling formulas (VOSviewer-faithful) ─────────────────────────

def scale_node_size(weight: float, w_min: float, w_max: float,
                    size_min: float = 12.0, size_max: float = 55.0) -> float:
    """Square-root-compressed node size scaling matching VOSviewer."""
    scaled = ((weight - w_min) / (w_max - w_min + 1e-9)) ** 0.5
    return size_min + scaled * (size_max - size_min)


def scale_edge_width(weight: float, w_min: float, w_max: float,
                     width_min: float = 0.5, width_max: float = 8.0) -> float:
    """Logarithmic edge width scaling matching VOSviewer."""
    import math
    log_w = math.log(1 + weight - w_min)
    log_max = math.log(1 + w_max - w_min + 1e-9)
    return width_min + (log_w / log_max) * (width_max - width_min)


# ── Text utilities ────────────────────────────────────────────────────────────

def truncate(text: str, max_len: int, ellipsis: str = "…") -> str:
    if not text:
        return ""
    return text if len(text) <= max_len else text[: max_len - len(ellipsis)] + ellipsis


def clean_html(text: str) -> str:
    """Strip HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", text or "")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


# ── Hashing ───────────────────────────────────────────────────────────────────

def query_hash(query: str) -> str:
    """Stable 12-character hex hash for a query string — used for file naming."""
    return hashlib.sha256(query.encode()).hexdigest()[:12]


# ── JSON helpers ──────────────────────────────────────────────────────────────

def safe_json_loads(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


# ── Author name formatting ────────────────────────────────────────────────────

def format_author_short(name: str) -> str:
    """
    Convert "Lastname, Firstname Middle" to "Lastname FM" for graph labels.
    """
    if not name:
        return ""
    parts = name.split(",", 1)
    last = parts[0].strip()
    if len(parts) == 2:
        fore = parts[1].strip()
        initials = "".join(w[0].upper() for w in fore.split() if w)
        return f"{last} {initials}" if initials else last
    return last


# ── Percentile helper ─────────────────────────────────────────────────────────

def percentile(values: List[float], pct: float) -> float:
    """Return the *pct*-th percentile (0–100) of *values*."""
    if not values:
        return 0.0
    import statistics
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return float(sorted_vals[-1])
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])
