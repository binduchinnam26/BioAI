"""
Global configuration for BioLitAI-X.
All values are read from environment variables or defined as constants here.
No domain-specific defaults exist in this file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DATABASE_DIR = DATA_DIR / "database"
MODELS_DIR = BASE_DIR / "models"

# Ensure runtime directories exist
for _d in (RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, DATABASE_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = str(DATABASE_DIR / "biolita.db")

# ── NCBI Entrez ───────────────────────────────────────────────────────────────
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")
ENTREZ_API_KEY = os.getenv("ENTREZ_API_KEY")

if not ENTREZ_EMAIL:
    raise EnvironmentError(
        "ENTREZ_EMAIL not found in .env file. "
        "Please add ENTREZ_EMAIL=your_email@example.com to your .env file."
    )

# ── Query / retrieval limits ──────────────────────────────────────────────────
MAX_RESULTS_DEFAULT = 300
MAX_RESULTS_MIN = 100
MAX_RESULTS_MAX = 300

# Number of PMIDs per Entrez batch-fetch request
ENTREZ_BATCH_SIZE = 100

# Requests per second allowed by NCBI
# 3/sec without API key, 10/sec with key
ENTREZ_RATE_LIMIT_NO_KEY = 3
ENTREZ_RATE_LIMIT_WITH_KEY = 10

# ── NLP / embedding ───────────────────────────────────────────────────────────
SCISPACY_MODEL = "en_core_sci_lg"
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
EMBEDDING_BATCH_SIZE = 64
SEMANTIC_SIMILARITY_THRESHOLD = 0.85
SEMANTIC_SEARCH_TOP_K = 10

# ── BERTopic ──────────────────────────────────────────────────────────────────
BERTOPIC_MIN_TOPIC_SIZE = 5
BERTOPIC_NR_TOPICS = "auto"

# ── Knowledge graph ───────────────────────────────────────────────────────────
KG_SIMILARITY_THRESHOLD = 0.85
KG_MIN_SHARED_NEIGHBORS_FOR_GAP = 3
LOUVAIN_RANDOM_STATE = 42

# ── Bibliometric network thresholds ───────────────────────────────────────────
KEYWORD_MIN_FREQUENCY = 3          # minimum papers a keyword must appear in
COAUTHOR_MIN_PAPERS = 1            # minimum papers for author node inclusion

# ── OpenAlex ─────────────────────────────────────────────────────────────────
OPENALEX_API_BASE = "https://api.openalex.org"
OPENALEX_EMAIL = os.getenv("ENTREZ_EMAIL")   # reuse same contact email

# ── Hypothesis generation ─────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Default to flash — generous free-tier quota (15 RPM / 1 500 RPD).
# Override to "gemini-2.5-pro" in .env for higher quality if quota allows.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_TEMPERATURE = 0.3
GEMINI_TOP_P = 0.85
GEMINI_TOP_K = 40
GEMINI_MAX_OUTPUT_TOKENS = 2048
# Free-tier safe: process 5 gaps per run by default (flash: 15 RPM, pro: 2 RPM).
# Raise in .env (HYPOTHESIS_TOP_GAPS=10) only if on a paid plan.
HYPOTHESIS_TOP_GAPS = int(os.getenv("HYPOTHESIS_TOP_GAPS", "5"))
# Inter-call delay is enforced dynamically per model inside hypothesis_generator.py
HYPOTHESIS_API_DELAY_SECONDS = 4   # minimum delay; actual may be longer for pro
HYPOTHESIS_MAX_RETRIES = 3

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Visualization ─────────────────────────────────────────────────────────────
# Node size scaling
NODE_SIZE_MIN = 12
NODE_SIZE_MAX = 55

# Edge width scaling
EDGE_WIDTH_MIN = 0.5
EDGE_WIDTH_MAX = 8.0

# Canvas background
CANVAS_BG = "#0D1117"

# Community color palette (cycles if > 10 communities)
COMMUNITY_COLORS = [
    "#4E9AF1",  # 0 clear blue
    "#9B72CF",  # 1 muted violet
    "#34C78A",  # 2 emerald green
    "#F5A623",  # 3 warm amber
    "#E85D5D",  # 4 coral red
    "#26C9D3",  # 5 cyan teal
    "#F97B4F",  # 6 burnt orange
    "#D4699E",  # 7 dusty pink
    "#A3C44B",  # 8 yellow-green lime
    "#5BC8C8",  # 9 light teal
]

# Entity type colors for knowledge graph
ENTITY_TYPE_COLORS = {
    "DISEASE":              "#E85D5D",
    "GENE_OR_GENOME":       "#4E9AF1",
    "CHEMICAL":             "#34C78A",
    "BIOLOGICAL_PROCESS":   "#F97B4F",
    "CELL":                 "#9B72CF",
    "ORGANISM":             "#26C9D3",
    "LABORATORY_PROCEDURE": "#F5A623",
}

# ── Design system colors ──────────────────────────────────────────────────────
COLOR_BACKGROUND        = "#0A0F1E"
COLOR_SURFACE           = "#111827"
COLOR_SURFACE_ELEVATED  = "#1C2539"
COLOR_PRIMARY           = "#3B82F6"
COLOR_SECONDARY         = "#8B5CF6"
COLOR_SUCCESS           = "#10B981"
COLOR_WARNING           = "#F59E0B"
COLOR_DANGER            = "#EF4444"
COLOR_TEXT_PRIMARY      = "#F9FAFB"
COLOR_TEXT_SECONDARY    = "#9CA3AF"
COLOR_BORDER            = "#1F2937"
