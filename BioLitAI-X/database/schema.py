"""
SQLite schema definitions for BioLitAI-X.
All tables are created via plain DDL executed against the SQLite connection.
"""

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ─────────────────────────────────────────────────────────────────────────────
-- Core paper table
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS papers (
    pmid                TEXT PRIMARY KEY,
    title               TEXT,
    abstract            TEXT,
    authors             TEXT,          -- JSON-serialised list of author dicts
    keywords            TEXT,          -- JSON-serialised list (author keywords only)
    mesh_terms          TEXT,          -- JSON-serialised list of MeSH descriptor dicts
    mesh_qualifiers     TEXT,          -- JSON-serialised list of qualifier strings
    chemical_terms      TEXT,          -- JSON-serialised list of chemical dicts
    publication_types   TEXT,          -- JSON-serialised list of strings
    grant_info          TEXT,          -- JSON-serialised list of grant dicts
    journal             TEXT,
    volume              TEXT,
    issue               TEXT,
    pages               TEXT,
    pub_date            TEXT,
    doi                 TEXT,
    language            TEXT,
    query_used          TEXT,
    fetched_at          TEXT
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Normalised author table
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS authors (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL,
    normalized_name     TEXT,
    affiliation         TEXT
);

CREATE TABLE IF NOT EXISTS paper_authors (
    paper_pmid          TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    author_id           INTEGER NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
    author_position     INTEGER,
    PRIMARY KEY (paper_pmid, author_id)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Keywords (author-supplied keywords only)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS keywords (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword             TEXT NOT NULL,
    normalized_keyword  TEXT,
    keyword_type        TEXT DEFAULT 'author_keyword'
);

CREATE TABLE IF NOT EXISTS paper_keywords (
    paper_pmid          TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    keyword_id          INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE,
    PRIMARY KEY (paper_pmid, keyword_id)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- MeSH terms (descriptor + qualifier stored separately)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS mesh_terms (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    descriptor          TEXT NOT NULL,
    qualifier           TEXT,
    is_major_topic      INTEGER DEFAULT 0   -- 1 = major topic, 0 = minor
);

CREATE TABLE IF NOT EXISTS paper_mesh (
    paper_pmid          TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    mesh_term_id        INTEGER NOT NULL REFERENCES mesh_terms(id) ON DELETE CASCADE,
    PRIMARY KEY (paper_pmid, mesh_term_id)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Chemical / substance terms
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chemical_terms (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL,
    registry_number     TEXT
);

CREATE TABLE IF NOT EXISTS paper_chemicals (
    paper_pmid          TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    chemical_term_id    INTEGER NOT NULL REFERENCES chemical_terms(id) ON DELETE CASCADE,
    PRIMARY KEY (paper_pmid, chemical_term_id)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Publication types
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS publication_types (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    publication_type    TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS paper_publication_types (
    paper_pmid          TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    publication_type_id INTEGER NOT NULL REFERENCES publication_types(id) ON DELETE CASCADE,
    PRIMARY KEY (paper_pmid, publication_type_id)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- BERTopic topics
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS topics (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_label         TEXT,
    top_words           TEXT,          -- JSON-serialised list of (word, score) tuples
    paper_count         INTEGER DEFAULT 0,
    query_used          TEXT
);

CREATE TABLE IF NOT EXISTS paper_topics (
    paper_pmid          TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    topic_id            INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    probability         REAL DEFAULT 0.0,
    PRIMARY KEY (paper_pmid, topic_id)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- NER entities
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS entities (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL,
    entity_type         TEXT,
    umls_id             TEXT
);

CREATE TABLE IF NOT EXISTS paper_entities (
    paper_pmid          TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    entity_id           INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    sentence_context    TEXT,
    PRIMARY KEY (paper_pmid, entity_id)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Entity relationships
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS relationships (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity_id    INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id    INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relationship_type   TEXT,
    evidence_pmid       TEXT,
    confidence_score    REAL DEFAULT 0.0
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Hypotheses
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS hypotheses (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_a           TEXT,
    concept_b           TEXT,
    hypothesis_text     TEXT,
    evidence_pmids      TEXT,          -- JSON-serialised list
    confidence_score    REAL DEFAULT 0.0,
    created_at          TEXT,
    query_used          TEXT,
    raw_response        TEXT
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Embedding metadata
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS embeddings_meta (
    pmid                TEXT PRIMARY KEY REFERENCES papers(pmid) ON DELETE CASCADE,
    embedding_path      TEXT,
    model_used          TEXT,
    created_at          TEXT
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Query sessions
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_sessions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text          TEXT NOT NULL,
    max_results         INTEGER,
    papers_fetched      INTEGER DEFAULT 0,
    pipeline_status     TEXT DEFAULT 'pending',
    created_at          TEXT
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Indexes for common look-ups
-- ─────────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_papers_query     ON papers(query_used);
CREATE INDEX IF NOT EXISTS idx_papers_pubdate   ON papers(pub_date);
CREATE INDEX IF NOT EXISTS idx_authors_norm     ON authors(normalized_name);
CREATE INDEX IF NOT EXISTS idx_entities_type    ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_name    ON entities(name);
CREATE INDEX IF NOT EXISTS idx_rels_source      ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_rels_target      ON relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_hyp_query        ON hypotheses(query_used);
CREATE INDEX IF NOT EXISTS idx_sessions_query   ON query_sessions(query_text);
"""
