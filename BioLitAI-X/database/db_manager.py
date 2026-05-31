"""
DatabaseManager — thread-safe SQLite interface for BioLitAI-X.
Provides all CRUD operations needed by the pipeline and UI layers.
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from database.schema import SCHEMA_SQL
from config import DB_PATH

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Thread-safe SQLite database manager with connection pooling via threading.local."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._local = threading.local()
        self._init_schema()

    # ── Connection management ─────────────────────────────────────────────────

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _transaction(self):
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self):
        try:
            conn = self._get_connection()
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            logger.info("Database schema initialised at %s", self.db_path)
        except Exception as exc:
            logger.error("Schema initialisation failed: %s", exc)
            raise

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ── Papers ────────────────────────────────────────────────────────────────

    def insert_paper(self, paper: Dict[str, Any]) -> bool:
        """Insert or replace a paper record. Returns True on success."""
        sql = """
            INSERT OR REPLACE INTO papers
              (pmid, title, abstract, authors, keywords, mesh_terms,
               mesh_qualifiers, chemical_terms, publication_types, grant_info,
               journal, volume, issue, pages, pub_date, doi, language,
               query_used, fetched_at)
            VALUES
              (:pmid, :title, :abstract, :authors, :keywords, :mesh_terms,
               :mesh_qualifiers, :chemical_terms, :publication_types, :grant_info,
               :journal, :volume, :issue, :pages, :pub_date, :doi, :language,
               :query_used, :fetched_at)
        """
        # Serialise list/dict fields to JSON
        row = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
               for k, v in paper.items()}
        row.setdefault("fetched_at", datetime.now(timezone.utc).isoformat())
        try:
            with self._transaction() as conn:
                conn.execute(sql, row)
            return True
        except Exception as exc:
            logger.error("insert_paper failed for pmid=%s: %s", paper.get("pmid"), exc)
            return False

    def get_paper(self, pmid: str) -> Optional[Dict]:
        try:
            conn = self._get_connection()
            row = conn.execute("SELECT * FROM papers WHERE pmid=?", (pmid,)).fetchone()
            return self._deserialise_paper(dict(row)) if row else None
        except Exception as exc:
            logger.error("get_paper failed for pmid=%s: %s", pmid, exc)
            return None

    def get_all_papers(self) -> List[Dict]:
        try:
            conn = self._get_connection()
            rows = conn.execute("SELECT * FROM papers").fetchall()
            return [self._deserialise_paper(dict(r)) for r in rows]
        except Exception as exc:
            logger.error("get_all_papers failed: %s", exc)
            return []

    def get_papers_by_query(self, query_text: str) -> List[Dict]:
        try:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM papers WHERE query_used=?", (query_text,)
            ).fetchall()
            return [self._deserialise_paper(dict(r)) for r in rows]
        except Exception as exc:
            logger.error("get_papers_by_query failed: %s", exc)
            return []

    def search_papers(self, term: str) -> List[Dict]:
        """Full-text search across title and abstract."""
        try:
            like = f"%{term}%"
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM papers WHERE title LIKE ? OR abstract LIKE ?",
                (like, like),
            ).fetchall()
            return [self._deserialise_paper(dict(r)) for r in rows]
        except Exception as exc:
            logger.error("search_papers failed: %s", exc)
            return []

    @staticmethod
    def _deserialise_paper(row: Dict) -> Dict:
        json_fields = (
            "authors", "keywords", "mesh_terms", "mesh_qualifiers",
            "chemical_terms", "publication_types", "grant_info",
        )
        for field in json_fields:
            if field in row and isinstance(row[field], str):
                try:
                    row[field] = json.loads(row[field])
                except (json.JSONDecodeError, TypeError):
                    row[field] = []
        return row

    # ── Authors ───────────────────────────────────────────────────────────────

    def insert_author(self, name: str, normalized_name: str,
                      affiliation: Optional[str] = None) -> int:
        """Insert author if not exists; return author id."""
        try:
            conn = self._get_connection()
            existing = conn.execute(
                "SELECT id FROM authors WHERE normalized_name=?",
                (normalized_name,),
            ).fetchone()
            if existing:
                return existing["id"]
            with self._transaction() as conn:
                cur = conn.execute(
                    "INSERT INTO authors (name, normalized_name, affiliation) "
                    "VALUES (?, ?, ?)",
                    (name, normalized_name, affiliation),
                )
                return cur.lastrowid
        except Exception as exc:
            logger.error("insert_author failed for name=%s: %s", name, exc)
            return -1

    def link_paper_author(self, pmid: str, author_id: int, position: int = 0):
        try:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO paper_authors "
                    "(paper_pmid, author_id, author_position) VALUES (?, ?, ?)",
                    (pmid, author_id, position),
                )
        except Exception as exc:
            logger.error("link_paper_author failed pmid=%s author=%d: %s",
                         pmid, author_id, exc)

    # ── MeSH terms ────────────────────────────────────────────────────────────

    def insert_mesh_term(self, descriptor: str, qualifier: Optional[str],
                         is_major: bool) -> int:
        try:
            conn = self._get_connection()
            existing = conn.execute(
                "SELECT id FROM mesh_terms "
                "WHERE descriptor=? AND (qualifier=? OR (qualifier IS NULL AND ? IS NULL))",
                (descriptor, qualifier, qualifier),
            ).fetchone()
            if existing:
                return existing["id"]
            with self._transaction() as conn:
                cur = conn.execute(
                    "INSERT INTO mesh_terms (descriptor, qualifier, is_major_topic) "
                    "VALUES (?, ?, ?)",
                    (descriptor, qualifier, int(is_major)),
                )
                return cur.lastrowid
        except Exception as exc:
            logger.error("insert_mesh_term failed descriptor=%s: %s", descriptor, exc)
            return -1

    def link_paper_mesh(self, pmid: str, mesh_id: int):
        try:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO paper_mesh (paper_pmid, mesh_term_id) "
                    "VALUES (?, ?)",
                    (pmid, mesh_id),
                )
        except Exception as exc:
            logger.error("link_paper_mesh failed pmid=%s mesh=%d: %s",
                         pmid, mesh_id, exc)

    def get_mesh_by_paper(self, pmid: str) -> List[Dict]:
        try:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT m.* FROM mesh_terms m "
                "JOIN paper_mesh pm ON pm.mesh_term_id=m.id "
                "WHERE pm.paper_pmid=?",
                (pmid,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("get_mesh_by_paper failed pmid=%s: %s", pmid, exc)
            return []

    # ── Chemical terms ────────────────────────────────────────────────────────

    def insert_chemical_term(self, name: str,
                             registry_number: Optional[str] = None) -> int:
        try:
            conn = self._get_connection()
            existing = conn.execute(
                "SELECT id FROM chemical_terms WHERE name=?", (name,)
            ).fetchone()
            if existing:
                return existing["id"]
            with self._transaction() as conn:
                cur = conn.execute(
                    "INSERT INTO chemical_terms (name, registry_number) VALUES (?, ?)",
                    (name, registry_number),
                )
                return cur.lastrowid
        except Exception as exc:
            logger.error("insert_chemical_term failed name=%s: %s", name, exc)
            return -1

    def link_paper_chemical(self, pmid: str, chem_id: int):
        try:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO paper_chemicals "
                    "(paper_pmid, chemical_term_id) VALUES (?, ?)",
                    (pmid, chem_id),
                )
        except Exception as exc:
            logger.error("link_paper_chemical failed pmid=%s chem=%d: %s",
                         pmid, chem_id, exc)

    def get_chemicals_by_paper(self, pmid: str) -> List[Dict]:
        try:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT c.* FROM chemical_terms c "
                "JOIN paper_chemicals pc ON pc.chemical_term_id=c.id "
                "WHERE pc.paper_pmid=?",
                (pmid,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("get_chemicals_by_paper failed pmid=%s: %s", pmid, exc)
            return []

    # ── Publication types ─────────────────────────────────────────────────────

    def insert_publication_type(self, pub_type: str) -> int:
        try:
            conn = self._get_connection()
            existing = conn.execute(
                "SELECT id FROM publication_types WHERE publication_type=?",
                (pub_type,),
            ).fetchone()
            if existing:
                return existing["id"]
            with self._transaction() as conn:
                cur = conn.execute(
                    "INSERT INTO publication_types (publication_type) VALUES (?)",
                    (pub_type,),
                )
                return cur.lastrowid
        except Exception as exc:
            logger.error("insert_publication_type failed type=%s: %s", pub_type, exc)
            return -1

    def link_paper_publication_type(self, pmid: str, pt_id: int):
        try:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO paper_publication_types "
                    "(paper_pmid, publication_type_id) VALUES (?, ?)",
                    (pmid, pt_id),
                )
        except Exception as exc:
            logger.error("link_paper_publication_type failed pmid=%s pt=%d: %s",
                         pmid, pt_id, exc)

    def get_publication_types_by_paper(self, pmid: str) -> List[str]:
        try:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT pt.publication_type FROM publication_types pt "
                "JOIN paper_publication_types ppt ON ppt.publication_type_id=pt.id "
                "WHERE ppt.paper_pmid=?",
                (pmid,),
            ).fetchall()
            return [r["publication_type"] for r in rows]
        except Exception as exc:
            logger.error("get_publication_types_by_paper failed pmid=%s: %s", pmid, exc)
            return []

    # ── Keywords ──────────────────────────────────────────────────────────────

    def insert_keyword(self, keyword: str, normalized: str,
                       kw_type: str = "author_keyword") -> int:
        try:
            conn = self._get_connection()
            existing = conn.execute(
                "SELECT id FROM keywords WHERE normalized_keyword=? AND keyword_type=?",
                (normalized, kw_type),
            ).fetchone()
            if existing:
                return existing["id"]
            with self._transaction() as conn:
                cur = conn.execute(
                    "INSERT INTO keywords (keyword, normalized_keyword, keyword_type) "
                    "VALUES (?, ?, ?)",
                    (keyword, normalized, kw_type),
                )
                return cur.lastrowid
        except Exception as exc:
            logger.error("insert_keyword failed keyword=%s: %s", keyword, exc)
            return -1

    def link_paper_keyword(self, pmid: str, keyword_id: int):
        try:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO paper_keywords "
                    "(paper_pmid, keyword_id) VALUES (?, ?)",
                    (pmid, keyword_id),
                )
        except Exception as exc:
            logger.error("link_paper_keyword failed pmid=%s kw=%d: %s",
                         pmid, keyword_id, exc)

    # ── Entities ──────────────────────────────────────────────────────────────

    def insert_entity(self, name: str, entity_type: str,
                      umls_id: Optional[str] = None) -> int:
        try:
            conn = self._get_connection()
            existing = conn.execute(
                "SELECT id FROM entities WHERE name=? AND entity_type=?",
                (name, entity_type),
            ).fetchone()
            if existing:
                return existing["id"]
            with self._transaction() as conn:
                cur = conn.execute(
                    "INSERT INTO entities (name, entity_type, umls_id) "
                    "VALUES (?, ?, ?)",
                    (name, entity_type, umls_id),
                )
                return cur.lastrowid
        except Exception as exc:
            logger.error("insert_entity failed name=%s: %s", name, exc)
            return -1

    def link_paper_entity(self, pmid: str, entity_id: int,
                          sentence_context: Optional[str] = None):
        try:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO paper_entities "
                    "(paper_pmid, entity_id, sentence_context) VALUES (?, ?, ?)",
                    (pmid, entity_id, sentence_context),
                )
        except Exception as exc:
            logger.error("link_paper_entity failed pmid=%s ent=%d: %s",
                         pmid, entity_id, exc)

    # ── Relationships ─────────────────────────────────────────────────────────

    def insert_relationship(self, source_id: int, target_id: int,
                            rel_type: str, evidence_pmid: str,
                            confidence: float = 0.0) -> int:
        try:
            with self._transaction() as conn:
                cur = conn.execute(
                    "INSERT INTO relationships "
                    "(source_entity_id, target_entity_id, relationship_type, "
                    " evidence_pmid, confidence_score) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (source_id, target_id, rel_type, evidence_pmid, confidence),
                )
                return cur.lastrowid
        except Exception as exc:
            logger.error("insert_relationship failed: %s", exc)
            return -1

    def get_relationships_by_paper(self, pmid: str) -> List[Dict]:
        try:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM relationships WHERE evidence_pmid=?", (pmid,)
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("get_relationships_by_paper failed pmid=%s: %s", pmid, exc)
            return []

    def get_all_relationships(self) -> List[Dict]:
        try:
            conn = self._get_connection()
            rows = conn.execute("SELECT * FROM relationships").fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("get_all_relationships failed: %s", exc)
            return []

    def get_all_entities(self) -> List[Dict]:
        try:
            conn = self._get_connection()
            rows = conn.execute("SELECT * FROM entities").fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("get_all_entities failed: %s", exc)
            return []

    # ── Hypotheses ────────────────────────────────────────────────────────────

    def insert_hypothesis(self, hyp: Dict[str, Any]) -> int:
        try:
            with self._transaction() as conn:
                evidence = hyp.get("evidence_pmids", [])
                cur = conn.execute(
                    "INSERT INTO hypotheses "
                    "(concept_a, concept_b, hypothesis_text, evidence_pmids, "
                    " confidence_score, created_at, query_used, raw_response) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        hyp.get("concept_a"),
                        hyp.get("concept_b"),
                        hyp.get("hypothesis_text"),
                        json.dumps(evidence) if isinstance(evidence, list) else evidence,
                        hyp.get("confidence_score", 0.0),
                        hyp.get("created_at",
                                datetime.now(timezone.utc).isoformat()),
                        hyp.get("query_used"),
                        hyp.get("raw_response"),
                    ),
                )
                return cur.lastrowid
        except Exception as exc:
            logger.error("insert_hypothesis failed: %s", exc)
            return -1

    def get_hypotheses_by_query(self, query_text: str) -> List[Dict]:
        try:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM hypotheses WHERE query_used=? "
                "ORDER BY confidence_score DESC",
                (query_text,),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["evidence_pmids"] = json.loads(d["evidence_pmids"] or "[]")
                except (json.JSONDecodeError, TypeError):
                    d["evidence_pmids"] = []
                result.append(d)
            return result
        except Exception as exc:
            logger.error("get_hypotheses_by_query failed: %s", exc)
            return []

    # ── Query sessions ────────────────────────────────────────────────────────

    def save_query_session(self, query_text: str, max_results: int) -> int:
        try:
            with self._transaction() as conn:
                cur = conn.execute(
                    "INSERT INTO query_sessions "
                    "(query_text, max_results, pipeline_status, created_at) "
                    "VALUES (?, ?, 'pending', ?)",
                    (query_text, max_results,
                     datetime.now(timezone.utc).isoformat()),
                )
                return cur.lastrowid
        except Exception as exc:
            logger.error("save_query_session failed: %s", exc)
            return -1

    def update_query_session(self, session_id: int, **kwargs):
        if not kwargs:
            return
        valid = {"papers_fetched", "pipeline_status"}
        fields = {k: v for k, v in kwargs.items() if k in valid}
        if not fields:
            return
        set_clause = ", ".join(f"{k}=?" for k in fields)
        values = list(fields.values()) + [session_id]
        try:
            with self._transaction() as conn:
                conn.execute(
                    f"UPDATE query_sessions SET {set_clause} WHERE id=?", values
                )
        except Exception as exc:
            logger.error("update_query_session failed session=%d: %s",
                         session_id, exc)

    def get_all_sessions(self) -> List[Dict]:
        try:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM query_sessions ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("get_all_sessions failed: %s", exc)
            return []

    def get_session_by_id(self, session_id: int) -> Optional[Dict]:
        try:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT * FROM query_sessions WHERE id=?", (session_id,)
            ).fetchone()
            return dict(row) if row else None
        except Exception as exc:
            logger.error("get_session_by_id failed id=%d: %s", session_id, exc)
            return None

    # ── Embeddings metadata ───────────────────────────────────────────────────

    def upsert_embedding_meta(self, pmid: str, embedding_path: str,
                              model_used: str):
        try:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings_meta "
                    "(pmid, embedding_path, model_used, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (pmid, embedding_path, model_used,
                     datetime.now(timezone.utc).isoformat()),
                )
        except Exception as exc:
            logger.error("upsert_embedding_meta failed pmid=%s: %s", pmid, exc)

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_statistics(self, query_text: Optional[str] = None) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            where = "WHERE query_used=?" if query_text else ""
            params: Tuple = (query_text,) if query_text else ()

            paper_count = conn.execute(
                f"SELECT COUNT(*) FROM papers {where}", params
            ).fetchone()[0]

            author_count = conn.execute(
                "SELECT COUNT(DISTINCT author_id) FROM paper_authors pa "
                + ("JOIN papers p ON p.pmid=pa.paper_pmid " + where if query_text else ""),
                params,
            ).fetchone()[0]

            entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            rel_count = conn.execute(
                "SELECT COUNT(*) FROM relationships"
            ).fetchone()[0]
            hyp_count = conn.execute(
                f"SELECT COUNT(*) FROM hypotheses {where}", params
            ).fetchone()[0]

            return {
                "paper_count": paper_count,
                "author_count": author_count,
                "entity_count": entity_count,
                "relationship_count": rel_count,
                "hypothesis_count": hyp_count,
            }
        except Exception as exc:
            logger.error("get_statistics failed: %s", exc)
            return {}

    # ── Topics ────────────────────────────────────────────────────────────────

    def insert_topic(self, topic_label: str, top_words: List,
                     paper_count: int, query_used: str) -> int:
        try:
            with self._transaction() as conn:
                cur = conn.execute(
                    "INSERT INTO topics (topic_label, top_words, paper_count, query_used) "
                    "VALUES (?, ?, ?, ?)",
                    (topic_label, json.dumps(top_words), paper_count, query_used),
                )
                return cur.lastrowid
        except Exception as exc:
            logger.error("insert_topic failed: %s", exc)
            return -1

    def link_paper_topic(self, pmid: str, topic_id: int, probability: float):
        try:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO paper_topics "
                    "(paper_pmid, topic_id, probability) VALUES (?, ?, ?)",
                    (pmid, topic_id, probability),
                )
        except Exception as exc:
            logger.error("link_paper_topic failed pmid=%s topic=%d: %s",
                         pmid, topic_id, exc)

    def get_topics_by_query(self, query_text: str) -> List[Dict]:
        try:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM topics WHERE query_used=? ORDER BY paper_count DESC",
                (query_text,),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["top_words"] = json.loads(d["top_words"] or "[]")
                except (json.JSONDecodeError, TypeError):
                    d["top_words"] = []
                result.append(d)
            return result
        except Exception as exc:
            logger.error("get_topics_by_query failed: %s", exc)
            return []
