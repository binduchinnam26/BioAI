"""
TopicModeler — BERTopic-based topic modelling over a paper corpus.

Topic labels are derived entirely from the data (top BERTopic keywords).
No domain-specific labels are ever hardcoded.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    BERTOPIC_MIN_TOPIC_SIZE,
    BERTOPIC_NR_TOPICS,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


class TopicModeler:
    """
    Wraps BERTopic for corpus-level topic discovery and temporal analysis.
    Labels are always generated from the data — never hardcoded.
    """

    def __init__(self):
        self._model = None
        self._topic_info: Optional[pd.DataFrame] = None
        self._topics: Optional[List[int]] = None
        self._probs: Optional[np.ndarray] = None

    # ── Model initialisation ──────────────────────────────────────────────────

    def _init_model(self):
        try:
            from bertopic import BERTopic
            from umap import UMAP
            from hdbscan import HDBSCAN
        except ImportError as exc:
            raise ImportError(
                "BERTopic / UMAP / HDBSCAN not installed. "
                "Run: pip install bertopic umap-learn hdbscan"
            ) from exc

        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=BERTOPIC_MIN_TOPIC_SIZE,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        nr_topics = (
            None if BERTOPIC_NR_TOPICS == "auto" else int(BERTOPIC_NR_TOPICS)
        )
        self._model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=nr_topics,
            calculate_probabilities=True,
            verbose=False,
        )
        logger.info("BERTopic model initialised.")

    # ── Fit and transform ─────────────────────────────────────────────────────

    def fit_transform(
        self,
        abstracts: List[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Fit BERTopic on *abstracts* (optionally using pre-computed
        *embeddings* to skip internal re-embedding).

        Returns:
            topics  — list of topic IDs, one per document (-1 = outlier)
            probs   — array of shape (N, num_topics) with assignment probs
        """
        if not abstracts:
            logger.warning("fit_transform called with empty abstract list")
            return [], np.array([])

        self._init_model()

        # BERTopic expects clean text; replace empty strings with a placeholder
        docs = [a if (a and a.strip()) else "[NO ABSTRACT]" for a in abstracts]

        logger.info("Fitting BERTopic on %d documents …", len(docs))
        try:
            if embeddings is not None and len(embeddings) == len(docs):
                self._topics, self._probs = self._model.fit_transform(
                    docs, embeddings.astype(np.float32)
                )
            else:
                self._topics, self._probs = self._model.fit_transform(docs)
        except Exception as exc:
            logger.error("BERTopic fit_transform failed: %s", exc)
            # Return all-outlier assignment as graceful fallback
            self._topics = [-1] * len(docs)
            self._probs = np.zeros((len(docs), 1))

        self._topic_info = self._model.get_topic_info()
        n_topics = len(self._topic_info[self._topic_info["Topic"] >= 0])
        logger.info(
            "BERTopic complete: %d topics discovered (+ outlier bucket)",
            n_topics,
        )
        return self._topics, self._probs

    # ── Topic labels ──────────────────────────────────────────────────────────

    def get_topic_labels(self) -> Dict[int, str]:
        """
        Return a mapping {topic_id: label_string} where each label is
        the top-3 keywords for that topic joined by ' / '.
        Labels are derived from the data — never hardcoded.
        """
        if self._model is None:
            return {}
        labels: Dict[int, str] = {}
        for topic_id in self._model.get_topics():
            if topic_id == -1:
                labels[-1] = "Outliers"
                continue
            words = self._model.get_topic(topic_id)
            if words:
                top_words = [w for w, _ in words[:3]]
                labels[topic_id] = " / ".join(top_words)
            else:
                labels[topic_id] = f"Topic {topic_id}"
        return labels

    def get_top_words(self, topic_id: int, n: int = 10) -> List[Tuple[str, float]]:
        """Return top *n* (word, score) pairs for *topic_id*."""
        if self._model is None:
            return []
        words = self._model.get_topic(topic_id)
        return words[:n] if words else []

    # ── Temporal analysis ─────────────────────────────────────────────────────

    def get_topic_over_time(
        self, papers_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Compute topic frequency over publication years.

        Requires papers_df to have 'abstract', 'pub_year' columns, and
        fit_transform to have been called first.

        Returns a DataFrame with columns:
            Topic, Words, Frequency, Year
        """
        if self._model is None or self._topics is None:
            logger.warning("get_topic_over_time: model not fitted yet.")
            return None

        if "pub_year" not in papers_df.columns:
            logger.warning("get_topic_over_time: 'pub_year' column missing.")
            return None

        docs = papers_df["abstract"].fillna("").tolist()
        timestamps = papers_df["pub_year"].fillna(0).astype(int).tolist()

        if len(docs) != len(self._topics):
            logger.warning(
                "get_topic_over_time: doc count (%d) ≠ topic count (%d)",
                len(docs), len(self._topics),
            )
            return None

        try:
            tot = self._model.topics_over_time(
                docs, timestamps, nr_bins=10, global_tuning=True
            )
            return tot
        except Exception as exc:
            logger.error("topics_over_time failed: %s", exc)
            return None

    # ── Paper–topic mapping ───────────────────────────────────────────────────

    def get_paper_topic_assignments(
        self, pmids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Return a list of {pmid, topic_id, probability} dicts for every paper.
        """
        if self._topics is None or self._probs is None:
            return []
        results = []
        for i, (pmid, topic_id) in enumerate(zip(pmids, self._topics)):
            if i < len(self._probs):
                probs_row = self._probs[i]
                if hasattr(probs_row, "__len__") and len(probs_row) > 0:
                    prob = float(
                        probs_row[topic_id] if topic_id >= 0
                        and topic_id < len(probs_row)
                        else 0.0
                    )
                else:
                    prob = float(probs_row) if topic_id >= 0 else 0.0
            else:
                prob = 0.0
            results.append(
                {"pmid": pmid, "topic_id": int(topic_id), "probability": prob}
            )
        return results

    def get_topic_paper_counts(self) -> Dict[int, int]:
        """Return {topic_id: paper_count} excluding outliers (-1)."""
        if self._topics is None:
            return {}
        from collections import Counter
        return {k: v for k, v in Counter(self._topics).items() if k >= 0}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_model(self, path: Optional[str] = None):
        """Save the fitted BERTopic model to disk."""
        if self._model is None:
            logger.warning("save_model: no model to save.")
            return
        save_path = path or str(Path(MODELS_DIR) / "bertopic_model")
        try:
            self._model.save(save_path, serialization="pickle",
                             save_ctfidf=True, save_embedding_model=False)
            logger.info("BERTopic model saved to %s", save_path)
        except Exception as exc:
            logger.error("save_model failed: %s", exc)

    def load_model(self, path: Optional[str] = None):
        """Load a previously saved BERTopic model from disk."""
        load_path = path or str(Path(MODELS_DIR) / "bertopic_model")
        try:
            from bertopic import BERTopic
            self._model = BERTopic.load(load_path)
            self._topic_info = self._model.get_topic_info()
            logger.info("BERTopic model loaded from %s", load_path)
        except Exception as exc:
            logger.error("load_model failed: %s", exc)

    # ── Topic info summary ────────────────────────────────────────────────────

    def get_topic_summary(self) -> List[Dict[str, Any]]:
        """
        Return a list of topic summary dicts for UI display:
            { topic_id, label, top_words, paper_count }
        """
        if self._model is None:
            return []
        labels = self.get_topic_labels()
        counts = self.get_topic_paper_counts()
        summary = []
        for topic_id, label in labels.items():
            if topic_id == -1:
                continue
            summary.append(
                {
                    "topic_id": topic_id,
                    "label": label,
                    "top_words": self.get_top_words(topic_id, n=10),
                    "paper_count": counts.get(topic_id, 0),
                }
            )
        summary.sort(key=lambda x: x["paper_count"], reverse=True)
        return summary

    # ── DB persistence helper ─────────────────────────────────────────────────

    def persist_to_db(self, pmids: List[str], query_used: str, db_manager):
        """
        Store all fitted topics and paper–topic assignments to the database.
        """
        if self._model is None or self._topics is None:
            return

        labels = self.get_topic_labels()
        counts = self.get_topic_paper_counts()

        topic_db_ids: Dict[int, int] = {}
        for topic_id, label in labels.items():
            if topic_id == -1:
                continue
            top_words = self.get_top_words(topic_id, n=10)
            db_id = db_manager.insert_topic(
                topic_label=label,
                top_words=top_words,
                paper_count=counts.get(topic_id, 0),
                query_used=query_used,
            )
            if db_id > 0:
                topic_db_ids[topic_id] = db_id

        assignments = self.get_paper_topic_assignments(pmids)
        for asgn in assignments:
            topic_id = asgn["topic_id"]
            if topic_id == -1:
                continue
            db_id = topic_db_ids.get(topic_id)
            if db_id:
                db_manager.link_paper_topic(
                    asgn["pmid"], db_id, asgn["probability"]
                )

        logger.info(
            "Topics persisted: %d topics, %d assignments",
            len(topic_db_ids), len(assignments),
        )
