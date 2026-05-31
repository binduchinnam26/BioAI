"""
EmbeddingEngine — dense vector embeddings + FAISS semantic search.

Model: pritamdeka/S-PubMedBert-MS-MARCO
Index: FAISS IndexFlatIP (cosine similarity via L2-normalised vectors)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDINGS_DIR,
    SEMANTIC_SEARCH_TOP_K,
)
from utils.helpers import query_hash

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Embeds a corpus of paper abstracts with a biomedical sentence transformer
    and stores a FAISS index for fast semantic retrieval.

    A separate index is maintained per query (keyed by query hash) so that
    multiple sessions can coexist without interference.
    """

    def __init__(self):
        self._model = None
        self._index: Optional[Any] = None   # faiss.Index
        self._pmid_list: List[str] = []     # positional mapping index → PMID
        self._current_query_hash: Optional[str] = None

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        logger.info("Loading embedding model: %s …", EMBEDDING_MODEL)
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")

    # ── Corpus embedding ──────────────────────────────────────────────────────

    def embed_corpus(
        self,
        papers_df,
        query: str = "",
        progress_callback=None,
    ) -> np.ndarray:
        """
        Embed all abstracts in *papers_df*.

        - Builds a FAISS IndexFlatIP (inner-product on L2-normalised vectors
          equals cosine similarity).
        - Saves the index as  data/embeddings/<query_hash>.faiss
        - Saves the PMID mapping as  data/embeddings/<query_hash>_pmids.json

        Returns the embeddings array of shape (N, D).
        """
        import pandas as pd
        self._load_model()

        if papers_df is None or (hasattr(papers_df, "empty") and papers_df.empty):
            logger.warning("embed_corpus: empty DataFrame")
            return np.array([])

        pmids = papers_df["pmid"].astype(str).tolist()
        abstracts = papers_df["abstract"].fillna("").tolist()

        # Replace empty abstracts with title so we still have something to embed
        titles = papers_df["title"].fillna("").tolist()
        texts = [
            ab if ab.strip() else ti
            for ab, ti in zip(abstracts, titles)
        ]

        logger.info("Embedding %d texts in batches of %d …",
                    len(texts), EMBEDDING_BATCH_SIZE)

        all_embeddings: List[np.ndarray] = []
        total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        for batch_idx in range(total_batches):
            start = batch_idx * EMBEDDING_BATCH_SIZE
            end = start + EMBEDDING_BATCH_SIZE
            batch_texts = texts[start:end]

            try:
                batch_emb = self._model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,  # L2 normalise for cosine via IP
                )
                all_embeddings.append(batch_emb.astype(np.float32))
            except Exception as exc:
                logger.error("Embedding batch %d failed: %s", batch_idx, exc)
                # Insert zeros so positional index stays aligned
                dim = all_embeddings[0].shape[1] if all_embeddings else 768
                zeros = np.zeros((len(batch_texts), dim), dtype=np.float32)
                all_embeddings.append(zeros)

            if progress_callback:
                try:
                    progress_callback(min(end, len(texts)), len(texts))
                except Exception:
                    pass

        embeddings = np.vstack(all_embeddings)  # (N, D)

        # ── Build FAISS index ─────────────────────────────────────────────────
        self._build_index(embeddings, pmids, query)

        logger.info("Embedding complete: shape=%s", embeddings.shape)
        return embeddings

    def _build_index(
        self, embeddings: np.ndarray, pmids: List[str], query: str
    ):
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            ) from exc

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self._index = index
        self._pmid_list = pmids
        self._current_query_hash = query_hash(query) if query else "default"

        # Persist
        qhash = self._current_query_hash
        index_path = Path(EMBEDDINGS_DIR) / f"{qhash}.faiss"
        pmid_path = Path(EMBEDDINGS_DIR) / f"{qhash}_pmids.json"

        try:
            faiss.write_index(index, str(index_path))
            with open(pmid_path, "w", encoding="utf-8") as f:
                json.dump(pmids, f)
            logger.info("FAISS index saved to %s", index_path)
        except Exception as exc:
            logger.warning("Could not persist FAISS index: %s", exc)

    # ── Load existing index ───────────────────────────────────────────────────

    def load_index(self, query: str) -> bool:
        """
        Load a previously saved FAISS index for *query*.
        Returns True if successful, False if no saved index exists.
        """
        qhash = query_hash(query)
        index_path = Path(EMBEDDINGS_DIR) / f"{qhash}.faiss"
        pmid_path = Path(EMBEDDINGS_DIR) / f"{qhash}_pmids.json"

        if not index_path.exists() or not pmid_path.exists():
            return False

        try:
            import faiss
            self._index = faiss.read_index(str(index_path))
            with open(pmid_path, "r", encoding="utf-8") as f:
                self._pmid_list = json.load(f)
            self._current_query_hash = qhash
            logger.info("Loaded FAISS index for query hash %s (%d vectors)",
                        qhash, self._index.ntotal)
            return True
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)
            return False

    # ── Semantic search ───────────────────────────────────────────────────────

    def semantic_search(
        self,
        query_text: str,
        top_k: int = SEMANTIC_SEARCH_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Embed *query_text* at runtime and return the top-k most similar papers
        from the currently loaded index.

        Returns a list of dicts:
          { pmid, score, rank }
        sorted by score descending.
        """
        if self._index is None:
            logger.warning("semantic_search called but no FAISS index is loaded.")
            return []

        self._load_model()

        try:
            query_emb = self._model.encode(
                [query_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(query_emb, k)

            results = []
            for rank, (idx, score) in enumerate(
                zip(indices[0], scores[0]), start=1
            ):
                if idx < 0 or idx >= len(self._pmid_list):
                    continue
                results.append(
                    {
                        "pmid": self._pmid_list[idx],
                        "score": float(score),
                        "rank": rank,
                    }
                )
            return results

        except Exception as exc:
            logger.error("semantic_search failed: %s", exc)
            return []

    def semantic_search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = SEMANTIC_SEARCH_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Search using a pre-computed embedding vector (shape (D,) or (1, D)).
        Used internally by the hypothesis generator.
        """
        if self._index is None:
            return []
        try:
            import faiss
            q = query_embedding.astype(np.float32)
            if q.ndim == 1:
                q = q[np.newaxis, :]
            faiss.normalize_L2(q)
            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(q, k)
            results = []
            for rank, (idx, score) in enumerate(
                zip(indices[0], scores[0]), start=1
            ):
                if idx < 0 or idx >= len(self._pmid_list):
                    continue
                results.append(
                    {"pmid": self._pmid_list[idx],
                     "score": float(score),
                     "rank": rank}
                )
            return results
        except Exception as exc:
            logger.error("semantic_search_by_embedding failed: %s", exc)
            return []

    # ── Encode a single text ──────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string and return a normalised embedding."""
        self._load_model()
        return self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)[0]

    # ── Index introspection ───────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def index_size(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def pmid_list(self) -> List[str]:
        return list(self._pmid_list)

    def list_available_indices(self) -> List[str]:
        """Return query hashes for which saved FAISS indices exist."""
        return [
            p.stem
            for p in Path(EMBEDDINGS_DIR).glob("*.faiss")
        ]
