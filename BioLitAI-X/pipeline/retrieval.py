"""
PubMedRetriever — fetches PMIDs and raw XML records from NCBI Entrez.
Implements rate limiting, batch fetching, and progress tracking.
"""

import logging
import time
from typing import Callable, List, Optional

from Bio import Entrez

from config import (
    ENTREZ_EMAIL,
    ENTREZ_API_KEY,
    ENTREZ_BATCH_SIZE,
    ENTREZ_RATE_LIMIT_NO_KEY,
    ENTREZ_RATE_LIMIT_WITH_KEY,
    MAX_RESULTS_DEFAULT,
)

logger = logging.getLogger(__name__)


class PubMedRetriever:
    """Retrieves PubMed records for any user-supplied biomedical query."""

    def __init__(self):
        if not ENTREZ_EMAIL:
            raise EnvironmentError(
                "ENTREZ_EMAIL not found in .env file. "
                "Please add ENTREZ_EMAIL=your_email@example.com to your .env file."
            )
        Entrez.email = ENTREZ_EMAIL
        if ENTREZ_API_KEY:
            Entrez.api_key = ENTREZ_API_KEY
            self._min_interval = 1.0 / ENTREZ_RATE_LIMIT_WITH_KEY
        else:
            self._min_interval = 1.0 / ENTREZ_RATE_LIMIT_NO_KEY
        self._last_request_time: float = 0.0

    # ── Rate limiting ─────────────────────────────────────────────────────────

    def _throttle(self):
        elapsed = time.monotonic() - self._last_request_time
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.monotonic()

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str,
               max_results: int = MAX_RESULTS_DEFAULT) -> List[str]:
        """
        Search PubMed for *query* and return up to *max_results* PMIDs.
        The search is entirely driven by the query string — no domain
        assumptions are made.
        """
        logger.info("Searching PubMed: query=%r max_results=%d", query, max_results)
        pmids: List[str] = []
        try:
            self._throttle()
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                usehistory="y",
            )
            record = Entrez.read(handle)
            handle.close()
            pmids = record.get("IdList", [])
            total = int(record.get("Count", 0))
            logger.info(
                "PubMed search returned %d PMIDs (total available: %d)",
                len(pmids), total,
            )
        except Exception as exc:
            logger.error("PubMed search failed for query=%r: %s", query, exc)
        return pmids

    def fetch_records(self, pmids: List[str]) -> str:
        """
        Fetch full PubMed XML for a list of PMIDs in one call.
        Caller is responsible for splitting into batches if needed.
        Returns raw XML string.
        """
        if not pmids:
            return ""
        ids = ",".join(pmids)
        try:
            self._throttle()
            handle = Entrez.efetch(
                db="pubmed",
                id=ids,
                rettype="xml",
                retmode="xml",
            )
            xml_data = handle.read()
            handle.close()
            if isinstance(xml_data, bytes):
                xml_data = xml_data.decode("utf-8", errors="replace")
            return xml_data
        except Exception as exc:
            logger.error("fetch_records failed for %d PMIDs: %s", len(pmids), exc)
            return ""

    def fetch_with_progress(
        self,
        query: str,
        max_results: int = MAX_RESULTS_DEFAULT,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        """
        Full pipeline: search → batch-fetch all records → return list of XML
        strings (one per batch).

        *progress_callback(fetched, total)* is called after each batch so the
        UI can update a progress bar.  Pass None to disable.
        """
        pmids = self.search(query, max_results)
        if not pmids:
            logger.warning("No PMIDs found for query=%r", query)
            return []

        total = len(pmids)
        xml_batches: List[str] = []
        fetched = 0

        for batch_start in range(0, total, ENTREZ_BATCH_SIZE):
            batch = pmids[batch_start: batch_start + ENTREZ_BATCH_SIZE]
            xml = self._fetch_batch_with_retry(batch)
            if xml:
                xml_batches.append(xml)
            fetched += len(batch)
            if progress_callback:
                try:
                    progress_callback(fetched, total)
                except Exception:
                    pass
            logger.debug("Fetched %d / %d records", fetched, total)

        return xml_batches

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_batch_with_retry(self, pmids: List[str],
                                max_retries: int = 3) -> str:
        """Fetch a single batch with exponential-backoff retry on failure."""
        delay = 2.0
        for attempt in range(max_retries):
            xml = self.fetch_records(pmids)
            if xml:
                return xml
            if attempt < max_retries - 1:
                logger.warning(
                    "Batch fetch failed (attempt %d/%d), retrying in %.1fs …",
                    attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
                delay *= 2
        logger.error(
            "Batch fetch permanently failed after %d attempts for PMIDs: %s",
            max_retries, pmids[:5],
        )
        return ""
