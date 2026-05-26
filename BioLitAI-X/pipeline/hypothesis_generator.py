"""
HypothesisGenerator — AI-powered research hypothesis generation using Gemini.

Free-tier design:
  - Default model: gemini-1.5-flash  (15 RPM / 1 500 RPD on free tier)
    Switch to gemini-1.5-pro in .env for higher quality if quota allows.
  - Conservative inter-call delay (4 s for flash, 32 s for pro).
  - Top 5 gaps processed per run (configurable) to stay within daily limits.
  - All generated hypotheses cached to SQLite — skips re-generation on re-run.
  - Exponential backoff starting at 60 s on quota-exceeded (HTTP 429).
  - Concise structured prompts to minimise token usage.
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DailyQuotaError(Exception):
    """
    Raised when all API keys have exhausted their free-tier daily (RPD) quota.
    Unlike per-minute rate limits, daily quota does not recover after a short wait —
    it resets at midnight Pacific Time.
    """

# ── Free-tier constants ────────────────────────────────────────────────────────
# gemini-2.0-flash-lite free tier: 30 RPM  →  1 call / 4 s is safe
# gemini-1.5-pro         free tier:  2 RPM  →  1 call / 32 s is safe
_FLASH_DELAY = 8        # seconds between Flash calls (conservative: ~7.5 RPM)
_PRO_DELAY   = 32       # seconds between Pro calls
_MAX_RETRIES = 3        # network / transient errors
_QUOTA_BACKOFF_BASE = 65  # seconds; doubles on each quota retry (>60s to clear RPM window)
_TOP_GAPS_DEFAULT  = 5    # how many gaps to process per run (free-tier safe)

# ── Model selection ────────────────────────────────────────────────────────────
# gemini-2.0-flash-lite: 30 RPM / 1500 RPD on free tier — most generous flash option
_FLASH_MODEL = "gemini-2.0-flash-lite"
_PRO_MODEL   = "gemini-2.5-pro"

# REST API endpoint — avoids grpc/cffi dependency issues with the Python SDK
_GEMINI_REST_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


def _get_call_delay(model_name: str) -> float:
    return _PRO_DELAY if "pro" in model_name.lower() else _FLASH_DELAY


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_INSTRUCTION = (
    "You are an expert biomedical research scientist specialising in hypothesis "
    "generation from literature evidence. Generate hypotheses grounded strictly in "
    "the provided evidence. Never speculate beyond what the evidence supports. "
    "Always cite specific PMIDs from the provided context. Be precise, concise, "
    "and scientifically rigorous."
)

_HYPOTHESIS_PROMPT_TEMPLATE = """\
Based on the literature evidence below, generate a research hypothesis exploring \
the relationship between "{concept_a}" and "{concept_b}".

EVIDENCE (from PubMed papers):
{evidence_context}

Respond using EXACTLY this numbered structure (one section per line group):

1. HYPOTHESIS
[One sentence: testable scientific claim about {concept_a} and {concept_b}]

2. BIOLOGICAL RATIONALE
[2-3 sentences explaining the mechanistic basis]

3. SUPPORTING EVIDENCE
[Cite 2-4 specific PMIDs and what each supports, format: PMID:XXXXXXXX — finding]

4. SUGGESTED EXPERIMENT
[One concrete experimental approach to test the hypothesis]

5. CONFIDENCE
[Single word: Low | Medium | High]

6. NOVELTY
[One sentence: why this connection is understudied or novel]
"""


class HypothesisGenerator:
    """
    Generates evidence-grounded biomedical hypotheses from knowledge-graph
    gap pairs using the Gemini API.

    All results are stored to the database to avoid redundant API calls on
    subsequent runs — important for free-tier quota management.
    """

    def __init__(self):
        self._api_keys: List[str] = []   # all keys in priority order
        self._key_index: int = 0          # index of currently active key
        self._exhausted: set = set()      # indices of quota-exhausted keys
        self._model_name: str = ""
        self._gen_config: dict = {}
        self._call_delay: float = _FLASH_DELAY
        self._last_call_time: float = 0.0

    @property
    def _api_key(self) -> str:
        """Active API key (first non-exhausted key, or last key if all exhausted)."""
        for i in range(self._key_index, len(self._api_keys)):
            if i not in self._exhausted:
                self._key_index = i
                return self._api_keys[i]
        # All keys exhausted — return the last one so callers get a proper 429
        return self._api_keys[-1] if self._api_keys else ""

    def _rotate_key(self) -> bool:
        """
        Mark the current key as exhausted and advance to the next available one.
        Returns True if a fresh key is available, False if all are exhausted.
        """
        self._exhausted.add(self._key_index)
        for i, key in enumerate(self._api_keys):
            if i not in self._exhausted:
                self._key_index = i
                logger.info(
                    "API key %d/%d exhausted — rotating to key %d.",
                    list(self._exhausted)[-1] + 1, len(self._api_keys), i + 1,
                )
                return True
        logger.warning("All %d API key(s) are quota-exhausted.", len(self._api_keys))
        return False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        """
        Initialise Gemini via the REST API (avoids grpc/cffi SDK issues).

        Reads API keys from environment in order:
          GEMINI_API_KEY, GEMINI_API_KEY_2

        When a key hits its daily quota the app automatically rotates to the
        next one, so adding more free-tier keys (from different Google accounts)
        multiplies the daily request budget.

        Raises EnvironmentError if no valid key is found.
        """
        _placeholder = ("", "your_gemini_api_key_here")
        keys: List[str] = []

        # Collect GEMINI_API_KEY and GEMINI_API_KEY_2
        primary = os.getenv("GEMINI_API_KEY", "")
        if primary.strip() not in _placeholder:
            keys.append(primary.strip())

        idx = 2
        while True:
            k = os.getenv(f"GEMINI_API_KEY_{idx}", "")
            if not k or k.strip() in _placeholder:
                break
            keys.append(k.strip())
            idx += 1

        if not keys:
            raise EnvironmentError(
                "GEMINI_API_KEY not found in .env file.\n"
                "Add your Google Gemini API key:\n"
                "  GEMINI_API_KEY=your_key_here\n\n"
                "For automatic key rotation (doubles your daily quota),\n"
                "add a key from a second Google account:\n"
                "  GEMINI_API_KEY_2=second_key_here\n\n"
                "Free keys: https://aistudio.google.com/app/apikey"
            )

        from config import GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_TOP_P, \
            GEMINI_TOP_K, GEMINI_MAX_OUTPUT_TOKENS

        # Remap deprecated / lower-quota model names to better free-tier equivalents
        requested_model = os.getenv("GEMINI_MODEL", GEMINI_MODEL)
        _deprecated = {
            "gemini-1.5-flash":        "gemini-2.0-flash-lite",
            "gemini-1.5-flash-latest": "gemini-2.0-flash-lite",
            "gemini-1.5-pro":          "gemini-2.5-pro",
            "gemini-1.5-pro-latest":   "gemini-2.5-pro",
            "gemini-pro":              "gemini-2.0-flash-lite",
            "gemini-2.5-flash-lite":   "gemini-2.0-flash-lite",  # lower quota → better model
        }
        if requested_model in _deprecated:
            new_name = _deprecated[requested_model]
            logger.warning("Model '%s' deprecated — using '%s'.", requested_model, new_name)
            requested_model = new_name

        self._api_keys = keys
        self._key_index = 0
        self._exhausted = set()
        self._model_name = requested_model
        self._call_delay = _get_call_delay(self._model_name)
        self._gen_config = {
            "temperature": GEMINI_TEMPERATURE,
            "topP": GEMINI_TOP_P,
            "topK": GEMINI_TOP_K,
            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
        }

        # Verify at least the first key can reach the model
        import requests as _req
        test_url = _GEMINI_REST_URL.format(model=self._model_name)
        try:
            r = _req.post(
                test_url,
                params={"key": self._api_keys[0]},
                json={"contents": [{"parts": [{"text": "ping"}]}]},
                timeout=20,
            )
            if r.status_code == 404:
                raise EnvironmentError(
                    f"Gemini model '{self._model_name}' not found. "
                    "Set GEMINI_MODEL=gemini-2.5-flash-lite in your .env file."
                )
            if r.status_code not in (200, 429):
                err = r.json().get("error", {})
                raise EnvironmentError(
                    f"Gemini API error {r.status_code}: {err.get('message', r.text[:200])}"
                )
        except _req.exceptions.RequestException as exc:
            raise EnvironmentError(f"Cannot reach Gemini API: {exc}") from exc

        logger.info(
            "Gemini REST API ready — model '%s', %d key(s) loaded.",
            self._model_name, len(self._api_keys),
        )

    # ── Rate-limiting ─────────────────────────────────────────────────────────

    def _throttle(self):
        """Enforce minimum inter-call delay for free-tier quota safety."""
        elapsed = time.monotonic() - self._last_call_time
        wait = self._call_delay - elapsed
        if wait > 0:
            logger.debug("Rate-limit wait: %.1f s", wait)
            time.sleep(wait)
        self._last_call_time = time.monotonic()

    # ── Evidence context builder ──────────────────────────────────────────────

    def build_evidence_context(
        self,
        gap_pair: Dict[str, Any],
        papers_df,
        embedder=None,
        max_papers: int = 5,
    ) -> str:
        """
        Build a concise evidence string for the hypothesis prompt.

        Uses FAISS semantic search (if embedder is provided) to find the most
        relevant papers for each concept; falls back to keyword matching.
        Limits to *max_papers* per concept to keep prompts token-efficient.
        """
        concept_a = str(gap_pair.get("concept_a", ""))
        concept_b = str(gap_pair.get("concept_b", "") or "")
        evidence_pmids = gap_pair.get("evidence_pmids", [])

        if papers_df is None or (hasattr(papers_df, "empty") and papers_df.empty):
            return "No paper evidence available."

        context_parts: List[str] = []
        seen_pmids: set = set()

        def _add_paper(row, label: str):
            pmid = str(row.get("pmid", "") if isinstance(row, dict)
                       else getattr(row, "pmid", ""))
            if pmid in seen_pmids:
                return
            seen_pmids.add(pmid)
            title = str((row.get("title", "") if isinstance(row, dict)
                     else getattr(row, "title", "")) or "")
            abstract = str((row.get("abstract", "") if isinstance(row, dict)
                        else getattr(row, "abstract", "")) or "")
            snippet = abstract[:300].replace("\n", " ").strip()
            context_parts.append(
                f"[PMID:{pmid} | {label}]\n"
                f"Title: {title}\n"
                f"Evidence: {snippet}…"
            )

        # 1 — Papers already identified as gap evidence
        pmid_set = set(str(p) for p in evidence_pmids)
        for _, row in papers_df.iterrows():
            pmid = str(row.get("pmid", ""))
            if pmid in pmid_set and len(context_parts) < max_papers:
                _add_paper(row.to_dict(), "gap evidence")

        # 2 — Semantic search per concept (if embedder available)
        if embedder is not None and embedder.is_ready:
            for concept, label in [(concept_a, "concept A"), (concept_b, "concept B")]:
                if not concept:
                    continue
                hits = embedder.semantic_search(concept, top_k=max_papers)
                hit_pmids = {h["pmid"] for h in hits}
                for _, row in papers_df.iterrows():
                    if len(context_parts) >= max_papers * 2:
                        break
                    if str(row.get("pmid", "")) in hit_pmids:
                        _add_paper(row.to_dict(), label)
        else:
            # Fallback: split concept into tokens, match any word >= 3 chars
            for concept, label in [(concept_a, "concept A"), (concept_b, "concept B")]:
                if not concept:
                    continue
                tokens = [t for t in re.split(r'\W+', concept.lower()) if len(t) >= 3]
                if not tokens:
                    tokens = [concept.lower()]
                count = 0
                for _, row in papers_df.iterrows():
                    if count >= max_papers:
                        break
                    title = str(row.get("title", "")).lower()
                    abstract = str(row.get("abstract", "")).lower()
                    combined = title + " " + abstract
                    if any(tok in combined for tok in tokens):
                        _add_paper(row.to_dict(), label)
                        count += 1

        return "\n\n".join(context_parts) if context_parts else (
            f"No direct evidence found for '{concept_a}' / '{concept_b}'."
        )

    # ── Single hypothesis generation ──────────────────────────────────────────

    def generate_hypothesis(
        self,
        gap_pair: Dict[str, Any],
        evidence_context: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Call Gemini to generate a structured hypothesis for one gap pair.
        Returns a parsed hypothesis dict, or None on unrecoverable failure.
        """
        if not self._api_key:
            raise RuntimeError("Call setup() before generate_hypothesis().")

        concept_a = str(gap_pair.get("concept_a", ""))
        concept_b = str(gap_pair.get("concept_b", "") or "")

        prompt = _HYPOTHESIS_PROMPT_TEMPLATE.format(
            concept_a=concept_a,
            concept_b=concept_b,
            evidence_context=evidence_context[:3000],  # cap tokens
        )

        raw_text = self._call_with_retry(prompt)
        if raw_text == "__DAILY_QUOTA_EXCEEDED__":
            raise DailyQuotaError(
                "All Gemini API keys have exhausted their free-tier daily quota (RPD). "
                "Quota resets at midnight Pacific Time (00:00 PT). "
                "Please try again tomorrow, or add API keys from additional Google accounts."
            )
        if not raw_text:
            return None

        parsed = self._parse_hypothesis_response(raw_text)
        if not parsed:
            logger.warning(
                "Could not parse Gemini response for %s/%s — skipping.",
                concept_a, concept_b,
            )
            return None

        evidence_pmids = list(
            set(gap_pair.get("evidence_pmids", []))
            | set(parsed.get("cited_pmids", []))
        )

        return {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "hypothesis_text": parsed.get("hypothesis", ""),
            "rationale": parsed.get("rationale", ""),
            "supporting_evidence": parsed.get("supporting_evidence", ""),
            "suggested_experiment": parsed.get("suggested_experiment", ""),
            "confidence_score": _confidence_to_float(parsed.get("confidence", "Low")),
            "novelty": parsed.get("novelty", ""),
            "evidence_pmids": evidence_pmids,
            "raw_response": raw_text,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Batch hypothesis generation ───────────────────────────────────────────

    def generate_batch_hypotheses(
        self,
        gap_report: List[Dict[str, Any]],
        papers_df,
        db_manager=None,
        query_used: str = "",
        embedder=None,
        top_n: int = _TOP_GAPS_DEFAULT,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        Process the top *top_n* gaps from *gap_report* sequentially.

        Free-tier safe:
          - Skips gaps whose hypothesis already exists in the database.
          - Enforces inter-call delay via _throttle().
          - Processes at most *top_n* gaps (default 5).

        Returns hypotheses sorted by confidence descending.
        """
        if not self._api_key:
            raise RuntimeError("Call setup() before generate_batch_hypotheses().")

        # Filter to gaps with two distinct concepts (skips temporal gaps where concept_b is None)
        candidates = [
            g for g in gap_report
            if g.get("concept_a") and g.get("concept_b")
               and g["concept_a"] != g.get("concept_b")
        ]
        if not candidates:
            logger.warning(
                "generate_batch_hypotheses: all %d gaps are single-concept "
                "(temporal) gaps with no concept_b — cannot generate hypotheses.",
                len(gap_report),
            )
            return []
        candidates = candidates[:top_n]

        hypotheses: List[Dict[str, Any]] = []
        total = len(candidates)

        for i, gap in enumerate(candidates):
            try:
                concept_a = str(gap.get("concept_a") or "")
                concept_b = str(gap.get("concept_b") or "")

                # Cache check — skip if already in DB
                if db_manager is not None:
                    existing = _find_existing_hypothesis(
                        db_manager, concept_a, concept_b, query_used
                    )
                    if existing:
                        logger.info(
                            "Hypothesis for %s/%s already in DB — skipping.",
                            concept_a, concept_b,
                        )
                        hypotheses.append(existing)
                        if progress_callback:
                            progress_callback(i + 1, total, "cached")
                        continue

                logger.info(
                    "Generating hypothesis %d/%d: %s ↔ %s",
                    i + 1, total, concept_a, concept_b,
                )

                evidence = self.build_evidence_context(
                    gap, papers_df, embedder=embedder
                )
                hyp = self.generate_hypothesis(gap, evidence)

                if hyp:
                    hyp["query_used"] = query_used
                    if db_manager is not None:
                        hyp_id = db_manager.insert_hypothesis(hyp)
                        hyp["id"] = hyp_id
                    hypotheses.append(hyp)

            except DailyQuotaError:
                # Daily quota wall — no point trying remaining gaps, re-raise immediately
                logger.error(
                    "Daily quota exhausted at gap %d/%d — aborting batch.", i + 1, total
                )
                raise
            except Exception as gap_exc:
                import traceback as _tb
                logger.error(
                    "Gap %d/%d failed: %s\n%s",
                    i + 1, total, gap_exc, _tb.format_exc()
                )
                # Continue processing remaining gaps

            if progress_callback:
                progress_callback(i + 1, total, "generated" if (len(hypotheses) > i) else "failed")

        hypotheses.sort(key=lambda h: h.get("confidence_score", 0.0), reverse=True)
        logger.info(
            "Batch complete: %d hypotheses generated from %d gaps.",
            len(hypotheses), total,
        )
        return hypotheses

    # ── Literature chat ───────────────────────────────────────────────────────

    def chat_about_literature(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        papers_df,
        embedder=None,
    ) -> Dict[str, Any]:
        """
        Answer a user question grounded strictly in the fetched paper set.

        Returns:
          { response_text, source_pmids, source_titles }

        Never allows Gemini to answer from general knowledge outside the
        retrieved papers.  Uses LangChain memory for multi-turn context.
        """
        if not self._api_key:
            raise RuntimeError("Call setup() before chat_about_literature().")

        # Retrieve relevant papers
        context_papers = self._retrieve_context_papers(
            user_message, papers_df, embedder, top_k=5
        )

        paper_context = self._format_paper_context(context_papers)
        source_pmids = [str(p.get("pmid", "")) for p in context_papers]
        source_titles = [str(p.get("title", "")) for p in context_papers]

        # Build history string (last 4 turns max to keep tokens low)
        history_str = ""
        for turn in conversation_history[-4:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")[:400]
            history_str += f"{role.upper()}: {content}\n"

        grounded_prompt = (
            "You are a biomedical research assistant. "
            "Answer using ONLY the evidence from the papers provided below. "
            "If the answer cannot be found in these papers, say clearly: "
            "'This information is not available in the current paper set.' "
            "Always cite PMIDs in your answer.\n\n"
            f"RETRIEVED PAPERS:\n{paper_context}\n\n"
            + (f"CONVERSATION HISTORY:\n{history_str}\n\n"
               if history_str else "")
            + f"USER QUESTION: {user_message}"
        )

        response_text = self._call_with_retry(grounded_prompt, quick_fail=True)
        if response_text == "__QUOTA_EXCEEDED__":
            response_text = (
                "**Gemini free-tier quota exceeded.** "
                "The free tier allows only a limited number of requests per day "
                "(typically 50 RPD for gemini-2.5-flash-lite). "
                "Please wait until your quota resets (usually midnight Pacific Time) "
                "or upgrade your API key at https://aistudio.google.com/app/apikey"
            )
        elif not response_text:
            response_text = (
                "I was unable to generate a response at this time. "
                "Please check your internet connection and try again."
            )

        return {
            "response_text": response_text,
            "source_pmids": source_pmids,
            "source_titles": source_titles,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call_with_retry(self, prompt: str, quick_fail: bool = False) -> Optional[str]:
        """
        Call Gemini REST API with throttling + exponential backoff.

        quick_fail=True: return None immediately on quota errors (for interactive
        chat) rather than waiting 60-120 s per retry.

        Returns the response text or None on permanent failure.
        """
        import requests as _req

        url = _GEMINI_REST_URL.format(model=self._model_name)
        payload = {
            "systemInstruction": {"parts": [{"text": _SYSTEM_INSTRUCTION}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": self._gen_config,
        }

        backoff = _QUOTA_BACKOFF_BASE
        for attempt in range(_MAX_RETRIES):
            self._throttle()
            try:
                r = _req.post(
                    url,
                    params={"key": self._api_key},
                    json=payload,
                    timeout=60,
                )
            except _req.exceptions.RequestException as exc:
                delay = 2 ** attempt * 2
                logger.warning(
                    "Gemini network error (attempt %d/%d): %s — retry in %ds",
                    attempt + 1, _MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
                continue

            if r.status_code == 200:
                data = r.json()
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError) as exc:
                    logger.warning("Unexpected Gemini response shape: %s", data)
                    return None

            err = r.json().get("error", {})
            err_msg = err.get("message", "")
            if r.status_code == 429 or "resource_exhausted" in err_msg.lower():
                # Detect daily quota (RPD) vs per-minute rate limit (RPM).
                # Daily quota: resets at midnight PT — no point retrying.
                # RPM: resets in ~60 s — worth waiting and retrying.
                #
                # Use Google's structured `details` array which contains a
                # `quota_limit` metadata key — e.g.
                #   "requests_per_minute_per_project"  → RPM
                #   "requests_per_day_per_project"     → RPD (daily)
                # This is far more reliable than parsing the human-readable
                # `message` string, which may contain the word "minute" even
                # for daily-quota errors (e.g. "retry after a few minutes").
                _quota_limit = ""
                for _detail in err.get("details", []):
                    _meta = _detail.get("metadata", {})
                    if "quota_limit" in _meta:
                        _quota_limit = _meta["quota_limit"].lower()
                        break

                if _quota_limit:
                    # Authoritative: use the structured quota_limit field.
                    _is_rpm = "per_minute" in _quota_limit or "minute" in _quota_limit
                    _is_daily = "per_day" in _quota_limit or "_per_1_day" in _quota_limit
                    # If quota_limit exists but matches neither pattern, treat
                    # as daily (safe default — avoids infinite retry loops).
                    if not _is_rpm and not _is_daily:
                        _is_daily = True
                else:
                    # Fallback: no details field — infer from message text.
                    # Only treat as RPM if the message explicitly names a
                    # per-minute / rate-limit concept; default to daily.
                    _err_lower = err_msg.lower()
                    _is_rpm = any(k in _err_lower for k in (
                        "per_minute", "per minute", "rpm",
                        "rate_limit", "rate limit", "rate-limit",
                    ))
                    _is_daily = not _is_rpm

                # Try rotating to the next key before waiting
                if self._rotate_key():
                    logger.info("Rotated to next API key — retrying immediately.")
                    continue   # retry the loop immediately with the new key

                # All keys exhausted — check if it's a daily quota wall
                if _is_daily:
                    logger.error(
                        "Daily quota (RPD) exhausted on all %d key(s). "
                        "Free-tier quota resets at midnight Pacific Time.",
                        len(self._api_keys),
                    )
                    return "__DAILY_QUOTA_EXCEEDED__"

                if quick_fail:
                    logger.warning("All API keys quota-exhausted — failing fast.")
                    return "__QUOTA_EXCEEDED__"

                logger.warning(
                    "All keys exhausted (attempt %d/%d). Waiting %ds for quota to recover.",
                    attempt + 1, _MAX_RETRIES, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                # Reset exhausted flags after sleeping so all keys get a fresh
                # chance — the backoff wait is specifically to let RPM quota recover.
                self._exhausted.clear()
                self._key_index = 0
                logger.info("Quota backoff complete — retrying all keys.")
            elif r.status_code in (400, 403):
                logger.warning("Gemini blocked/invalid (%d): %s", r.status_code, err_msg)
                return None
            elif r.status_code == 404:
                logger.error(
                    "Gemini model '%s' not found (404). "
                    "Set GEMINI_MODEL=gemini-2.5-flash-lite in .env.",
                    self._model_name,
                )
                return None
            else:
                delay = 2 ** attempt * 2
                logger.warning(
                    "Gemini HTTP %d (attempt %d/%d): %s — retry in %ds",
                    r.status_code, attempt + 1, _MAX_RETRIES, err_msg[:100], delay,
                )
                time.sleep(delay)

        logger.error("Gemini call permanently failed after %d attempts.", _MAX_RETRIES)
        return None

    @staticmethod
    def _parse_hypothesis_response(text: str) -> Optional[Dict[str, str]]:
        """
        Parse the structured numbered response from Gemini.
        Tolerant of minor formatting variations.
        """
        if not text:
            return None

        sections = {
            "hypothesis": "",
            "rationale": "",
            "supporting_evidence": "",
            "suggested_experiment": "",
            "confidence": "Low",
            "novelty": "",
            "cited_pmids": [],
        }

        patterns = [
            ("hypothesis",          r"1\.\s*HYPOTHESIS[:\s]+(.*?)(?=2\.|$)"),
            ("rationale",           r"2\.\s*BIOLOGICAL RATIONALE[:\s]+(.*?)(?=3\.|$)"),
            ("supporting_evidence", r"3\.\s*SUPPORTING EVIDENCE[:\s]+(.*?)(?=4\.|$)"),
            ("suggested_experiment",r"4\.\s*SUGGESTED EXPERIMENT[:\s]+(.*?)(?=5\.|$)"),
            ("confidence",          r"5\.\s*CONFIDENCE[:\s]+(.*?)(?=6\.|$)"),
            ("novelty",             r"6\.\s*NOVELTY[:\s]+(.*?)$"),
        ]

        for key, pattern in patterns:
            m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if m:
                sections[key] = m.group(1).strip()

        # Extract cited PMIDs from supporting evidence section
        pmid_matches = re.findall(
            r"PMID[:\s]*(\d{5,8})", sections["supporting_evidence"], re.IGNORECASE
        )
        sections["cited_pmids"] = list(set(pmid_matches))

        # Normalise confidence to one of Low / Medium / High
        conf_raw = sections["confidence"].lower()
        if "high" in conf_raw:
            sections["confidence"] = "High"
        elif "medium" in conf_raw or "moderate" in conf_raw:
            sections["confidence"] = "Medium"
        else:
            sections["confidence"] = "Low"

        # Require at minimum a hypothesis text
        if not sections["hypothesis"]:
            return None

        return sections

    def _retrieve_context_papers(
        self,
        query: str,
        papers_df,
        embedder=None,
        top_k: int = 5,
    ) -> List[Dict]:
        """Return up to *top_k* most relevant papers for a query string."""
        if papers_df is None or (hasattr(papers_df, "empty") and papers_df.empty):
            return []

        if embedder is not None and embedder.is_ready:
            hits = embedder.semantic_search(query, top_k=top_k)
            hit_pmids = {h["pmid"] for h in hits}
            rows = []
            for _, row in papers_df.iterrows():
                if str(row.get("pmid", "")) in hit_pmids:
                    rows.append(row.to_dict())
                if len(rows) >= top_k:
                    break
            return rows
        else:
            # Keyword fallback: split query into tokens, match any word >= 4 chars
            tokens = [t for t in re.split(r'\W+', query.lower()) if len(t) >= 4]
            if not tokens:
                tokens = [query.lower()[:20]]
            rows = []
            for _, row in papers_df.iterrows():
                title = str(row.get("title", "")).lower()
                abstract = str(row.get("abstract", "")).lower()
                combined = title + " " + abstract
                if any(tok in combined for tok in tokens):
                    rows.append(row.to_dict())
                if len(rows) >= top_k:
                    break
            return rows

    @staticmethod
    def _format_paper_context(papers: List[Dict]) -> str:
        parts = []
        for p in papers:
            pmid = p.get("pmid", "")
            title = p.get("title", "")
            abstract = str(p.get("abstract", "") or "")[:400]
            parts.append(
                f"PMID:{pmid} | {title}\n{abstract}…"
            )
        return "\n\n".join(parts) if parts else "No papers retrieved."


# ── Module-level helpers ──────────────────────────────────────────────────────

def _confidence_to_float(label: str) -> float:
    return {"High": 0.85, "Medium": 0.55, "Low": 0.25}.get(label, 0.25)


def _find_existing_hypothesis(
    db_manager, concept_a: str, concept_b: str, query_used: str
) -> Optional[Dict]:
    """Check the DB for a cached hypothesis matching this concept pair."""
    try:
        existing = db_manager.get_hypotheses_by_query(query_used)
        for h in existing:
            if (h.get("concept_a", "").lower() == concept_a.lower()
                    and h.get("concept_b", "").lower() == concept_b.lower()):
                return h
    except Exception as exc:
        logger.warning("Cache check failed: %s", exc)
    return None
