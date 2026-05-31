"""
DataCleaner — normalises, deduplicates, and structures raw PubMed records.

Fully query-agnostic: no disease, gene, or domain names are referenced anywhere
in this module.  All logic operates on whatever text the data contains.
"""

import logging
import re
import unicodedata
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from thefuzz import fuzz

logger = logging.getLogger(__name__)

# ── Boilerplate phrases stripped from abstracts ────────────────────────────────
_BOILERPLATE_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"copyright\s+©?\s*\d{4}.*",
        r"all rights reserved\.?",
        r"published by elsevier.*",
        r"this article is protected by copyright.*",
        r"©\s*\d{4}.*",
        r"this is an open.access article.*",
        r"for correspondence.*",
        r"electronic supplementary material.*",
    ]
]

# Month strings to zero-padded integers
_MONTH_MAP = {
    "january": "01",  "jan": "01",
    "february": "02", "feb": "02",
    "march": "03",    "mar": "03",
    "april": "04",    "apr": "04",
    "may": "05",
    "june": "06",     "jun": "06",
    "july": "07",     "jul": "07",
    "august": "08",   "aug": "08",
    "september": "09","sep": "09",
    "october": "10",  "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}


class DataCleaner:
    """
    Cleans and normalises a list of raw paper dicts produced by XMLParser,
    returning a pandas DataFrame ready for NLP and graph construction.
    """

    # ── Author normalisation ───────────────────────────────────────────────────

    def normalize_author_name(self, name: str) -> str:
        """
        Return a canonical "Lastname, Firstname" string.
        Handles "Lastname, Firstname", "Firstname Lastname", plain lastnames,
        and unicode characters.
        """
        if not name or not isinstance(name, str):
            return ""
        # Normalise unicode to ASCII where possible (accents → base chars)
        name = unicodedata.normalize("NFD", name)
        name = "".join(c for c in name if unicodedata.category(c) != "Mn")
        name = name.strip()

        if "," in name:
            # Already "Lastname, Fore" format
            parts = [p.strip() for p in name.split(",", 1)]
            last = self._title_case(parts[0])
            fore = self._title_case(parts[1]) if len(parts) > 1 else ""
            return f"{last}, {fore}".strip(", ") if fore else last
        else:
            # "Firstname Lastname" or single token
            tokens = name.split()
            if len(tokens) >= 2:
                last = self._title_case(tokens[-1])
                fore = self._title_case(" ".join(tokens[:-1]))
                return f"{last}, {fore}"
            return self._title_case(name)

    @staticmethod
    def _title_case(text: str) -> str:
        if not text:
            return ""
        return " ".join(w.capitalize() for w in text.split())

    def deduplicate_authors(
        self, author_list: List[str]
    ) -> Dict[str, str]:
        """
        Cluster author names with thefuzz token_sort_ratio >= 90 OR by
        last-name + initial-prefix agreement, and return a mapping
        {original_name: canonical_name}.
        The canonical form is the longest variant in each cluster.
        """
        if not author_list:
            return {}
        normalised = [self.normalize_author_name(a) for a in author_list]
        clusters: List[List[int]] = []
        assigned = set()

        for i, name_i in enumerate(normalised):
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j, name_j in enumerate(normalised):
                if j in assigned:
                    continue
                if self._author_names_match(name_i, name_j):
                    cluster.append(j)
                    assigned.add(j)
            clusters.append(cluster)

        mapping: Dict[str, str] = {}
        for cluster in clusters:
            # Canonical = longest normalised name (most complete)
            canonical = max((normalised[idx] for idx in cluster), key=len)
            for idx in cluster:
                mapping[author_list[idx]] = canonical
        return mapping

    @staticmethod
    def _author_names_match(name_a: str, name_b: str) -> bool:
        """
        Two author names are considered the same person if either:
        (a) token_sort_ratio >= 90, OR
        (b) their last names are identical AND the shorter forename is a
            prefix (or single initial) of the longer forename.
        """
        if fuzz.token_sort_ratio(name_a, name_b) >= 90:
            return True
        # Initial / abbreviated forename check
        def _split(n):
            parts = n.split(",", 1)
            last = parts[0].strip().lower()
            fore = parts[1].strip().lower() if len(parts) > 1 else ""
            return last, fore

        last_a, fore_a = _split(name_a)
        last_b, fore_b = _split(name_b)
        if last_a != last_b:
            return False
        if not fore_a or not fore_b:
            return False
        # One forename is a single initial that is the first letter of the other
        short, long = (fore_a, fore_b) if len(fore_a) <= len(fore_b) \
                      else (fore_b, fore_a)
        if len(short) == 1 and long.startswith(short):
            return True
        # One forename starts with the other (handles "J" vs "John" vs "John A")
        if long.startswith(short):
            return True
        return False

    # ── Keyword normalisation ─────────────────────────────────────────────────

    def normalize_keyword(self, keyword: str) -> str:
        """
        Lowercase, strip punctuation noise, collapse whitespace.
        MeSH preferred-term mapping is handled at database-insert time
        by the network builder; this method returns a clean surface form.
        """
        if not keyword or not isinstance(keyword, str):
            return ""
        kw = unicodedata.normalize("NFC", keyword)
        kw = kw.lower().strip()
        # Remove leading/trailing punctuation except hyphens inside words
        kw = re.sub(r"^[^\w]+|[^\w]+$", "", kw)
        kw = re.sub(r"\s+", " ", kw)
        return kw

    # ── Abstract cleaning ─────────────────────────────────────────────────────

    def clean_abstract(self, text: str) -> Optional[str]:
        """
        Strip HTML tags, normalise unicode, remove publisher boilerplate,
        and collapse whitespace.
        """
        if not text or not isinstance(text, str):
            return None
        # Strip HTML
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode HTML entities
        text = text.replace("&amp;", "&").replace("&lt;", "<") \
                   .replace("&gt;", ">").replace("&quot;", '"') \
                   .replace("&#39;", "'").replace("&nbsp;", " ")
        # Normalise unicode
        text = unicodedata.normalize("NFC", text)
        # Remove boilerplate at end of abstract
        for pattern in _BOILERPLATE_PATTERNS:
            text = pattern.sub("", text)
        # Collapse whitespace / newlines
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = "\n".join(line.strip() for line in text.splitlines())
        text = text.strip()
        return text if text else None

    # ── Duplicate detection ────────────────────────────────────────────────────

    def detect_duplicates(self, papers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate by:
        1. Exact PMID match (keep first occurrence)
        2. Title similarity >= 95 via thefuzz token_sort_ratio (keep first)

        Returns the deduplicated DataFrame with a reset index.
        """
        if papers_df is None or papers_df.empty:
            return papers_df

        initial_count = len(papers_df)

        # 1 — exact PMID dedup
        papers_df = papers_df.drop_duplicates(subset=["pmid"], keep="first")
        after_pmid = len(papers_df)

        # 2 — title similarity dedup
        titles = papers_df["title"].fillna("").tolist()
        indices_to_drop = set()
        title_lower = [t.lower().strip() for t in titles]

        for i in range(len(title_lower)):
            if i in indices_to_drop:
                continue
            for j in range(i + 1, len(title_lower)):
                if j in indices_to_drop:
                    continue
                if not title_lower[i] or not title_lower[j]:
                    continue
                score = fuzz.token_sort_ratio(title_lower[i], title_lower[j])
                if score >= 95:
                    indices_to_drop.add(j)

        keep_mask = [
            i not in indices_to_drop for i in range(len(papers_df))
        ]
        papers_df = papers_df[keep_mask].reset_index(drop=True)
        after_title = len(papers_df)

        logger.info(
            "Deduplication: %d → %d (PMID) → %d (title similarity)",
            initial_count, after_pmid, after_title,
        )
        return papers_df

    # ── Date normalisation ────────────────────────────────────────────────────

    def normalize_dates(self, date_string: Optional[str]) -> Optional[str]:
        """
        Parse all PubMed date formats and return a uniform YYYY-MM-DD string,
        or YYYY-MM or YYYY if day/month are unavailable.
        Returns None if no year can be extracted.
        """
        if not date_string or not isinstance(date_string, str):
            return None
        s = date_string.strip()

        # Already ISO format
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            return s
        if re.match(r"^\d{4}-\d{2}$", s):
            return s
        if re.match(r"^\d{4}$", s):
            return s

        # "YYYY Mon DD" or "YYYY Mon" — but NOT "YYYY Mon-Mon" (range → year only)
        m = re.match(
            r"(\d{4})\s+([A-Za-z]+)(?:\s+(\d{1,2}))?", s
        )
        if m:
            year, month_str, day = m.group(1), m.group(2), m.group(3)
            # If the character after the month abbreviation is a hyphen it's
            # a month-range like "Nov-Dec" — return year only
            month_end = m.end(2)
            if month_end < len(s) and s[month_end] == "-":
                return year
            month = _MONTH_MAP.get(month_str.lower(), None)
            if month:
                if day:
                    return f"{year}-{month}-{int(day):02d}"
                return f"{year}-{month}"
            return year

        # "Mon YYYY" or "Month YYYY"
        m = re.match(r"([A-Za-z]+)\s+(\d{4})", s)
        if m:
            month_str, year = m.group(1), m.group(2)
            month = _MONTH_MAP.get(month_str.lower(), None)
            if month:
                return f"{year}-{month}"
            return year

        # "DD Mon YYYY"
        m = re.match(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", s)
        if m:
            day, month_str, year = m.group(1), m.group(2), m.group(3)
            month = _MONTH_MAP.get(month_str.lower(), None)
            if month:
                return f"{year}-{month}-{int(day):02d}"
            return year

        # "YYYY/MM/DD" or "YYYY/MM"
        m = re.match(r"(\d{4})/(\d{1,2})(?:/(\d{1,2}))?", s)
        if m:
            year, month, day = m.group(1), m.group(2), m.group(3)
            result = f"{year}-{int(month):02d}"
            if day:
                result += f"-{int(day):02d}"
            return result

        # MedlineDate: "YYYY Mon-Mon" or "YYYY Season"
        m = re.match(r"(\d{4})", s)
        if m:
            return m.group(1)

        return None

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run_full_pipeline(
        self, raw_papers: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Apply all cleaning steps in sequence to a list of raw paper dicts
        (as produced by XMLParser.parse_batches).

        Steps:
          1. Convert to DataFrame
          2. Normalise pub_date
          3. Clean abstracts
          4. Normalise keywords (author keywords)
          5. Normalise author names
          6. Deduplicate by PMID and title similarity
          7. Add derived columns: pub_year, first_author

        Returns a clean, structured DataFrame. Fully query-agnostic.
        """
        if not raw_papers:
            logger.warning("run_full_pipeline received empty paper list")
            return pd.DataFrame()

        logger.info("DataCleaner: starting full pipeline on %d papers", len(raw_papers))

        # ── Step 1: build DataFrame ───────────────────────────────────────────
        df = pd.DataFrame(raw_papers)

        # Ensure required columns exist even if all records were missing them
        for col in (
            "pmid", "title", "abstract", "authors", "keywords",
            "mesh_terms", "mesh_qualifiers", "chemical_terms",
            "publication_types", "grant_info", "journal",
            "volume", "issue", "pages", "pub_date", "doi",
            "language", "query_used", "fetched_at",
        ):
            if col not in df.columns:
                df[col] = None

        # ── Step 2: normalise dates ───────────────────────────────────────────
        df["pub_date"] = df["pub_date"].apply(self.normalize_dates)

        # ── Step 3: clean abstracts ───────────────────────────────────────────
        df["abstract"] = df["abstract"].apply(
            lambda x: self.clean_abstract(x) if isinstance(x, str) else x
        )

        # ── Step 4: normalise author-supplied keywords ────────────────────────
        def _clean_kw_list(kw_list) -> List[str]:
            if not isinstance(kw_list, list):
                return []
            seen = set()
            result = []
            for kw in kw_list:
                norm = self.normalize_keyword(str(kw))
                if norm and norm not in seen:
                    seen.add(norm)
                    result.append(norm)
            return result

        df["keywords"] = df["keywords"].apply(_clean_kw_list)

        # ── Step 5: normalise author names within each paper ──────────────────
        def _clean_authors(authors) -> List[Dict[str, Any]]:
            if not isinstance(authors, list):
                return []
            cleaned = []
            for a in authors:
                if not isinstance(a, dict):
                    continue
                raw_name = a.get("name", "")
                norm_name = self.normalize_author_name(raw_name)
                cleaned.append({**a, "name": norm_name,
                                 "normalized_name": norm_name})
            return cleaned

        df["authors"] = df["authors"].apply(_clean_authors)

        # ── Step 6: deduplication ─────────────────────────────────────────────
        df = self.detect_duplicates(df)

        # ── Step 7: derived columns ───────────────────────────────────────────
        df["pub_year"] = df["pub_date"].apply(self._extract_year)

        df["first_author"] = df["authors"].apply(
            lambda auths: auths[0]["name"]
            if isinstance(auths, list) and auths else None
        )

        df["author_count"] = df["authors"].apply(
            lambda auths: len(auths) if isinstance(auths, list) else 0
        )

        df["keyword_count"] = df["keywords"].apply(
            lambda kws: len(kws) if isinstance(kws, list) else 0
        )

        df["mesh_count"] = df["mesh_terms"].apply(
            lambda m: len(m) if isinstance(m, list) else 0
        )

        logger.info(
            "DataCleaner: pipeline complete — %d clean papers, "
            "year range %s–%s",
            len(df),
            df["pub_year"].min() if not df.empty else "N/A",
            df["pub_year"].max() if not df.empty else "N/A",
        )
        return df

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_year(date_str: Optional[str]) -> Optional[int]:
        """Return the 4-digit year from a normalised date string, or None."""
        if not date_str or not isinstance(date_str, str):
            return None
        m = re.match(r"(\d{4})", date_str)
        return int(m.group(1)) if m else None

    # ── Corpus-level author deduplication ─────────────────────────────────────

    def deduplicate_authors_corpus(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Build a global author name mapping across the entire corpus and
        apply it to the DataFrame's 'authors' column.

        Returns:
            (updated_df, global_name_map)
        where global_name_map maps every raw name to its canonical form.
        """
        # Collect all unique raw names
        all_raw_names: List[str] = []
        for auths in df["authors"]:
            if isinstance(auths, list):
                for a in auths:
                    if isinstance(a, dict) and a.get("name"):
                        all_raw_names.append(a["name"])

        unique_names = list(set(all_raw_names))
        logger.info(
            "Building global author map for %d unique names …", len(unique_names)
        )
        global_map = self.deduplicate_authors(unique_names)

        def _apply_map(authors):
            if not isinstance(authors, list):
                return authors
            result = []
            for a in authors:
                if isinstance(a, dict):
                    raw = a.get("name", "")
                    canonical = global_map.get(raw, raw)
                    result.append({**a, "name": canonical,
                                   "normalized_name": canonical})
            return result

        df = df.copy()
        df["authors"] = df["authors"].apply(_apply_map)
        df["first_author"] = df["authors"].apply(
            lambda auths: auths[0]["name"]
            if isinstance(auths, list) and auths else None
        )
        logger.info("Global author deduplication complete.")
        return df, global_map
