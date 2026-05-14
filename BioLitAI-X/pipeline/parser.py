"""
XMLParser — extracts all structured fields from PubMed XML records.
Works entirely from the data returned by Entrez; no domain assumptions.
"""

import logging
import re
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

# Month abbreviation → zero-padded number
_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


class XMLParser:
    """
    Parses batches of PubMed XML strings and returns a list of structured
    paper dictionaries suitable for database insertion.
    """

    def parse_batch(self, xml_string: str,
                    query_used: str = "") -> List[Dict[str, Any]]:
        """
        Parse one XML batch (the string returned by Entrez.efetch).
        Returns a list of paper dicts.
        """
        if not xml_string or not xml_string.strip():
            return []
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as exc:
            logger.error("XML parse error: %s", exc)
            return []

        papers = []
        for article_node in root.findall(".//PubmedArticle"):
            try:
                paper = self._parse_article(article_node, query_used)
                if paper:
                    papers.append(paper)
            except Exception as exc:
                logger.warning("Skipping malformed article node: %s", exc)
        return papers

    def parse_batches(self, xml_batches: List[str],
                      query_used: str = "") -> List[Dict[str, Any]]:
        """Parse multiple XML batch strings and return a deduplicated flat list."""
        all_papers: Dict[str, Dict] = {}
        for xml in xml_batches:
            for paper in self.parse_batch(xml, query_used):
                pmid = paper.get("pmid")
                if pmid and pmid not in all_papers:
                    all_papers[pmid] = paper
        return list(all_papers.values())

    # ── Article-level parsing ─────────────────────────────────────────────────

    def _parse_article(self, node: ET.Element,
                       query_used: str) -> Optional[Dict[str, Any]]:
        pmid = self._text(node, ".//PMID")
        if not pmid:
            logger.warning("Article node has no PMID — skipping")
            return None

        title = self._parse_title(node)
        abstract = self._parse_abstract(node)

        if not title:
            logger.warning("PMID %s missing title", pmid)
        if not abstract:
            logger.warning("PMID %s missing abstract", pmid)

        authors = self._parse_authors(node)
        keywords = self._parse_author_keywords(node)
        mesh_terms, mesh_qualifiers = self._parse_mesh(node)
        chemical_terms = self._parse_chemicals(node)
        publication_types = self._parse_publication_types(node)
        grant_info = self._parse_grants(node)
        journal_meta = self._parse_journal(node)
        doi = self._parse_doi(node)
        language = self._text(node, ".//Language")
        pub_date = self._parse_pub_date(node)

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "keywords": keywords,
            "mesh_terms": mesh_terms,
            "mesh_qualifiers": mesh_qualifiers,
            "chemical_terms": chemical_terms,
            "publication_types": publication_types,
            "grant_info": grant_info,
            "journal": journal_meta.get("journal"),
            "volume": journal_meta.get("volume"),
            "issue": journal_meta.get("issue"),
            "pages": journal_meta.get("pages"),
            "pub_date": pub_date,
            "doi": doi,
            "language": language,
            "query_used": query_used,
            "fetched_at": datetime.utcnow().isoformat(),
        }

    # ── Field extractors ──────────────────────────────────────────────────────

    def _parse_title(self, node: ET.Element) -> Optional[str]:
        title_node = node.find(".//ArticleTitle")
        if title_node is None:
            return None
        return self._inner_text(title_node)

    def _parse_abstract(self, node: ET.Element) -> Optional[str]:
        """
        Preserves structured abstract section labels (e.g. BACKGROUND,
        METHODS) as "LABEL: text" separated by newlines.
        """
        abstract_node = node.find(".//Abstract")
        if abstract_node is None:
            return None

        parts = []
        for text_node in abstract_node.findall("AbstractText"):
            label = text_node.get("Label", "")
            text = self._inner_text(text_node) or ""
            if label:
                parts.append(f"{label}: {text}")
            elif text:
                parts.append(text)

        return "\n".join(parts) if parts else None

    def _parse_authors(self, node: ET.Element) -> List[Dict[str, Any]]:
        authors = []
        for auth in node.findall(".//Author"):
            last = self._text(auth, "LastName") or ""
            fore = self._text(auth, "ForeName") or ""
            initials = self._text(auth, "Initials") or ""
            collective = self._text(auth, "CollectiveName") or ""
            name = (
                f"{last}, {fore}".strip(", ")
                if (last or fore)
                else collective
            )
            affiliations = [
                self._inner_text(a)
                for a in auth.findall(".//Affiliation")
                if self._inner_text(a)
            ]
            if name:
                authors.append({
                    "name": name,
                    "last_name": last,
                    "fore_name": fore,
                    "initials": initials,
                    "affiliation": "; ".join(affiliations) if affiliations else None,
                })
        return authors

    def _parse_author_keywords(self, node: ET.Element) -> List[str]:
        keywords = []
        for kw_list in node.findall(".//KeywordList"):
            for kw in kw_list.findall("Keyword"):
                text = self._inner_text(kw)
                if text:
                    keywords.append(text)
        return keywords

    def _parse_mesh(
        self, node: ET.Element
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Returns:
          mesh_terms  — list of {descriptor, qualifier, is_major_topic}
          mesh_qualifiers — flat deduplicated list of qualifier strings
        """
        mesh_terms: List[Dict[str, Any]] = []
        qualifiers_seen: set = set()

        for mh in node.findall(".//MeshHeading"):
            descriptor_node = mh.find("DescriptorName")
            if descriptor_node is None:
                continue
            descriptor = self._inner_text(descriptor_node) or ""
            is_major = descriptor_node.get("MajorTopicYN", "N") == "Y"

            qualifier_nodes = mh.findall("QualifierName")
            if qualifier_nodes:
                for qn in qualifier_nodes:
                    qualifier = self._inner_text(qn) or ""
                    q_major = qn.get("MajorTopicYN", "N") == "Y"
                    mesh_terms.append({
                        "descriptor": descriptor,
                        "qualifier": qualifier,
                        "is_major_topic": is_major or q_major,
                    })
                    qualifiers_seen.add(qualifier)
            else:
                mesh_terms.append({
                    "descriptor": descriptor,
                    "qualifier": None,
                    "is_major_topic": is_major,
                })

        return mesh_terms, sorted(qualifiers_seen)

    def _parse_chemicals(self, node: ET.Element) -> List[Dict[str, str]]:
        chemicals = []
        for chem in node.findall(".//Chemical"):
            name_node = chem.find("NameOfSubstance")
            if name_node is None:
                continue
            name = self._inner_text(name_node) or ""
            registry = self._text(chem, "RegistryNumber") or ""
            if name:
                chemicals.append({"name": name, "registry_number": registry or None})
        return chemicals

    def _parse_publication_types(self, node: ET.Element) -> List[str]:
        types = []
        for pt in node.findall(".//PublicationType"):
            text = self._inner_text(pt)
            if text:
                types.append(text)
        return types

    def _parse_grants(self, node: ET.Element) -> List[Dict[str, str]]:
        grants = []
        for grant in node.findall(".//Grant"):
            grant_id = self._text(grant, "GrantID") or ""
            agency = self._text(grant, "Agency") or ""
            country = self._text(grant, "Country") or ""
            if grant_id or agency:
                grants.append({"id": grant_id, "agency": agency, "country": country})
        return grants

    def _parse_journal(self, node: ET.Element) -> Dict[str, Optional[str]]:
        journal = self._text(node, ".//Journal/Title") or \
                  self._text(node, ".//MedlineTA") or None
        volume = self._text(node, ".//JournalIssue/Volume")
        issue = self._text(node, ".//JournalIssue/Issue")
        pages = self._text(node, ".//MedlinePgn")
        return {"journal": journal, "volume": volume,
                "issue": issue, "pages": pages}

    def _parse_doi(self, node: ET.Element) -> Optional[str]:
        for aid in node.findall(".//ArticleId"):
            if aid.get("IdType", "").lower() == "doi":
                return (aid.text or "").strip() or None
        # Also try ELocationID
        for loc in node.findall(".//ELocationID"):
            if loc.get("EIdType", "").lower() == "doi":
                return (loc.text or "").strip() or None
        return None

    def _parse_pub_date(self, node: ET.Element) -> Optional[str]:
        """Return a normalised ISO-style date string (YYYY-MM-DD or YYYY-MM or YYYY)."""
        # Prefer ArticleDate (electronic publication date)
        for adt in node.findall(".//ArticleDate"):
            y = self._text(adt, "Year")
            m = self._text(adt, "Month")
            d = self._text(adt, "Day")
            if y:
                return self._build_date(y, m, d)

        # Fall back to PubDate in JournalIssue
        pd = node.find(".//JournalIssue/PubDate")
        if pd is not None:
            y = self._text(pd, "Year")
            m = self._text(pd, "Month")
            d = self._text(pd, "Day")
            medline = self._text(pd, "MedlineDate")
            if y:
                return self._build_date(y, m, d)
            if medline:
                return self._parse_medline_date(medline)
        return None

    # ── Date helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_date(year: Optional[str], month: Optional[str],
                    day: Optional[str]) -> str:
        if not year:
            return ""
        parts = [year.zfill(4)]
        if month:
            m = _MONTH_MAP.get(month.lower()[:3], month.zfill(2))
            parts.append(m)
            if day:
                parts.append(day.zfill(2))
        return "-".join(parts)

    @staticmethod
    def _parse_medline_date(medline: str) -> str:
        """Parse MedlineDate strings like '2003 Nov-Dec' or '2003 Spring'."""
        match = re.match(r"(\d{4})", medline)
        if match:
            year = match.group(1)
            # Try to extract month abbreviation
            m_match = re.search(
                r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",
                medline, re.IGNORECASE,
            )
            if m_match:
                month = _MONTH_MAP.get(m_match.group(1).lower(), "01")
                return f"{year}-{month}"
            return year
        return medline

    # ── XML utility helpers ───────────────────────────────────────────────────

    @staticmethod
    def _text(node: ET.Element, path: str) -> Optional[str]:
        """Find first matching child by XPath and return its text."""
        found = node.find(path)
        if found is not None and found.text:
            return found.text.strip()
        return None

    @staticmethod
    def _inner_text(node: ET.Element) -> Optional[str]:
        """Return all text inside *node*, including text from child elements."""
        if node is None:
            return None
        parts = []
        if node.text:
            parts.append(node.text)
        for child in node:
            if child.text:
                parts.append(child.text)
            if child.tail:
                parts.append(child.tail)
        text = "".join(parts).strip()
        return text if text else None
