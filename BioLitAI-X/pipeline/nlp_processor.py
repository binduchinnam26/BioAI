"""
NLPProcessor — biomedical NER, UMLS linking, and relationship extraction.

Uses SciSpaCy en_core_sci_lg + UMLS entity linker.
Fully query-agnostic: entity types and relationship logic are driven by
whatever the model finds in the text, not by any hardcoded domain.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Entity types we surface from SciSpaCy / UMLS
TARGET_ENTITY_TYPES = {
    "DISEASE",
    "GENE_OR_GENOME",
    "CHEMICAL",
    "BIOLOGICAL_PROCESS",
    "CELL",
    "ORGANISM",
    "LABORATORY_PROCEDURE",
}

# Dependency relation labels that signal a relationship between two entities
RELATION_VERBS = {
    "nsubj", "nsubjpass", "dobj", "iobj", "prep", "agent",
    "ccomp", "xcomp", "relcl", "advcl",
}

# ── Regex-based fallback patterns (used when SciSpaCy is not installed) ────────

# Common non-biomedical uppercase abbreviations to exclude from gene matches
_UPPERCASE_STOPWORDS = {
    "A", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "IF", "IN", "IS", "IT",
    "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE",
    "AND", "ARE", "BUT", "CAN", "DID", "FOR", "HAD", "HAS", "HER", "HIM",
    "HIS", "HOW", "ITS", "LET", "MAY", "NOT", "NOW", "OUR", "OUT", "OWN",
    "SAY", "SHE", "THE", "TWO", "USE", "VIA", "WAS", "WHO", "WHY", "YET",
    "ALSO", "BEEN", "BOTH", "CAME", "DOES", "EACH", "EVEN", "FROM", "GAVE",
    "HAVE", "HERE", "INTO", "JUST", "LIKE", "MADE", "MANY", "MORE", "MOST",
    "MUCH", "MUST", "OVER", "SAME", "SUCH", "THAN", "THAT", "THEM", "THEN",
    "THEY", "THIS", "THUS", "UPON", "USED", "VERY", "WERE", "WITH", "YEAR",
    "AFTER", "AMONG", "BEING", "COULD", "FOUND", "GIVEN", "GROUP", "HEART",
    "HUMAN", "LEVEL", "LOWER", "MAJOR", "MIGHT", "NOTED", "OFTEN", "OTHER",
    "SHOWN", "SINCE", "STUDY", "THESE", "THREE", "TOTAL", "UNDER", "USING",
    "WEEKS", "WHICH", "WHILE", "YEARS", "ABOVE", "ABOUT",
    # Location/org abbreviations
    "USA", "UK", "EU", "UN", "WHO", "FDA", "NIH", "CDC", "EPA",
    # Generic scientific
    "DNA", "RNA", "PCR", "MRI", "CT", "UV", "IR",
}

_FALLBACK_PATTERNS: List[tuple] = [
    # Gene/protein symbols: 2-6 uppercase letters + optional digits
    (re.compile(r'\b([A-Z]{2,6}\d*)\b'), "GENE_OR_GENOME"),
    # Gene with embedded digits: BRCA1, TP53, CDK4, HER2
    (re.compile(r'\b([A-Z]{1,4}\d+[A-Z]?)\b'), "GENE_OR_GENOME"),
    # Cytokine/growth-factor families: IL-6, TNF-α, IFN-gamma, VEGF-A
    (re.compile(
        r'\b((?:IL|TNF|IFN|TGF|VEGF|EGF|FGF|IGF|CSF|HGF|PDGF|NGF|BMP|WNT)'
        r'[-–][\w]+)\b', re.IGNORECASE
    ), "GENE_OR_GENOME"),
    # Diseases by suffix (min 6 chars to skip short common words)
    (re.compile(
        r'\b(\w{3,}(?:oma|itis|emia|pathy|osis|plasia|trophy|oma|ectomy|otomy))\b',
        re.IGNORECASE
    ), "DISEASE"),
    # Cancer/syndrome/disorder as standalone or with preceding adjective
    (re.compile(
        r'\b((?:[A-Z][a-z]+[-\s]){0,2}'
        r'(?:cancer|carcinoma|sarcoma|lymphoma|leukemia|melanoma|glioma'
        r'|syndrome|disorder|disease|tumor|tumour))\b'
    ), "DISEASE"),
    # Drugs/biologics by suffix
    (re.compile(
        r'\b(\w{4,}(?:mab|nib|zumab|tinib|ciclib|mycin|cillin|cycline'
        r'|statin|sartan|prazole|vir|ide|ine|ol))\b',
        re.IGNORECASE
    ), "CHEMICAL"),
    # Cell types by suffix
    (re.compile(r'\b(\w{3,}(?:cyte|blast|phage|phil|sphere)s?)\b',
                re.IGNORECASE), "CELL"),
    # Biological processes (high-confidence specific terms)
    (re.compile(
        r'\b(apoptosis|autophagy|necrosis|ferroptosis|pyroptosis'
        r'|angiogenesis|metastasis|proliferation|differentiation'
        r'|senescence|inflammation|fibrosis|neurodegeneration)\b',
        re.IGNORECASE
    ), "BIOLOGICAL_PROCESS"),
]


class NLPProcessor:
    """
    Performs biomedical NER and relationship extraction on abstract text.
    Lazy-loads SciSpaCy to avoid import-time crashes when the model is not
    installed yet.
    """

    def __init__(self):
        self._nlp = None
        self._linker = None
        self._loaded = False
        self._scispacy_available: Optional[bool] = None  # None = not yet checked

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load SciSpaCy model and UMLS linker on first use.

        Sets self._scispacy_available to False (without raising) if scispacy or
        its model is missing so callers can fall back to regex NER.
        """
        if self._loaded or self._scispacy_available is False:
            return
        try:
            import spacy
            from scispacy.linking import EntityLinker  # noqa: F401
        except ImportError:
            logger.warning(
                "SciSpaCy not installed — entity extraction will use the "
                "built-in regex fallback. "
                "Install with: pip install scispacy && "
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                "releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
            )
            self._scispacy_available = False
            return

        logger.info("Loading SciSpaCy model: en_core_sci_lg …")
        try:
            self._nlp = spacy.load("en_core_sci_lg")
        except OSError:
            logger.warning(
                "SciSpaCy model 'en_core_sci_lg' not found — "
                "falling back to regex NER. "
                "Install model with: pip install https://s3-us-west-2.amazonaws.com/"
                "ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
            )
            self._scispacy_available = False
            return

        logger.info("Adding UMLS entity linker …")
        self._nlp.add_pipe(
            "scispacy_linker",
            config={
                "resolve_abbreviations": True,
                "linker_name": "umls",
                "threshold": 0.85,
                "filter_for_definitions": False,
                "max_entities_per_mention": 1,
            },
        )
        self._linker = self._nlp.get_pipe("scispacy_linker")
        self._loaded = True
        self._scispacy_available = True
        logger.info("NLPProcessor model loaded successfully.")

    # ── Entity extraction ─────────────────────────────────────────────────────

    def extract_entities(
        self, abstract_text: str
    ) -> List[Dict[str, Optional[str]]]:
        """
        Run NER + UMLS linking on *abstract_text*.

        Returns a list of dicts:
          { entity_text, entity_type, umls_id, sentence_context }

        Only entities whose SciSpaCy label maps to one of TARGET_ENTITY_TYPES
        are returned.  Unknown / unlabelled entities are tagged as 'UNKNOWN'.
        """
        if not abstract_text or not isinstance(abstract_text, str):
            return []
        self._load_model()

        if not self._loaded:
            return self._extract_entities_fallback(abstract_text)

        try:
            doc = self._nlp(abstract_text[:100_000])  # guard against huge texts
        except Exception as exc:
            logger.warning("NLP processing failed: %s", exc)
            return []

        results: List[Dict[str, Optional[str]]] = []
        seen: set = set()

        for ent in doc.ents:
            entity_text = ent.text.strip()
            if not entity_text or len(entity_text) < 2:
                continue

            entity_type = self._map_entity_type(ent.label_)
            umls_id = self._get_umls_id(ent)
            sentence_context = self._sentence_for_span(ent)

            key = (entity_text.lower(), entity_type)
            if key in seen:
                continue
            seen.add(key)

            results.append(
                {
                    "entity_text": entity_text,
                    "entity_type": entity_type,
                    "umls_id": umls_id,
                    "sentence_context": sentence_context,
                }
            )

        return results

    # ── Relationship extraction ───────────────────────────────────────────────

    def extract_relationships(
        self,
        abstract_text: str,
        entities: List[Dict[str, Optional[str]]],
    ) -> List[Dict[str, Any]]:
        """
        Use dependency parsing to find verb-mediated relationships between
        entity pairs within the same sentence.

        Returns a list of dicts:
          { source_entity, target_entity, relationship_type, sentence_context }
        """
        if not abstract_text or not entities:
            return []
        self._load_model()

        if not self._loaded:
            return self._extract_relationships_fallback(abstract_text, entities)

        entity_texts = {e["entity_text"].lower() for e in entities}
        relationships: List[Dict[str, Any]] = []

        try:
            doc = self._nlp(abstract_text[:100_000])
        except Exception as exc:
            logger.warning("Relationship extraction NLP failed: %s", exc)
            return []

        for sent in doc.sents:
            sent_entities = self._entities_in_sent(sent, entity_texts)
            if len(sent_entities) < 2:
                continue

            for token in sent:
                if token.pos_ != "VERB":
                    continue
                subj = self._find_dep_entity(token, "nsubj", sent_entities)
                obj = self._find_dep_entity(token, "dobj", sent_entities)
                if not subj and not obj:
                    subj = self._find_dep_entity(token, "nsubjpass", sent_entities)
                    obj = self._find_dep_entity(token, "agent", sent_entities)
                if subj and obj and subj != obj:
                    relationships.append(
                        {
                            "source_entity": subj,
                            "target_entity": obj,
                            "relationship_type": token.lemma_.lower(),
                            "sentence_context": sent.text.strip(),
                        }
                    )

        return relationships

    # ── Corpus processing ─────────────────────────────────────────────────────

    def process_corpus(
        self,
        papers_df,
        db_manager=None,
        progress_callback=None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply NER + relationship extraction to all abstracts.

        Args:
            papers_df: pandas DataFrame with at least 'pmid' and 'abstract' columns.
            db_manager: optional DatabaseManager — if provided, entities and
                        relationships are stored directly to the database.
            progress_callback(done, total): optional progress hook for the UI.

        Returns:
            (all_entities, all_relationships) as flat lists of dicts with
            'pmid' added to each record.
        """
        import pandas as pd

        if papers_df is None or (hasattr(papers_df, "empty") and papers_df.empty):
            return [], []

        total = len(papers_df)
        all_entities: List[Dict] = []
        all_relationships: List[Dict] = []

        for i, row in enumerate(papers_df.itertuples(index=False)):
            pmid = str(getattr(row, "pmid", ""))
            abstract = getattr(row, "abstract", "") or ""

            entities = self.extract_entities(abstract)
            rels = self.extract_relationships(abstract, entities)

            for ent in entities:
                ent["pmid"] = pmid
                all_entities.append(ent)

            for rel in rels:
                rel["pmid"] = pmid
                all_relationships.append(rel)

            if db_manager is not None:
                self._persist_to_db(pmid, entities, rels, db_manager)

            if progress_callback:
                try:
                    progress_callback(i + 1, total)
                except Exception:
                    pass

            if (i + 1) % 50 == 0:
                logger.info(
                    "NLP: processed %d / %d papers (%d entities, %d rels so far)",
                    i + 1, total, len(all_entities), len(all_relationships),
                )

        logger.info(
            "NLP corpus processing complete: %d entities, %d relationships",
            len(all_entities), len(all_relationships),
        )
        return all_entities, all_relationships

    # ── Database persistence ──────────────────────────────────────────────────

    @staticmethod
    def _persist_to_db(pmid, entities, relationships, db):
        """Write entities and relationships for one paper to the database."""
        entity_id_map: Dict[str, int] = {}
        for ent in entities:
            eid = db.insert_entity(
                name=ent["entity_text"],
                entity_type=ent["entity_type"],
                umls_id=ent.get("umls_id"),
            )
            if eid > 0:
                db.link_paper_entity(pmid, eid, ent.get("sentence_context"))
                entity_id_map[ent["entity_text"].lower()] = eid

        for rel in relationships:
            src_id = entity_id_map.get(rel["source_entity"].lower())
            tgt_id = entity_id_map.get(rel["target_entity"].lower())
            if src_id and tgt_id:
                db.insert_relationship(
                    source_id=src_id,
                    target_id=tgt_id,
                    rel_type=rel["relationship_type"],
                    evidence_pmid=pmid,
                    confidence=0.7,
                )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _map_entity_type(spacy_label: str) -> str:
        """
        Map SciSpaCy entity labels to our canonical TARGET_ENTITY_TYPES.
        Unknown labels are kept as-is so no information is silently dropped.
        """
        mapping = {
            "DISEASE": "DISEASE",
            "CANCER": "DISEASE",
            "GENE_OR_GENE_PRODUCT": "GENE_OR_GENOME",
            "GENE_OR_GENOME": "GENE_OR_GENOME",
            "SIMPLE_CHEMICAL": "CHEMICAL",
            "CHEMICAL": "CHEMICAL",
            "DRUG": "CHEMICAL",
            "BIOLOGICAL_PROCESS": "BIOLOGICAL_PROCESS",
            "CELL": "CELL",
            "CELL_TYPE": "CELL",
            "CELL_LINE": "CELL",
            "ORGANISM": "ORGANISM",
            "TAXON": "ORGANISM",
            "LABORATORY_PROCEDURE": "LABORATORY_PROCEDURE",
            "PROCEDURE": "LABORATORY_PROCEDURE",
            "DIAGNOSTIC_PROCEDURE": "LABORATORY_PROCEDURE",
        }
        return mapping.get(spacy_label.upper(), spacy_label)

    def _get_umls_id(self, ent) -> Optional[str]:
        """Extract the top UMLS CUI for a spaCy entity span."""
        try:
            kb_ents = ent._.kb_ents
            if kb_ents:
                return kb_ents[0][0]  # (CUI, score) — take CUI
        except AttributeError:
            pass
        return None

    @staticmethod
    def _sentence_for_span(ent) -> str:
        """Return the sentence containing the entity span."""
        try:
            return ent.sent.text.strip()
        except Exception:
            return ent.text

    def _extract_entities_fallback(
        self, abstract_text: str
    ) -> List[Dict[str, Optional[str]]]:
        """
        Regex-based entity extractor used when SciSpaCy is not installed.
        Identifies gene symbols, disease terms, drug/chemical names, and cell
        types using pattern matching against _FALLBACK_PATTERNS.
        """
        results: List[Dict[str, Optional[str]]] = []
        seen: set = set()
        sentences = re.split(r'(?<=[.!?])\s+', abstract_text.strip())

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            for pattern, etype in _FALLBACK_PATTERNS:
                for m in pattern.finditer(sent):
                    entity_text = m.group(1).strip()
                    if len(entity_text) < 3:
                        continue
                    if entity_text.upper() in _UPPERCASE_STOPWORDS:
                        continue
                    key = (entity_text.lower(), etype)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(
                        {
                            "entity_text": entity_text,
                            "entity_type": etype,
                            "umls_id": None,
                            "sentence_context": sent,
                        }
                    )
        return results

    def _extract_relationships_fallback(
        self,
        abstract_text: str,
        entities: List[Dict[str, Optional[str]]],
    ) -> List[Dict[str, Any]]:
        """
        Co-occurrence-based relationship extractor used when SciSpaCy is
        unavailable.

        Strategy:
        - Split the abstract into sentences.
        - Within each sentence, find all entity pairs that both appear.
        - Assign a relationship type based on the entity type combination
          (e.g. GENE_OR_GENOME + DISEASE → "associated_with") or a generic
          "co_occurs_with" label.
        - Limit to MAX_PAIRS_PER_SENT pairs per sentence to avoid
          combinatorial explosion on entity-dense sentences.
        """
        MAX_PAIRS_PER_SENT = 6

        # Build a lookup: entity_text_lower → entity record
        entity_map: Dict[str, Dict] = {
            e["entity_text"].lower(): e for e in entities
        }

        # Type-pair → relationship label
        _TYPE_REL: Dict[tuple, str] = {
            ("GENE_OR_GENOME", "DISEASE"): "associated_with",
            ("DISEASE", "GENE_OR_GENOME"): "associated_with",
            ("GENE_OR_GENOME", "CHEMICAL"): "interacts_with",
            ("CHEMICAL", "GENE_OR_GENOME"): "interacts_with",
            ("CHEMICAL", "DISEASE"): "treats",
            ("DISEASE", "CHEMICAL"): "treated_by",
            ("GENE_OR_GENOME", "BIOLOGICAL_PROCESS"): "regulates",
            ("BIOLOGICAL_PROCESS", "GENE_OR_GENOME"): "regulated_by",
            ("CELL", "DISEASE"): "implicated_in",
            ("GENE_OR_GENOME", "CELL"): "expressed_in",
        }

        sentences = re.split(r'(?<=[.!?])\s+', abstract_text.strip())
        relationships: List[Dict[str, Any]] = []
        seen_pairs: set = set()

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_lower = sent.lower()

            # Find which entities appear in this sentence
            present = [
                ent for txt, ent in entity_map.items()
                if txt in sent_lower
            ]
            if len(present) < 2:
                continue

            pairs_added = 0
            for i in range(len(present)):
                if pairs_added >= MAX_PAIRS_PER_SENT:
                    break
                for j in range(i + 1, len(present)):
                    if pairs_added >= MAX_PAIRS_PER_SENT:
                        break
                    src = present[i]["entity_text"]
                    tgt = present[j]["entity_text"]
                    if src == tgt:
                        continue
                    pair_key = tuple(sorted([src.lower(), tgt.lower()]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    src_type = present[i].get("entity_type", "UNKNOWN")
                    tgt_type = present[j].get("entity_type", "UNKNOWN")
                    rel = _TYPE_REL.get(
                        (src_type, tgt_type),
                        "co_occurs_with",
                    )
                    relationships.append(
                        {
                            "source_entity": src,
                            "target_entity": tgt,
                            "relationship_type": rel,
                            "sentence_context": sent,
                        }
                    )
                    pairs_added += 1

        return relationships

    @staticmethod
    def _entities_in_sent(sent, entity_texts: set) -> List[str]:
        """Return entity text strings present in a sentence."""
        sent_text = sent.text.lower()
        return [e for e in entity_texts if e in sent_text]

    @staticmethod
    def _find_dep_entity(
        verb_token, dep_label: str, sent_entities: List[str]
    ) -> Optional[str]:
        """
        Find a child of *verb_token* with dependency label *dep_label* whose
        text matches one of *sent_entities*.
        """
        for child in verb_token.children:
            if child.dep_ == dep_label:
                child_text = child.text.lower()
                for ent_text in sent_entities:
                    if ent_text in child_text or child_text in ent_text:
                        return ent_text
        return None
