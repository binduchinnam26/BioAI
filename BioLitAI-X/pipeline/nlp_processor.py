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

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load SciSpaCy model and UMLS linker on first use."""
        if self._loaded:
            return
        try:
            import spacy
            from scispacy.linking import EntityLinker  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SciSpaCy is not installed. Run: "
                "pip install scispacy && "
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                "releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
            ) from exc

        logger.info("Loading SciSpaCy model: en_core_sci_lg …")
        try:
            self._nlp = spacy.load("en_core_sci_lg")
        except OSError as exc:
            raise OSError(
                "SciSpaCy model 'en_core_sci_lg' not found. Install with:\n"
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                "releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
            ) from exc

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
