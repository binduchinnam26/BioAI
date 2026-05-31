"""
GapDetector — identifies structural, temporal, and cross-domain research gaps
in the knowledge graph and publiation timeline.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from config import KG_MIN_SHARED_NEIGHBORS_FOR_GAP

logger = logging.getLogger(__name__)


class GapDetector:
    """
    Analyses a KnowledgeGraph and papers DataFrame to surface candidate
    research gaps ranked by evidence strength.
    """

    # ── Structural gaps ───────────────────────────────────────────────────────

    def find_structural_gaps(
        self, knowledge_graph: nx.MultiDiGraph
    ) -> List[Dict[str, Any]]:
        """
        Identify pairs of nodes that:
          - share >= KG_MIN_SHARED_NEIGHBORS_FOR_GAP common neighbours
          - have NO direct edge between them

        Ranked by: shared_neighbors × geometric_mean(degree_a, degree_b)
        This surfaces the most "conspicuous" missing connections first.
        """
        if knowledge_graph.number_of_nodes() == 0:
            return []

        undirected = knowledge_graph.to_undirected()
        degrees = dict(undirected.degree())
        gaps: List[Dict[str, Any]] = []

        nodes = list(undirected.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                if undirected.has_edge(a, b):
                    continue
                # Common neighbours
                nbrs_a = set(undirected.neighbors(a))
                nbrs_b = set(undirected.neighbors(b))
                shared = nbrs_a & nbrs_b
                if len(shared) < KG_MIN_SHARED_NEIGHBORS_FOR_GAP:
                    continue

                import math
                deg_a = degrees.get(a, 1)
                deg_b = degrees.get(b, 1)
                score = len(shared) * math.sqrt(deg_a * deg_b)

                # Gather evidence PMIDs from neighbours
                evidence_pmids: List[str] = []
                for nb in shared:
                    for _, _, data in knowledge_graph.in_edges(nb, data=True):
                        evidence_pmids.extend(
                            data.get("evidence_pmids", [])
                        )
                    for _, _, data in knowledge_graph.out_edges(nb, data=True):
                        evidence_pmids.extend(
                            data.get("evidence_pmids", [])
                        )
                evidence_pmids = list(set(evidence_pmids))[:10]

                gaps.append(
                    {
                        "type": "structural",
                        "concept_a": a,
                        "concept_b": b,
                        "shared_neighbors": sorted(shared),
                        "shared_neighbor_count": len(shared),
                        "degree_a": deg_a,
                        "degree_b": deg_b,
                        "score": round(score, 4),
                        "evidence_pmids": evidence_pmids,
                        "entity_type_a": knowledge_graph.nodes[a].get(
                            "entity_type", "UNKNOWN"
                        ),
                        "entity_type_b": knowledge_graph.nodes[b].get(
                            "entity_type", "UNKNOWN"
                        ),
                    }
                )

        gaps.sort(key=lambda x: x["score"], reverse=True)
        logger.info("Structural gaps found: %d", len(gaps))
        return gaps

    # ── Temporal gaps ─────────────────────────────────────────────────────────

    def find_temporal_gaps(
        self,
        papers_df: pd.DataFrame,
        knowledge_graph: nx.MultiDiGraph,
    ) -> List[Dict[str, Any]]:
        """
        Identify entities that were actively studied in older papers but have
        very few papers in the most recent 3 years — suggesting neglected topics
        that may warrant renewed investigation.
        """
        if papers_df is None or papers_df.empty:
            return []

        # Determine year threshold
        all_years = pd.to_numeric(
            papers_df.get("pub_year", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        if all_years.empty:
            return []

        max_year = int(all_years.max())
        recent_cutoff = max_year - 3
        early_cutoff = max_year - 10

        # Count publications per entity per era
        entity_recent: Dict[str, int] = defaultdict(int)
        entity_early: Dict[str, int] = defaultdict(int)

        for _, row in papers_df.iterrows():
            year = row.get("pub_year")
            if not year:
                continue
            year = int(year)
            pmid = str(row.get("pmid", ""))
            # Find entities connected to this paper in the graph
            for node, data in knowledge_graph.nodes(data=True):
                for _, _, edata in knowledge_graph.in_edges(node, data=True):
                    if pmid in edata.get("evidence_pmids", []):
                        if year >= recent_cutoff:
                            entity_recent[node] += 1
                        elif year >= early_cutoff:
                            entity_early[node] += 1
                        break

        gaps: List[Dict[str, Any]] = []
        for entity in knowledge_graph.nodes():
            early = entity_early.get(entity, 0)
            recent = entity_recent.get(entity, 0)
            if early >= 3 and recent == 0:
                score = float(early) / (recent + 0.5)
                gaps.append(
                    {
                        "type": "temporal",
                        "concept_a": entity,
                        "concept_b": None,
                        "early_paper_count": early,
                        "recent_paper_count": recent,
                        "score": round(score, 4),
                        "entity_type_a": knowledge_graph.nodes[entity].get(
                            "entity_type", "UNKNOWN"
                        ),
                        "evidence_pmids": [],
                    }
                )

        gaps.sort(key=lambda x: x["score"], reverse=True)
        logger.info("Temporal gaps found: %d", len(gaps))
        return gaps

    # ── Cross-domain gaps ─────────────────────────────────────────────────────

    def find_cross_domain_gaps(
        self, knowledge_graph: nx.MultiDiGraph
    ) -> List[Dict[str, Any]]:
        """
        Find pairs of entities from DIFFERENT entity types that share many
        common neighbours but have no direct edge — suggesting unexplored
        cross-domain connections (e.g. a CHEMICAL with shared neighbours as
        a DISEASE but no direct link).
        """
        if knowledge_graph.number_of_nodes() == 0:
            return []

        undirected = knowledge_graph.to_undirected()
        gaps: List[Dict[str, Any]] = []

        nodes = list(undirected.nodes())
        for i in range(len(nodes)):
            a = nodes[i]
            type_a = knowledge_graph.nodes[a].get("entity_type", "UNKNOWN")
            for j in range(i + 1, len(nodes)):
                b = nodes[j]
                type_b = knowledge_graph.nodes[b].get("entity_type", "UNKNOWN")
                # Only cross-domain pairs
                if type_a == type_b:
                    continue
                if undirected.has_edge(a, b):
                    continue
                nbrs_a = set(undirected.neighbors(a))
                nbrs_b = set(undirected.neighbors(b))
                shared = nbrs_a & nbrs_b
                if len(shared) < KG_MIN_SHARED_NEIGHBORS_FOR_GAP:
                    continue

                evidence_pmids: List[str] = []
                for nb in shared:
                    for _, _, data in knowledge_graph.out_edges(nb, data=True):
                        evidence_pmids.extend(data.get("evidence_pmids", []))
                evidence_pmids = list(set(evidence_pmids))[:10]

                gaps.append(
                    {
                        "type": "cross_domain",
                        "concept_a": a,
                        "concept_b": b,
                        "entity_type_a": type_a,
                        "entity_type_b": type_b,
                        "shared_neighbors": sorted(shared),
                        "shared_neighbor_count": len(shared),
                        "score": float(len(shared)),
                        "evidence_pmids": evidence_pmids,
                    }
                )

        gaps.sort(key=lambda x: x["score"], reverse=True)
        logger.info("Cross-domain gaps found: %d", len(gaps))
        return gaps

    # ── Unified gap report ────────────────────────────────────────────────────

    def compile_gap_report(
        self,
        knowledge_graph: nx.MultiDiGraph,
        papers_df: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run all gap detectors and return a single ranked list.

        Structural gaps score highest (× 1.5 multiplier), cross-domain second
        (× 1.2), temporal gaps third (× 1.0).  All are sorted by adjusted score
        descending.
        """
        structural = self.find_structural_gaps(knowledge_graph)
        cross_domain = self.find_cross_domain_gaps(knowledge_graph)
        temporal: List[Dict[str, Any]] = []
        if papers_df is not None and not papers_df.empty:
            temporal = self.find_temporal_gaps(papers_df, knowledge_graph)

        multipliers = {"structural": 1.5, "cross_domain": 1.2, "temporal": 1.0}
        all_gaps: List[Dict[str, Any]] = []
        for gap in structural + cross_domain + temporal:
            gap = dict(gap)  # copy
            gap["adjusted_score"] = round(
                gap.get("score", 0.0)
                * multipliers.get(gap.get("type", ""), 1.0),
                4,
            )
            all_gaps.append(gap)

        all_gaps.sort(key=lambda x: x["adjusted_score"], reverse=True)
        logger.info("Gap report compiled: %d total gaps", len(all_gaps))
        return all_gaps
