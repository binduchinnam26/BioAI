"""
KnowledgeGraph — builds a directed MultiDiGraph from NLP entities and
relationships, enriched with semantic similarity edges and Louvain communities.
"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from config import (
    ENTITY_TYPE_COLORS,
    COMMUNITY_COLORS,
    KG_SIMILARITY_THRESHOLD,
    LOUVAIN_RANDOM_STATE,
)

logger = logging.getLogger(__name__)


def _louvain_undirected(graph: nx.Graph) -> Dict[Any, int]:
    if graph.number_of_nodes() == 0:
        return {}
    try:
        from community import best_partition
        return best_partition(graph, random_state=LOUVAIN_RANDOM_STATE)
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("python-louvain failed: %s", exc)
    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(graph, seed=LOUVAIN_RANDOM_STATE)
        partition = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                partition[node] = cid
        return partition
    except Exception as exc:
        logger.warning("networkx louvain failed: %s", exc)
        return {n: 0 for n in graph.nodes()}


class KnowledgeGraph:
    """
    Directed biomedical knowledge graph.
    Nodes: entities (colored by type, sized by evidence count).
    Edges: verb-labelled relationships with evidence PMIDs and confidence.
    """

    def __init__(self):
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    # ── Build from entities / relationships ───────────────────────────────────

    def build_from_entities(
        self,
        entities_df,
        relationships_df,
    ) -> nx.MultiDiGraph:
        """
        Construct the knowledge graph from DataFrames or lists of dicts.

        entities_df columns:    name, entity_type, umls_id, pmid (optional)
        relationships_df cols:  source_entity, target_entity,
                                relationship_type, evidence_pmid,
                                confidence_score (optional)

        Node attributes: entity_type, color_hex, umls_id, paper_count,
                         community_id, weight, label
        Edge attributes: relationship_type, evidence_pmids (list),
                         confidence_score, weight
        """
        import pandas as pd

        G = nx.MultiDiGraph()

        # ── Entities ──────────────────────────────────────────────────────────
        entity_papers: Dict[str, set] = defaultdict(set)
        entity_type_map: Dict[str, str] = {}
        entity_umls: Dict[str, Optional[str]] = {}

        rows = (
            entities_df.to_dict("records")
            if hasattr(entities_df, "to_dict")
            else (entities_df or [])
        )
        for row in rows:
            name = str(row.get("name") or row.get("entity_text") or "").strip()
            if not name:
                continue
            etype = str(row.get("entity_type") or "UNKNOWN")
            umls = row.get("umls_id") or row.get("umls")
            pmid = str(row.get("pmid") or row.get("evidence_pmid") or "")
            entity_type_map[name] = etype
            entity_umls[name] = umls
            if pmid:
                entity_papers[name].add(pmid)

        for name, etype in entity_type_map.items():
            paper_count = len(entity_papers.get(name, set()))
            G.add_node(
                name,
                entity_type=etype,
                color_hex=ENTITY_TYPE_COLORS.get(etype, "#9CA3AF"),
                umls_id=entity_umls.get(name),
                paper_count=paper_count,
                weight=max(paper_count, 1),
                label=name,
                community_id=0,   # filled by Louvain below
            )

        # ── Relationships ─────────────────────────────────────────────────────
        rel_rows = (
            relationships_df.to_dict("records")
            if hasattr(relationships_df, "to_dict")
            else (relationships_df or [])
        )

        # Aggregate evidence PMIDs per (source, rel_type, target) triple
        edge_evidence: Dict[Tuple, List[str]] = defaultdict(list)
        edge_conf: Dict[Tuple, float] = defaultdict(float)

        for row in rel_rows:
            src = str(
                row.get("source_entity")
                or row.get("source_entity_name")
                or ""
            ).strip()
            tgt = str(
                row.get("target_entity")
                or row.get("target_entity_name")
                or ""
            ).strip()
            rel = str(row.get("relationship_type") or "associated_with").strip()
            pmid = str(row.get("evidence_pmid") or row.get("pmid") or "")
            conf = float(row.get("confidence_score") or 0.7)

            if not src or not tgt or src == tgt:
                continue
            if not G.has_node(src) or not G.has_node(tgt):
                continue

            key = (src, rel, tgt)
            if pmid:
                edge_evidence[key].append(pmid)
            edge_conf[key] = max(edge_conf[key], conf)

        for (src, rel, tgt), pmids in edge_evidence.items():
            G.add_edge(
                src, tgt,
                relationship_type=rel,
                evidence_pmids=list(set(pmids)),
                confidence_score=edge_conf[(src, rel, tgt)],
                weight=len(set(pmids)),
            )

        # ── Louvain on undirected projection ──────────────────────────────────
        if G.number_of_nodes() > 1:
            undirected = G.to_undirected()
            partition = _louvain_undirected(undirected)
            unique_cids = sorted(set(partition.values()))
            color_cycle = {
                cid: COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]
                for i, cid in enumerate(unique_cids)
            }
            for node in G.nodes():
                cid = partition.get(node, 0)
                G.nodes[node]["community_id"] = cid
                # NOTE: color_hex stays as ENTITY_TYPE color per spec;
                # community_id is stored for filtering only

        self.graph = G
        logger.info(
            "Knowledge graph built: %d entities, %d relationship edges",
            G.number_of_nodes(), G.number_of_edges(),
        )
        return G

    # ── Semantic similarity edges ─────────────────────────────────────────────

    def add_semantic_edges(
        self,
        embeddings: np.ndarray,
        pmid_list: List[str],
        similarity_threshold: float = KG_SIMILARITY_THRESHOLD,
    ):
        """
        Add undirected 'semantic_similarity' edges between entity nodes whose
        embedding cosine similarity exceeds *similarity_threshold*.

        embeddings: (N, D) array aligned with pmid_list.
        Only entities whose PMID appears in pmid_list are connected.
        """
        if embeddings is None or len(embeddings) == 0:
            return

        pmid_to_idx = {pmid: i for i, pmid in enumerate(pmid_list)}
        added = 0

        nodes = list(self.graph.nodes(data=True))
        for i, (node_a, data_a) in enumerate(nodes):
            pmids_a = list(
                set(
                    e[2].get("evidence_pmids", [])
                    for e in self.graph.out_edges(node_a, data=True)
                )
            )
            # flatten
            flat_a = []
            for item in pmids_a:
                if isinstance(item, list):
                    flat_a.extend(item)
                else:
                    flat_a.append(item)

            idxs_a = [pmid_to_idx[p] for p in flat_a if p in pmid_to_idx]
            if not idxs_a:
                continue
            emb_a = embeddings[idxs_a].mean(axis=0)
            norm_a = np.linalg.norm(emb_a)
            if norm_a < 1e-9:
                continue
            emb_a /= norm_a

            for node_b, data_b in nodes[i + 1:]:
                if self.graph.has_edge(node_a, node_b):
                    continue
                pmids_b = list(
                    set(
                        e[2].get("evidence_pmids", [])
                        for e in self.graph.out_edges(node_b, data=True)
                    )
                )
                flat_b = []
                for item in pmids_b:
                    if isinstance(item, list):
                        flat_b.extend(item)
                    else:
                        flat_b.append(item)

                idxs_b = [pmid_to_idx[p] for p in flat_b if p in pmid_to_idx]
                if not idxs_b:
                    continue
                emb_b = embeddings[idxs_b].mean(axis=0)
                norm_b = np.linalg.norm(emb_b)
                if norm_b < 1e-9:
                    continue
                emb_b /= norm_b

                sim = float(np.dot(emb_a, emb_b))
                if sim >= similarity_threshold:
                    self.graph.add_edge(
                        node_a, node_b,
                        relationship_type="semantic_similarity",
                        confidence_score=sim,
                        weight=1,
                        evidence_pmids=[],
                    )
                    added += 1

        logger.info("Added %d semantic similarity edges (threshold=%.2f)",
                    added, similarity_threshold)

    # ── Neighborhood query ────────────────────────────────────────────────────

    def get_entity_neighborhood(
        self, entity_name: str, depth: int = 2
    ) -> nx.MultiDiGraph:
        """
        Return the subgraph containing *entity_name* and all nodes within
        *depth* hops (using the undirected projection for hop counting).
        """
        if not self.graph.has_node(entity_name):
            return nx.MultiDiGraph()

        undirected = self.graph.to_undirected()
        neighbors: set = {entity_name}
        frontier = {entity_name}
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for nb in undirected.neighbors(node):
                    if nb not in neighbors:
                        neighbors.add(nb)
                        next_frontier.add(nb)
            frontier = next_frontier

        return self.graph.subgraph(neighbors).copy()

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_graph_statistics(self) -> Dict[str, Any]:
        G = self.graph
        if G.number_of_nodes() == 0:
            return {}

        entity_type_counts: Dict[str, int] = defaultdict(int)
        for _, data in G.nodes(data=True):
            entity_type_counts[data.get("entity_type", "UNKNOWN")] += 1

        rel_type_counts: Dict[str, int] = defaultdict(int)
        for _, _, data in G.edges(data=True):
            rel_type_counts[data.get("relationship_type", "unknown")] += 1

        # Most connected by total degree
        degrees = dict(G.degree())
        most_connected = (
            max(degrees, key=degrees.get) if degrees else None
        )

        # Modularity on undirected projection
        modularity = 0.0
        try:
            und = G.to_undirected()
            part = {n: G.nodes[n].get("community_id", 0) for n in und.nodes()}
            try:
                from community import modularity as _mod
                modularity = _mod(part, und)
            except ImportError:
                from networkx.algorithms.community.quality import modularity as nx_mod
                comm_sets = {}
                for node, cid in part.items():
                    comm_sets.setdefault(cid, set()).add(node)
                modularity = nx_mod(und, list(comm_sets.values()))
        except Exception:
            pass

        return {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "entity_type_counts": dict(entity_type_counts),
            "relationship_type_counts": dict(rel_type_counts),
            "most_connected_entity": most_connected,
            "most_connected_degree": degrees.get(most_connected, 0)
            if most_connected else 0,
            "modularity": modularity,
        }

    # ── Export ────────────────────────────────────────────────────────────────

    def export_to_json(self) -> Dict[str, Any]:
        """Serialise the graph to a JSON-compatible dict for download/API use."""
        nodes = []
        for node, data in self.graph.nodes(data=True):
            nodes.append({"id": node, **{k: v for k, v in data.items()
                                          if isinstance(v, (str, int, float, bool,
                                                            type(None)))}})
        edges = []
        for src, tgt, key, data in self.graph.edges(data=True, keys=True):
            edges.append({
                "source": src,
                "target": tgt,
                "key": key,
                **{k: (json.dumps(v) if isinstance(v, list) else v)
                   for k, v in data.items()},
            })
        return {"nodes": nodes, "edges": edges,
                "statistics": self.get_graph_statistics()}
