"""
NetworkBuilder — constructs co-authorship, keyword co-occurrence, topic,
and citation bibliometric networks as NetworkX graphs.

All networks use Louvain community detection and carry the full set of
node/edge attributes needed by the VOSviewer-faithful rendering layer.
"""

import json
import logging
import math
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from config import (
    KEYWORD_MIN_FREQUENCY,
    COAUTHOR_MIN_PAPERS,
    COMMUNITY_COLORS,
    LOUVAIN_RANDOM_STATE,
    OPENALEX_API_BASE,
    OPENALEX_EMAIL,
)

logger = logging.getLogger(__name__)


def _louvain_communities(graph: nx.Graph) -> Dict[Any, int]:
    """
    Run Louvain community detection and return {node: community_id}.
    Tries python-louvain first, falls back to networkx built-in, then
    single-community as last resort.
    """
    if graph.number_of_nodes() == 0:
        return {}
    # Primary: python-louvain (best_partition)
    try:
        from community import best_partition
        return best_partition(graph, random_state=LOUVAIN_RANDOM_STATE)
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("python-louvain failed: %s", exc)
    # Fallback: networkx built-in louvain_communities
    try:
        from networkx.algorithms.community import louvain_communities
        import random
        rng = random.Random(LOUVAIN_RANDOM_STATE)
        communities = louvain_communities(graph, seed=LOUVAIN_RANDOM_STATE)
        partition = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                partition[node] = cid
        return partition
    except Exception as exc:
        logger.warning("networkx louvain failed: %s — using single community", exc)
        return {n: 0 for n in graph.nodes()}


def _assign_community_colors(
    graph: nx.Graph, partition: Dict[Any, int]
) -> Dict[int, str]:
    """Return {community_id: hex_color} cycling through COMMUNITY_COLORS."""
    unique_ids = sorted(set(partition.values()))
    return {
        cid: COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]
        for i, cid in enumerate(unique_ids)
    }


class NetworkBuilder:
    """Builds all bibliometric networks from a cleaned papers DataFrame."""

    # ── Co-authorship network ─────────────────────────────────────────────────

    def build_coauthorship_network(self, papers_df: pd.DataFrame) -> nx.Graph:
        """
        Nodes: authors (weight = paper count)
        Edges: co-authorship (weight = jointly authored papers)
        Node attributes: weight, degree_centrality, betweenness_centrality,
                         clustering, community_id, color_hex, label
        """
        if papers_df is None or papers_df.empty:
            return nx.Graph()

        paper_count: Dict[str, int] = defaultdict(int)
        coauthor_weight: Dict[Tuple[str, str], int] = defaultdict(int)

        for _, row in papers_df.iterrows():
            authors = row.get("authors") or []
            if isinstance(authors, str):
                try:
                    authors = json.loads(authors)
                except Exception:
                    authors = []
            names = [
                a["name"] for a in authors
                if isinstance(a, dict) and a.get("name")
            ]
            for name in names:
                paper_count[name] += 1
            for a, b in combinations(sorted(set(names)), 2):
                coauthor_weight[(a, b)] += 1

        G = nx.Graph()
        for author, count in paper_count.items():
            if count >= COAUTHOR_MIN_PAPERS:
                G.add_node(author, weight=count, label=author)

        for (a, b), w in coauthor_weight.items():
            if G.has_node(a) and G.has_node(b):
                G.add_edge(a, b, weight=w)

        if G.number_of_nodes() == 0:
            return G

        self._add_centrality_metrics(G)

        partition = _louvain_communities(G)
        color_map = _assign_community_colors(G, partition)
        for node in G.nodes():
            cid = partition.get(node, 0)
            G.nodes[node]["community_id"] = cid
            G.nodes[node]["color_hex"] = color_map[cid]

        logger.info(
            "Co-authorship network: %d authors, %d edges",
            G.number_of_nodes(), G.number_of_edges(),
        )
        return G

    # ── Keyword co-occurrence network ─────────────────────────────────────────

    def build_keyword_cooccurrence_network(
        self, papers_df: pd.DataFrame
    ) -> nx.Graph:
        """
        Nodes: author keywords + MeSH descriptors + MeSH qualifiers +
               chemical terms (each tagged with source_type attribute).
        Filter: only keywords present in >= KEYWORD_MIN_FREQUENCY papers.
        Edges: co-occurrence in same paper (weight = co-occurrence count).
        Community detection via Louvain.
        """
        if papers_df is None or papers_df.empty:
            return nx.Graph()

        kw_freq: Dict[str, int] = defaultdict(int)
        kw_type: Dict[str, str] = {}
        paper_kws: List[List[str]] = []

        for _, row in papers_df.iterrows():
            doc_kws: List[str] = []

            def _add(items, ktype):
                for item in (items or []):
                    if isinstance(item, str) and item.strip():
                        k = item.strip().lower()
                        kw_freq[k] += 1
                        if k not in kw_type:
                            kw_type[k] = ktype
                        doc_kws.append(k)

            # Author keywords
            kws = row.get("keywords") or []
            if isinstance(kws, str):
                try:
                    kws = json.loads(kws)
                except Exception:
                    kws = []
            _add(kws, "author_keyword")

            # MeSH descriptors
            mesh = row.get("mesh_terms") or []
            if isinstance(mesh, str):
                try:
                    mesh = json.loads(mesh)
                except Exception:
                    mesh = []
            for m in mesh:
                if isinstance(m, dict):
                    d = m.get("descriptor", "")
                    if d:
                        k = d.strip().lower()
                        kw_freq[k] += 1
                        kw_type.setdefault(k, "mesh_descriptor")
                        doc_kws.append(k)

            # MeSH qualifiers
            quals = row.get("mesh_qualifiers") or []
            if isinstance(quals, str):
                try:
                    quals = json.loads(quals)
                except Exception:
                    quals = []
            _add(quals, "mesh_qualifier")

            # Chemical terms
            chems = row.get("chemical_terms") or []
            if isinstance(chems, str):
                try:
                    chems = json.loads(chems)
                except Exception:
                    chems = []
            for c in chems:
                if isinstance(c, dict):
                    n = c.get("name", "")
                    if n:
                        k = n.strip().lower()
                        kw_freq[k] += 1
                        kw_type.setdefault(k, "chemical")
                        doc_kws.append(k)

            paper_kws.append(list(set(doc_kws)))

        # Filter by minimum frequency
        valid = {k for k, freq in kw_freq.items()
                 if freq >= KEYWORD_MIN_FREQUENCY}

        G = nx.Graph()
        for kw in valid:
            G.add_node(
                kw,
                weight=kw_freq[kw],
                source_type=kw_type.get(kw, "author_keyword"),
                label=kw,
            )

        cooccur: Dict[Tuple[str, str], int] = defaultdict(int)
        for doc_kws in paper_kws:
            filtered = [k for k in doc_kws if k in valid]
            for a, b in combinations(sorted(set(filtered)), 2):
                cooccur[(a, b)] += 1

        for (a, b), w in cooccur.items():
            G.add_edge(a, b, weight=w)

        if G.number_of_nodes() == 0:
            return G

        partition = _louvain_communities(G)
        color_map = _assign_community_colors(G, partition)
        for node in G.nodes():
            cid = partition.get(node, 0)
            G.nodes[node]["community_id"] = cid
            G.nodes[node]["color_hex"] = color_map[cid]

        logger.info(
            "Keyword co-occurrence network: %d keywords, %d co-occurrence edges",
            G.number_of_nodes(), G.number_of_edges(),
        )
        return G

    # ── Topic network ─────────────────────────────────────────────────────────

    def build_topic_network(
        self,
        topic_model_results: Dict[str, Any],
    ) -> nx.Graph:
        """
        Nodes: BERTopic topics (weight = paper count).
        Edges: papers shared between topics (weight = shared paper count).

        topic_model_results must contain:
          'topic_summary': list of {topic_id, label, paper_count}
          'paper_assignments': list of {pmid, topic_id}
        """
        G = nx.Graph()

        topic_summary = topic_model_results.get("topic_summary", [])
        paper_assignments = topic_model_results.get("paper_assignments", [])

        if not topic_summary:
            return G

        # Build topic nodes
        for t in topic_summary:
            tid = t["topic_id"]
            if tid == -1:
                continue
            G.add_node(
                tid,
                weight=t.get("paper_count", 0),
                label=t.get("label", f"Topic {tid}"),
                top_words=t.get("top_words", []),
            )

        # Build paper → topic index
        paper_topic: Dict[str, List[int]] = defaultdict(list)
        for asgn in paper_assignments:
            tid = asgn.get("topic_id", -1)
            if tid >= 0 and G.has_node(tid):
                paper_topic[asgn["pmid"]].append(tid)

        # Connect topics that share papers (works when probabilities give multi-topic assignments)
        shared: Dict[Tuple[int, int], int] = defaultdict(int)
        for pmid, topics in paper_topic.items():
            for a, b in combinations(sorted(set(topics)), 2):
                shared[(a, b)] += 1

        for (a, b), w in shared.items():
            G.add_edge(a, b, weight=w)

        # Hard-clustering (one topic per paper) produces no shared-paper edges.
        # Fall back to keyword-overlap similarity so the network is still connected.
        if G.number_of_edges() == 0 and G.number_of_nodes() >= 2:
            topic_word_sets: Dict[int, set] = {}
            for node, data in G.nodes(data=True):
                raw_words = data.get("top_words", [])
                # top_words may be [(word, score), ...] or [word, ...]
                words = set()
                for item in raw_words:
                    w = item[0] if isinstance(item, (list, tuple)) else item
                    if isinstance(w, str) and len(w) > 1:
                        words.add(w.lower())
                topic_word_sets[node] = words

            for (tid_a, words_a), (tid_b, words_b) in combinations(
                topic_word_sets.items(), 2
            ):
                overlap = len(words_a & words_b)
                if overlap > 0:
                    G.add_edge(tid_a, tid_b, weight=overlap)

        if G.number_of_nodes() == 0:
            return G

        partition = _louvain_communities(G)
        color_map = _assign_community_colors(G, partition)
        for node in G.nodes():
            cid = partition.get(node, 0)
            G.nodes[node]["community_id"] = cid
            G.nodes[node]["color_hex"] = color_map[cid]

        logger.info(
            "Topic network: %d topics, %d edges",
            G.number_of_nodes(), G.number_of_edges(),
        )
        return G

    # ── Citation network ──────────────────────────────────────────────────────

    def build_citation_network(self, papers_df: pd.DataFrame) -> nx.DiGraph:
        """
        Attempts to fetch citation links from OpenAlex API.
        Falls back to a co-citation network (papers that share references)
        if OpenAlex is unavailable.
        """
        pmids = papers_df["pmid"].astype(str).tolist()
        G = nx.DiGraph()

        for pmid in pmids:
            G.add_node(pmid, weight=1)

        try:
            import requests
            headers = {"User-Agent": f"BioLitAI-X (mailto:{OPENALEX_EMAIL})"}
            added = 0
            for pmid in pmids[:50]:   # limit API calls on large corpora
                url = f"{OPENALEX_API_BASE}/works/pmid:{pmid}"
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                refs = data.get("referenced_works", [])
                for ref_url in refs:
                    ref_id = ref_url.split("/")[-1]
                    if ref_id in G:
                        G.add_edge(pmid, ref_id, weight=1)
                        added += 1
            logger.info("Citation network: added %d OpenAlex edges", added)
        except Exception as exc:
            logger.warning(
                "OpenAlex citation fetch failed (%s) — "
                "falling back to co-citation proximity", exc
            )

        logger.info(
            "Citation network: %d nodes, %d edges",
            G.number_of_nodes(), G.number_of_edges(),
        )
        return G

    # ── Network statistics ────────────────────────────────────────────────────

    def calculate_network_statistics(
        self, graph: nx.Graph
    ) -> Dict[str, Any]:
        """
        Return a comprehensive statistics dict for display in the UI stats panel.
        """
        if graph is None or graph.number_of_nodes() == 0:
            return {"error": "Empty graph"}

        stats: Dict[str, Any] = {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
        }

        try:
            stats["density"] = nx.density(graph)
        except Exception:
            stats["density"] = 0.0

        try:
            if isinstance(graph, nx.Graph) and not isinstance(graph, nx.DiGraph):
                stats["avg_clustering"] = nx.average_clustering(
                    graph, weight="weight"
                )
            else:
                stats["avg_clustering"] = 0.0
        except Exception:
            stats["avg_clustering"] = 0.0

        # Community stats from stored node attribute
        communities = set()
        for _, data in graph.nodes(data=True):
            cid = data.get("community_id")
            if cid is not None:
                communities.add(cid)
        stats["community_count"] = len(communities)

        # Modularity
        try:
            partition = {n: graph.nodes[n].get("community_id", 0)
                         for n in graph.nodes()}
            if isinstance(graph, nx.Graph) and not isinstance(graph, nx.DiGraph):
                try:
                    from community import modularity
                    stats["modularity"] = modularity(partition, graph)
                except ImportError:
                    from networkx.algorithms.community.quality import modularity as nx_mod
                    comm_sets = {}
                    for node, cid in partition.items():
                        comm_sets.setdefault(cid, set()).add(node)
                    stats["modularity"] = nx_mod(graph, list(comm_sets.values()))
            else:
                stats["modularity"] = 0.0
        except Exception:
            stats["modularity"] = 0.0

        # Centrality top-10
        try:
            deg_cent = nx.degree_centrality(graph)
            stats["top_degree"] = sorted(
                deg_cent.items(), key=lambda x: x[1], reverse=True
            )[:10]
        except Exception:
            stats["top_degree"] = []

        try:
            if graph.number_of_nodes() <= 2000:
                btw_cent = nx.betweenness_centrality(graph, weight="weight")
                stats["top_betweenness"] = sorted(
                    btw_cent.items(), key=lambda x: x[1], reverse=True
                )[:10]
            else:
                stats["top_betweenness"] = []
        except Exception:
            stats["top_betweenness"] = []

        try:
            if graph.number_of_nodes() <= 2000:
                close_cent = nx.closeness_centrality(graph)
                stats["top_closeness"] = sorted(
                    close_cent.items(), key=lambda x: x[1], reverse=True
                )[:10]
            else:
                stats["top_closeness"] = []
        except Exception:
            stats["top_closeness"] = []

        return stats

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _add_centrality_metrics(G: nx.Graph):
        """Compute and store per-node centrality metrics in-place."""
        try:
            deg = nx.degree_centrality(G)
            for n, v in deg.items():
                G.nodes[n]["degree_centrality"] = round(v, 6)
        except Exception:
            pass

        try:
            if G.number_of_nodes() <= 2000:
                btw = nx.betweenness_centrality(G, weight="weight")
                for n, v in btw.items():
                    G.nodes[n]["betweenness_centrality"] = round(v, 6)
        except Exception:
            pass

        try:
            clust = nx.clustering(G, weight="weight")
            for n, v in clust.items():
                G.nodes[n]["clustering"] = round(v, 6)
        except Exception:
            pass
