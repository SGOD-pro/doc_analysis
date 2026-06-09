"""
Module 5 (Refactored): Knowledge Graph Builder — Phase 3

Key improvements:
- Sentence-level co-occurrence (not page-level)
- Configurable entity window per sentence
- Edge weight threshold pruning (default: weight < 2 removed)
- Isolated node removal
- Extended metrics: Degree, Betweenness, Closeness, PageRank
- Clean relationship typing from RELATIONSHIP_PATTERNS
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any
import networkx as nx

from backend.core.entity_validator import get_config_value

DATA_DIR = Path("data")
GRAPHS_DIR = DATA_DIR / "graphs"

RELATIONSHIP_PATTERNS = [
    ("developed",   ["developed", "created", "built", "designed", "invented", "launched", "introduced"]),
    ("acquired",    ["acquired", "bought", "purchased", "merged", "acquired by"]),
    ("authored",    ["authored", "wrote", "published", "presented", "submitted", "proposed"]),
    ("located_in",  ["located in", "based in", "headquartered in", "founded in", "operates in"]),
    ("works_for",   ["works for", "employed by", "joined", "leads", "headed by", "member of"]),
    ("partner_of",  ["partnered with", "collaborated with", "allied with", "worked with"]),
    ("funded_by",   ["funded by", "invested in", "backed by", "supported by"]),
    ("uses",        ["uses", "utilizes", "employs", "leverages", "based on", "built on"]),
    ("outperforms", ["outperforms", "beats", "surpasses", "exceeds", "better than"]),
    ("evaluated_on",["evaluated on", "tested on", "benchmarked on", "measured on"]),
    ("trained_on",  ["trained on", "fine-tuned on", "pre-trained on"]),
]


def ensure_dirs():
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def _find_relationship(sent_text: str) -> str:
    """Detect relationship type from sentence text."""
    t = sent_text.lower()
    for rel_type, keywords in RELATIONSHIP_PATTERNS:
        for kw in keywords:
            if kw in t:
                return rel_type
    return "co-occurs_with"


def _extract_sentence_cooccurrence_edges(
    sentence_mentions: list[dict],
    pages: list[dict],
    max_ents_per_sentence: int = 8,
    weight_threshold: int = 2,
) -> list[dict]:
    """
    Phase 3: Sentence-level co-occurrence.
    Two entities share an edge only when they appear in the same sentence.
    Edges below weight_threshold are pruned.
    """
    # Build sentence text lookup: (page, sent_idx) -> text
    # We need the original spaCy sentences — approximate by splitting page text
    page_text_map = {p.get("page", 0): p.get("text", "") for p in pages}

    # Group mentions by (page, sentence_idx)
    sent_groups: dict[tuple, list[str]] = defaultdict(list)
    for mention in sentence_mentions:
        key = (mention["page"], mention.get("sentence_idx", 0))
        sent_groups[key].append(mention["entity"])

    edge_counter: Counter = Counter()
    edge_sentences: dict[tuple, str] = {}  # store example sentence per edge

    for (page_num, sent_idx), ents in sent_groups.items():
        # Cap entities per sentence to avoid explosion with entity-dense tables
        unique_ents = list(dict.fromkeys(ents))[:max_ents_per_sentence]

        for i in range(len(unique_ents)):
            for j in range(i + 1, len(unique_ents)):
                pair = tuple(sorted([unique_ents[i], unique_ents[j]]))
                # Ignore self-loops
                if pair[0] == pair[1]:
                    continue
                edge_counter[pair] += 1
                # Store sentence text for relationship detection
                if pair not in edge_sentences:
                    full_text = page_text_map.get(page_num, "")
                    # Get approximate sentence from text
                    edge_sentences[pair] = full_text

    # Build edge list with pruning
    edges = []
    for (src, dst), weight in edge_counter.items():
        if weight < weight_threshold:
            continue  # Phase 3: prune low-confidence edges
        rel = _find_relationship(edge_sentences.get((src, dst), ""))
        edges.append({
            "source":       src,
            "target":       dst,
            "relationship": rel,
            "weight":       weight,
        })

    # Sort by weight descending, keep top 300
    edges.sort(key=lambda x: x["weight"], reverse=True)
    return edges[:300]


def _compute_graph_metrics(G: nx.Graph) -> dict[str, dict]:
    """Phase 3: Compute all four centrality metrics."""
    n = G.number_of_nodes()
    metrics = {
        "degree":      {},
        "betweenness": {},
        "closeness":   {},
        "pagerank":    {},
    }
    if n == 0:
        return metrics

    metrics["degree"] = nx.degree_centrality(G)

    if n > 1:
        k = min(100, n)
        try:
            metrics["betweenness"] = nx.betweenness_centrality(G, k=k, normalized=True)
        except Exception:
            metrics["betweenness"] = {}

        try:
            metrics["closeness"] = nx.closeness_centrality(G)
        except Exception:
            metrics["closeness"] = {}

        try:
            metrics["pagerank"] = nx.pagerank(G, alpha=0.85, max_iter=100)
        except Exception:
            metrics["pagerank"] = {}

    return metrics


def build_knowledge_graph(
    doc_data: dict[str, Any],
    entity_data: dict[str, Any],
) -> dict[str, Any]:
    """Build a high-quality knowledge graph for a document."""
    ensure_dirs()

    doc_id   = doc_data["doc_id"]
    doc_name = doc_data["document_name"]
    pages    = doc_data.get("pages", [])
    entities = entity_data.get("entities", [])

    # Prefer sentence_mentions (new extractor); fall back to page mentions
    sentence_mentions = entity_data.get(
        "sentence_mentions",
        entity_data.get("mentions", [])
    )

    # Config
    max_ents = get_config_value("graph_max_sentence_entities", 8)
    weight_thresh = get_config_value("graph_edge_weight_threshold", 2)

    # Build node map: entity text → node ID
    G = nx.Graph()
    node_map: dict[str, str] = {}

    for ent in entities:
        raw_label = ent["entity"]
        node_id = re.sub(r"[^\w]", "_", raw_label)
        G.add_node(
            node_id,
            label=raw_label,
            type=ent["type"],
            type_label=ent.get("type_label", ent["type"]),
            frequency=ent["frequency"],
        )
        node_map[raw_label] = node_id

    # Extract edges (sentence-level)
    edges = _extract_sentence_cooccurrence_edges(
        sentence_mentions, pages,
        max_ents_per_sentence=max_ents,
        weight_threshold=weight_thresh,
    )

    for edge in edges:
        src_id = node_map.get(edge["source"])
        dst_id = node_map.get(edge["target"])
        if src_id and dst_id and src_id != dst_id:
            if G.has_edge(src_id, dst_id):
                G[src_id][dst_id]["weight"] += edge["weight"]
            else:
                G.add_edge(src_id, dst_id,
                           relationship=edge["relationship"],
                           weight=edge["weight"])

    # Phase 3: Remove isolated nodes (no edges)
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    # Compute metrics
    metrics = _compute_graph_metrics(G)
    dc  = metrics["degree"]
    bc  = metrics["betweenness"]
    cc  = metrics["closeness"]
    pr  = metrics["pagerank"]

    density = nx.density(G) if G.number_of_nodes() > 1 else 0.0

    most_connected_id = max(dc, key=dc.get) if dc else ""
    most_connected = G.nodes[most_connected_id].get("label", most_connected_id) if most_connected_id else ""

    # Serialize
    nodes_out = []
    for node_id, attrs in G.nodes(data=True):
        nodes_out.append({
            "id":          node_id,
            "label":       attrs.get("label", node_id),
            "type":        attrs.get("type", "UNKNOWN"),
            "type_label":  attrs.get("type_label", "Unknown"),
            "frequency":   attrs.get("frequency", 1),
            "degree":      G.degree(node_id),
            "centrality":  round(dc.get(node_id, 0), 5),
            "betweenness": round(bc.get(node_id, 0), 5),
            "closeness":   round(cc.get(node_id, 0), 5),
            "pagerank":    round(pr.get(node_id, 0), 5),
        })

    # Sort nodes by pagerank for more useful default ordering
    nodes_out.sort(key=lambda x: x["pagerank"], reverse=True)

    edges_out = []
    for u, v, attrs in G.edges(data=True):
        edges_out.append({
            "source":       u,
            "target":       v,
            "relationship": attrs.get("relationship", "co-occurs_with"),
            "weight":       attrs.get("weight", 1),
        })

    result = {
        "doc_id":        doc_id,
        "document_name": doc_name,
        "nodes":         nodes_out,
        "edges":         edges_out,
        "analytics": {
            "total_nodes":           G.number_of_nodes(),
            "total_edges":           G.number_of_edges(),
            "density":               round(density, 5),
            "most_connected_entity": most_connected,
            "average_degree":        round(
                sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1), 3
            ),
            "isolated_removed":      len(isolated),
        },
    }

    out_path = GRAPHS_DIR / f"{doc_id}_graph.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def load_graph(doc_id: str) -> dict[str, Any] | None:
    path = GRAPHS_DIR / f"{doc_id}_graph.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_all_graphs() -> dict[str, Any]:
    """Merge all document graphs; re-apply pruning on merged weights."""
    ensure_dirs()
    weight_thresh = get_config_value("graph_edge_weight_threshold", 2)

    all_nodes: dict[str, dict] = {}
    edge_weights: Counter = Counter()
    edge_rels: dict[tuple, str] = {}

    for gf in GRAPHS_DIR.glob("*_graph.json"):
        try:
            with open(gf, "r", encoding="utf-8") as f:
                gdata = json.load(f)
            for node in gdata.get("nodes", []):
                nid = node["id"]
                if nid in all_nodes:
                    all_nodes[nid]["frequency"] += node.get("frequency", 1)
                else:
                    all_nodes[nid] = dict(node)
            for edge in gdata.get("edges", []):
                pair = tuple(sorted([edge["source"], edge["target"]]))
                edge_weights[pair] += edge.get("weight", 1)
                if pair not in edge_rels:
                    edge_rels[pair] = edge.get("relationship", "co-occurs_with")
        except Exception:
            pass

    # Apply threshold to merged edges
    merged_edges = [
        {"source": s, "target": t, "relationship": edge_rels.get((s, t), "co-occurs_with"), "weight": w}
        for (s, t), w in edge_weights.items()
        if w >= weight_thresh
    ]
    merged_edges.sort(key=lambda x: x["weight"], reverse=True)
    merged_edges = merged_edges[:500]

    nodes_list = list(all_nodes.values())
    G = nx.Graph()
    edge_node_ids = {e["source"] for e in merged_edges} | {e["target"] for e in merged_edges}

    # Only include nodes that have at least one edge
    nodes_list = [n for n in nodes_list if n["id"] in edge_node_ids]
    for node in nodes_list:
        G.add_node(node["id"])
    for edge in merged_edges:
        G.add_edge(edge["source"], edge["target"], weight=edge["weight"])

    density = nx.density(G) if G.number_of_nodes() > 1 else 0.0
    dc = nx.degree_centrality(G) if G.number_of_nodes() > 0 else {}
    pr = {}
    if G.number_of_nodes() > 1:
        try:
            pr = nx.pagerank(G, alpha=0.85)
        except Exception:
            pass

    most_connected_id = max(pr, key=pr.get) if pr else (max(dc, key=dc.get) if dc else "")
    most_connected = all_nodes.get(most_connected_id, {}).get("label", most_connected_id)

    # Annotate merged nodes with metrics
    for node in nodes_list:
        nid = node["id"]
        node["centrality"] = round(dc.get(nid, 0), 5)
        node["pagerank"]   = round(pr.get(nid, 0), 5)

    return {
        "nodes": nodes_list,
        "edges": merged_edges,
        "analytics": {
            "total_nodes":           len(nodes_list),
            "total_edges":           len(merged_edges),
            "density":               round(density, 5),
            "most_connected_entity": most_connected,
            "average_degree":        round(
                sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1), 3
            ),
        },
    }


# Required import for node_id generation in build_knowledge_graph
import re
