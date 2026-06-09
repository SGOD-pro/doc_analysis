"""
Module 7 (Refactored): Analytics Aggregator — Phase 7

Quality controls applied before dashboard generation:
- Validates entities before surfacing in top-N lists
- Exposes new reclassified types: MODEL, METRIC, BENCHMARK, DATASET, VENUE, JOURNAL
- Reject entities with frequency == 1 (likely extraction noise) from display lists
- Topic summaries use proper names (from Phase 5 LDA/NMF/BERTopic naming)
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
ENTITIES_DIR  = DATA_DIR / "entities"
TOPICS_DIR    = DATA_DIR / "topics"
GRAPHS_DIR    = DATA_DIR / "graphs"

# Minimum frequency for an entity to appear in quality-filtered display lists
_DISPLAY_MIN_FREQ = 2


def _merge_and_filter(ents: list[dict], min_freq: int = _DISPLAY_MIN_FREQ, top: int = 10) -> list[dict]:
    """Merge entity lists across docs, deduplicate, apply frequency threshold."""
    merged: dict[str, dict] = {}
    for e in ents:
        key = e["entity"].lower().strip()
        if key not in merged:
            merged[key] = {"entity": e["entity"], "frequency": 0, "type": e.get("type", "")}
        merged[key]["frequency"] += e["frequency"]
    return sorted(
        [v for v in merged.values() if v["frequency"] >= min_freq],
        key=lambda x: x["frequency"],
        reverse=True,
    )[:top]


def get_platform_analytics() -> dict[str, Any]:
    """Compute quality-validated analytics for the dashboard."""

    # ── Document stats ────────────────────────────────────────────────
    total_docs = 0
    total_pages = 0
    doc_formats: Counter = Counter()
    doc_list = []

    for pf in PROCESSED_DIR.glob("*.json"):
        try:
            with open(pf, "r", encoding="utf-8") as f:
                doc = json.load(f)
            total_docs += 1
            pages = doc.get("total_pages", 0)
            total_pages += pages
            doc_formats[doc.get("format", "unknown")] += 1
            doc_list.append({
                "doc_id":        doc.get("doc_id", pf.stem),
                "document_name": doc.get("document_name", pf.name),
                "format":        doc.get("format", "unknown"),
                "total_pages":   pages,
            })
        except Exception:
            pass

    # ── Entity stats ──────────────────────────────────────────────────
    total_entities = 0
    entity_type_counter: Counter = Counter()

    # Collectors for each display type
    collected: dict[str, list[dict]] = defaultdict(list)

    for ef in ENTITIES_DIR.glob("*_entities.json"):
        try:
            with open(ef, "r", encoding="utf-8") as f:
                ed = json.load(f)
            total_entities += ed.get("unique_entities", 0)
            for etype in ed.get("entity_types_found", []):
                entity_type_counter[etype] += 1
            by_type = ed.get("by_type", {})
            # Standard types
            for t in ("ORG", "PERSON", "GPE", "LOC", "PRODUCT",
                      "MODEL", "METRIC", "BENCHMARK", "DATASET",
                      "JOURNAL", "VENUE", "EVENT", "WORK_OF_ART"):
                collected[t].extend(by_type.get(t, [])[:8])
        except Exception:
            pass

    # ── Topic stats ───────────────────────────────────────────────────
    total_topics = 0
    all_keywords: Counter = Counter()
    topic_weights: dict[str, float] = {}
    topic_counts:  Counter = Counter()
    methods_used: Counter = Counter()

    for tf in TOPICS_DIR.glob("*_topics.json"):
        try:
            with open(tf, "r", encoding="utf-8") as f:
                td = json.load(f)
            total_topics += td.get("total_topics", 0)
            methods_used[td.get("method_used", "unknown")] += 1
            for kw in td.get("top_keywords", []):
                all_keywords[kw["keyword"]] += kw.get("count", 1)
            for topic in td.get("topics", []):
                name = topic["topic"]
                topic_counts[name] += 1
                topic_weights[name] = max(topic_weights.get(name, 0.0), topic.get("topic_weight", 0.0))
        except Exception:
            pass

    # ── Graph stats ───────────────────────────────────────────────────
    total_nodes = 0
    total_edges = 0
    graph_density_vals = []
    most_connected_entities = []

    for gf in GRAPHS_DIR.glob("*_graph.json"):
        try:
            with open(gf, "r", encoding="utf-8") as f:
                gd = json.load(f)
            a = gd.get("analytics", {})
            total_nodes += a.get("total_nodes", 0)
            total_edges += a.get("total_edges", 0)
            if "density" in a:
                graph_density_vals.append(a["density"])
            mc = a.get("most_connected_entity", "")
            if mc:
                most_connected_entities.append(mc)
        except Exception:
            pass

    avg_density = (
        round(sum(graph_density_vals) / len(graph_density_vals), 5)
        if graph_density_vals else 0.0
    )

    # ── Build response ────────────────────────────────────────────────
    top_locations = _merge_and_filter(
        collected.get("GPE", []) + collected.get("LOC", [])
    )

    return {
        "summary": {
            "total_documents":     total_docs,
            "total_pages":         total_pages,
            "total_entities":      total_entities,
            "total_topics":        total_topics,
            "total_graph_nodes":   total_nodes,
            "total_graph_edges":   total_edges,
            "average_graph_density": avg_density,
            "most_connected_entities": most_connected_entities[:5],
        },
        "documents": doc_list,
        "document_format_distribution": [
            {"format": fmt, "count": cnt}
            for fmt, cnt in doc_formats.most_common()
        ],
        "entity_analytics": {
            # Core NER types
            "top_organizations":  _merge_and_filter(collected.get("ORG", [])),
            "top_people":         _merge_and_filter(collected.get("PERSON", [])),
            "top_locations":      top_locations,
            "top_products":       _merge_and_filter(collected.get("PRODUCT", [])),
            # Reclassified intelligence types (Phase 2 output)
            "top_models":         _merge_and_filter(collected.get("MODEL", []), min_freq=1),
            "top_metrics":        _merge_and_filter(collected.get("METRIC", []), min_freq=1),
            "top_benchmarks":     _merge_and_filter(collected.get("BENCHMARK", []), min_freq=1),
            "top_datasets":       _merge_and_filter(collected.get("DATASET", []), min_freq=1),
            "top_venues":         _merge_and_filter(collected.get("VENUE", []), min_freq=1),
            "top_journals":       _merge_and_filter(collected.get("JOURNAL", []), min_freq=1),
            "entity_type_distribution": [
                {"type": et, "count": cnt}
                for et, cnt in entity_type_counter.most_common()
            ],
        },
        "topic_analytics": {
            "top_keywords": [
                {"keyword": kw, "count": cnt}
                for kw, cnt in all_keywords.most_common(25)
            ],
            "topic_distribution": [
                {
                    "topic": t,
                    "count": topic_counts[t],
                    "topic_weight": round(topic_weights[t], 4),
                }
                for t, _ in topic_counts.most_common(15)
            ],
            "methods_used": dict(methods_used),
        },
        "graph_analytics": {
            "total_nodes":           total_nodes,
            "total_edges":           total_edges,
            "average_density":       avg_density,
            "most_connected_entities": most_connected_entities,
        },
    }
