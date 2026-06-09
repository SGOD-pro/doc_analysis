"""
Module 3 (Refactored): Entity Extractor
Hardened pipeline: spaCy NER → Validator → Normalizer → Storage

Architecture (Phase 8):
  Extractor → entity_validator.validate_entity() → Graph Builder → Analytics

Adapter pattern ensures this module can be replaced with SciSpacy, GLiNER,
HuggingFace NER, or LLM-based extraction without changing downstream logic.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any

from backend.core.entity_validator import (
    validate_entities_batch,
    canonical_key,
    ENTITY_LABELS,
)

DATA_DIR = Path("data")
ENTITIES_DIR = DATA_DIR / "entities"

# ── NLP backend (lazy loaded) ─────────────────────────────────────────────────
_nlp = None


def ensure_dirs():
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)


def _get_nlp():
    """Lazily load spaCy. Swap this function to use a different NER backend."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            for model in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
                try:
                    _nlp = spacy.load(model)
                    break
                except OSError:
                    continue
            if _nlp is None:
                raise RuntimeError(
                    "No spaCy model found. Run: python -m spacy download en_core_web_sm"
                )
        except ImportError:
            raise RuntimeError("spaCy not installed. Run: uv add spacy")
    return _nlp


# ── Sentence-aware text processing ───────────────────────────────────────────
def _extract_raw_entities_from_text(text: str, nlp) -> list[tuple[str, str, int]]:
    """
    Process text with spaCy and return raw (text, label, sent_idx) tuples.
    Sentence index is preserved for graph builder sentence-level co-occurrence.
    """
    max_len = nlp.max_length - 100
    results: list[tuple[str, str, int]] = []

    def _process_chunk(chunk: str, sent_offset: int = 0):
        doc = nlp(chunk)
        for sent_idx, sent in enumerate(doc.sents):
            for ent in sent.ents:
                raw = ent.text.strip()
                if raw:
                    results.append((raw, ent.label_, sent_idx + sent_offset))

    if len(text) <= max_len:
        _process_chunk(text)
    else:
        # Split on sentence-friendly boundaries
        offset = 0
        chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]
        for chunk in chunks:
            _process_chunk(chunk, offset)
            # Rough sentence offset estimate
            offset += chunk.count(". ") + chunk.count("! ") + chunk.count("? ")

    return results


# ── Main extraction function ──────────────────────────────────────────────────
def extract_entities(doc_data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract, validate, normalize, and reclassify entities from a processed document.

    Pipeline:
      spaCy NER  →  entity_validator.validate_entity()  →  JSON storage
    """
    ensure_dirs()
    nlp = _get_nlp()

    doc_id = doc_data["doc_id"]
    doc_name = doc_data["document_name"]
    pages = doc_data.get("pages", [])

    # ── Per-entity aggregation ────────────────────────────────────────────────
    # canonical_key → {type, type_label, frequency, pages, sentences}
    entity_registry: dict[str, dict] = {}
    # Store sentence-level mentions for graph builder
    sentence_mentions: list[dict] = []
    # Store page-level mentions for backward compat
    page_mentions: list[dict] = []

    for page in pages:
        page_num = page.get("page", 0)
        text = page.get("text", "")
        if not text.strip():
            continue

        raw_tuples = _extract_raw_entities_from_text(text, nlp)

        # Batch validate for this page
        validated = validate_entities_batch(
            [(raw, label) for raw, label, _ in raw_tuples]
        )
        validated_map = {canonical_key(v.text): v for v in validated}

        for raw_text, label, sent_idx in raw_tuples:
            from backend.core.entity_validator import validate_entity
            v = validate_entity(raw_text, label)
            if v is None:
                continue

            ck = canonical_key(v.text)

            if ck not in entity_registry:
                entity_registry[ck] = {
                    "entity": v.text,
                    "canonical": ck,
                    "type": v.type,
                    "type_label": v.type_label,
                    "frequency": 0,
                    "pages": set(),
                    "confidence": v.confidence,
                }
            entity_registry[ck]["frequency"] += 1
            entity_registry[ck]["pages"].add(page_num)

            # Record for graph builder (sentence-level)
            sentence_mentions.append({
                "entity": v.text,
                "canonical": ck,
                "type": v.type,
                "page": page_num,
                "sentence_idx": sent_idx,
            })
            page_mentions.append({
                "entity": v.text,
                "type": v.type,
                "page": page_num,
            })

    # ── Build output entities list ────────────────────────────────────────────
    entities = []
    for reg in entity_registry.values():
        entities.append({
            "entity":     reg["entity"],
            "type":       reg["type"],
            "type_label": reg["type_label"],
            "frequency":  reg["frequency"],
            "page_count": len(reg["pages"]),
            "confidence": reg["confidence"],
        })

    # Sort by frequency desc
    entities.sort(key=lambda x: x["frequency"], reverse=True)

    # ── Group by type ─────────────────────────────────────────────────────────
    by_type: dict[str, list] = defaultdict(list)
    for e in entities:
        by_type[e["type"]].append(e)

    result = {
        "doc_id":              doc_id,
        "document_name":       doc_name,
        "total_entities":      len(entities),
        "unique_entities":     len(entity_registry),
        "entity_types_found":  sorted(by_type.keys()),
        "entities":            entities,
        "by_type":             {k: v[:15] for k, v in by_type.items()},
        # sentence_mentions is used by graph_builder for sentence-level co-occurrence
        "sentence_mentions":   sentence_mentions,
        # page_mentions kept for backward compat
        "mentions":            page_mentions,
        "top_entities":        entities[:25],
    }

    out_path = ENTITIES_DIR / f"{doc_id}_entities.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def load_entities(doc_id: str) -> dict[str, Any] | None:
    path = ENTITIES_DIR / f"{doc_id}_entities.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_all_entities() -> dict[str, Any]:
    """Aggregate and validate entities across all documents (Phase 7)."""
    ensure_dirs()

    combined: dict[str, dict] = {}
    doc_count = 0

    for ent_file in ENTITIES_DIR.glob("*_entities.json"):
        try:
            with open(ent_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            doc_count += 1
            for e in data.get("entities", []):
                ck = canonical_key(e["entity"])
                if ck not in combined:
                    combined[ck] = {
                        "entity":     e["entity"],
                        "type":       e["type"],
                        "type_label": e["type_label"],
                        "frequency":  0,
                        "confidence": e.get("confidence", 1.0),
                    }
                combined[ck]["frequency"] += e["frequency"]
        except Exception:
            pass

    all_entities = sorted(combined.values(), key=lambda x: x["frequency"], reverse=True)

    by_type: dict[str, list] = defaultdict(list)
    for e in all_entities:
        by_type[e["type"]].append(e)

    # Phase 7 quality controls — filter display lists
    def _quality_top(etype: str, n: int = 10) -> list[dict]:
        items = by_type.get(etype, [])
        # Require confidence >= 0.85 and frequency > 1
        return [e for e in items if e.get("confidence", 1.0) >= 0.85 and e["frequency"] > 1][:n]

    return {
        "total_documents_analyzed":  doc_count,
        "total_unique_entities":     len(combined),
        "total_entity_mentions":     sum(e["frequency"] for e in all_entities),
        "top_entities":              all_entities[:50],
        "by_type":                   {k: v[:12] for k, v in by_type.items()},
        "top_organizations":         _quality_top("ORG"),
        "top_people":                _quality_top("PERSON"),
        "top_locations":             _quality_top("GPE") + _quality_top("LOC"),
        "top_products":              _quality_top("PRODUCT"),
        "top_models":                _quality_top("MODEL"),
        "top_metrics":               _quality_top("METRIC"),
        "top_benchmarks":            _quality_top("BENCHMARK"),
    }
