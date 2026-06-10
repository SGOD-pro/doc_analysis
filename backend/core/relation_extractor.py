"""
PHASE 3: Relation Extractor

Extracts semantic relations between entities using spaCy dependency parsing.
Replaces naive sentence co-occurrence with proper Subject-Verb-Object triples.

Uses:
- spaCy dependency parser (no ML models, no LLMs)
- Pattern-based relation extraction
- Confidence scoring based on parse quality

Output format:
[
    {
        "source": "Entity A",
        "relation": "developed",
        "target": "Transformer",
        "confidence": 0.95,
        "sentence": "...",
    },
    ...
]
"""

import re
from typing import Any
from collections import defaultdict


# ── Relation Patterns ──────────────────────────────────────────────────────────
# Maps dependency patterns to relation types

RELATION_VERBS = {
    # Development/Creation
    "develop": "developed",
    "developed": "developed",
    "create": "created",
    "created": "created",
    "build": "built",
    "built": "built",
    "design": "designed",
    "designed": "designed",
    "invent": "invented",
    "invented": "invented",
    "propose": "proposed",
    "proposed": "proposed",
    "introduce": "introduced",
    "introduced": "introduced",
    
    # Acquisition/Partnership
    "acquire": "acquired",
    "acquired": "acquired",
    "buy": "acquired",
    "bought": "acquired",
    "merge": "merged_with",
    "merged": "merged_with",
    "partner": "partnered_with",
    "partnered": "partnered_with",
    "collaborate": "collaborated_with",
    "collaborated": "collaborated_with",
    
    # Employment/Membership
    "work": "works_for",
    "working": "works_for",
    "employ": "employs",
    "employed": "employed_by",
    "join": "joined",
    "joined": "joined",
    "lead": "leads",
    "leading": "leads",
    "led": "led_by",
    
    # Location
    "locate": "located_in",
    "located": "located_in",
    "base": "based_in",
    "based": "based_in",
    "headquarter": "headquartered_in",
    "headquartered": "headquartered_in",
    "found": "founded_in",
    "founded": "founded_in",
    
    # Usage/Application
    "use": "uses",
    "using": "uses",
    "utilize": "utilizes",
    "utilized": "utilizes",
    "employ": "employs",
    "apply": "applies",
    "applied": "applies",
    "leverage": "leverages",
    "leveraged": "leverages",
    
    # Evaluation/Comparison
    "evaluate": "evaluated_on",
    "evaluated": "evaluated_on",
    "test": "tested_on",
    "tested": "tested_on",
    "benchmark": "benchmarked_on",
    "benchmarked": "benchmarked_on",
    "measure": "measured_on",
    "measured": "measured_on",
    "outperform": "outperforms",
    "outperformed": "outperforms",
    "beat": "outperforms",
    "beats": "outperforms",
    "beat": "outperforms",
    "surpass": "surpasses",
    "surpassed": "surpasses",
    "exceed": "exceeds",
    "exceeded": "exceeds",
    
    # Training/Learning
    "train": "trained_on",
    "trained": "trained_on",
    "fine-tune": "fine_tuned_on",
    "fine-tuned": "fine_tuned_on",
    "pretrain": "pretrained_on",
    "pretrained": "pretrained_on",
    
    # Authorship
    "author": "authored",
    "authored": "authored",
    "write": "wrote",
    "wrote": "wrote",
    "publish": "published",
    "published": "published",
    "present": "presented",
    "presented": "presented",
}

# Verbs that indicate strong relations
STRONG_RELATION_VERBS = {
    "developed", "created", "built", "designed", "invented",
    "acquired", "merged", "founded", "authored", "published",
}


def _get_lemma(verb_token) -> str:
    """Get lemma form of a verb token."""
    return verb_token.lemma_.lower()


def _extract_svo_triples(doc, entity_spans: dict[str, list]) -> list[dict]:
    """
    Extract Subject-Verb-Object triples from parsed doc.
    
    Args:
        doc: spaCy Doc object
        entity_spans: Dict mapping entity text to list of (start, end) char spans
    
    Returns:
        List of (source, relation, target) tuples
    """
    triples = []
    
    # Build reverse index: token index -> entity mentions at that position
    token_to_entity: dict[int, list[str]] = defaultdict(list)
    for entity_text, spans in entity_spans.items():
        for start, end in spans:
            # Find tokens within this span
            for token in doc:
                if start <= token.idx < end:
                    token_to_entity[token.i].append(entity_text)
    
    # Process each verb in the document
    for token in doc:
        if token.pos_ != "VERB" and token.pos_ != "AUX":
            continue
        
        verb_lemma = _get_lemma(token)
        relation_type = RELATION_VERBS.get(verb_lemma)
        
        if not relation_type:
            continue
        
        # Find subject (nsubj) - source entity
        source_entities = []
        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                # Check if subject token is part of an entity
                for ent in token_to_entity.get(child.i, []):
                    source_entities.append(ent)
                # Also check head of subject
                if child.head == token:
                    for desc in child.subtree:
                        for ent in token_to_entity.get(desc.i, []):
                            source_entities.append(ent)
        
        # Find object (dobj, pobj) - target entity
        target_entities = []
        for child in token.children:
            if child.dep_ in ("dobj", "attr"):
                for ent in token_to_entity.get(child.i, []):
                    target_entities.append(ent)
                for desc in child.subtree:
                    for ent in token_to_entity.get(desc.i, []):
                        target_entities.append(ent)
            
            # Handle prepositional objects (e.g., "based in Paris")
            elif child.dep_ == "prep":
                for prep_child in child.children:
                    if prep_child.dep_ == "pobj":
                        for ent in token_to_entity.get(prep_child.i, []):
                            target_entities.append(ent)
        
        # Generate triples
        for src in set(source_entities):
            for tgt in set(target_entities):
                if src != tgt:  # No self-loops
                    triples.append({
                        "source": src,
                        "relation": relation_type,
                        "target": tgt,
                        "verb": verb_lemma,
                        "sentence": token.sent.text.strip(),
                    })
    
    return triples


def _find_entity_spans(text: str, entities: list[dict]) -> dict[str, list[tuple[int, int]]]:
    """
    Find character spans for each entity in text.
    
    Returns dict: entity_text -> [(start, end), ...]
    """
    spans: dict[str, list[tuple[int, int]]] = {}
    text_lower = text.lower()
    
    for ent in entities:
        entity_text = ent.get("entity", "") if isinstance(ent, dict) else str(ent)
        if not entity_text:
            continue
        
        # Find all occurrences
        entity_lower = entity_text.lower()
        start = 0
        while True:
            pos = text_lower.find(entity_lower, start)
            if pos == -1:
                break
            if entity_text not in spans:
                spans[entity_text] = []
            spans[entity_text].append((pos, pos + len(entity_text)))
            start = pos + 1
    
    return spans


def extract_relations(
    doc_data: dict[str, Any],
    entity_data: dict[str, Any],
    nlp,
) -> list[dict]:
    """
    Extract relations between entities using dependency parsing.
    
    Args:
        doc_data: Document data with 'full_text' or 'pages'
        entity_data: Entity extraction results with 'entities' list
        nlp: spaCy NLP model
    
    Returns:
        List of relation triples with confidence scores
    """
    # Get full text
    full_text = doc_data.get("full_text", "")
    if not full_text:
        pages = doc_data.get("pages", [])
        full_text = "\n\n".join(p.get("text", "") for p in pages)
    
    # Get entities
    entities = entity_data.get("entities", [])
    
    # Build entity span index
    entity_spans = _find_entity_spans(full_text, entities)
    
    if not entity_spans:
        return []
    
    # Parse document in chunks (handle long docs)
    all_triples = []
    max_len = nlp.max_length - 100
    
    if len(full_text) <= max_len:
        doc = nlp(full_text)
        triples = _extract_svo_triples(doc, entity_spans)
        all_triples.extend(triples)
    else:
        # Chunk by sentences to avoid cutting relations
        chunks = []
        current_chunk = ""
        for sent in nlp(full_text).sents:
            sent_text = sent.text.strip()
            if len(current_chunk) + len(sent_text) < max_len:
                current_chunk += " " + sent_text
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sent_text
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        for chunk in chunks:
            doc = nlp(chunk)
            triples = _extract_svo_triples(doc, entity_spans)
            all_triples.extend(triples)
    
    # Deduplicate and score triples
    seen = set()
    unique_triples = []
    
    for triple in all_triples:
        key = (triple["source"], triple["relation"], triple["target"])
        if key in seen:
            continue
        seen.add(key)
        
        # Calculate confidence
        confidence = _calculate_relation_confidence(triple, entities)
        triple["confidence"] = confidence
        
        unique_triples.append(triple)
    
    # Sort by confidence
    unique_triples.sort(key=lambda x: x["confidence"], reverse=True)
    
    return unique_triples


def _calculate_relation_confidence(triple: dict, entities: list[dict]) -> float:
    """
    Calculate confidence score for a relation triple.
    
    Factors:
    - Verb strength (strong verbs = higher confidence)
    - Entity frequency (frequent entities = more reliable)
    - Sentence length (not too short, not too long)
    """
    confidence = 0.7  # Base confidence
    
    # Strong verb bonus
    if triple.get("verb", "") in STRONG_RELATION_VERBS:
        confidence += 0.15
    
    # Relation type bonus
    strong_relations = {"developed", "created", "founded", "authored", "acquired"}
    if triple.get("relation", "") in strong_relations:
        confidence += 0.1
    
    # Entity frequency check
    entity_freq = {e.get("entity", ""): e.get("frequency", 1) for e in entities}
    src_freq = entity_freq.get(triple["source"], 1)
    tgt_freq = entity_freq.get(triple["target"], 1)
    
    if src_freq >= 2 and tgt_freq >= 2:
        confidence += 0.05
    
    # Sentence length penalty (too short = context missing)
    sent_len = len(triple.get("sentence", "").split())
    if sent_len < 5:
        confidence -= 0.1
    elif sent_len > 50:
        confidence -= 0.05
    
    return round(min(max(confidence, 0.0), 1.0), 4)


def prune_relations(
    relations: list[dict],
    min_confidence: float = 0.5,
    max_relations: int = 200,
) -> list[dict]:
    """
    Prune low-quality relations.
    
    Args:
        relations: List of relation triples
        min_confidence: Minimum confidence threshold
        max_relations: Maximum number of relations to keep
    
    Returns:
        Pruned list of relations
    """
    # Filter by confidence
    filtered = [r for r in relations if r.get("confidence", 0) >= min_confidence]
    
    # Remove isolated relations (entities appearing only once)
    entity_counts = defaultdict(int)
    for rel in filtered:
        entity_counts[rel["source"]] += 1
        entity_counts[rel["target"]] += 1
    
    pruned = [
        r for r in filtered
        if entity_counts[r["source"]] >= 1 and entity_counts[r["target"]] >= 1
    ]
    
    # Limit count
    if len(pruned) > max_relations:
        pruned = pruned[:max_relations]
    
    return pruned


def relations_to_graph_edges(relations: list[dict]) -> list[dict]:
    """
    Convert relation triples to graph edge format.
    
    Args:
        relations: List of relation triples
    
    Returns:
        List of edge dicts suitable for graph_builder
    """
    edges = []
    for rel in relations:
        edges.append({
            "source": rel["source"],
            "target": rel["target"],
            "relationship": rel["relation"],
            "weight": int(rel.get("confidence", 0.5) * 10),  # Scale to 1-10
            "confidence": rel.get("confidence", 0.5),
        })
    return edges


# ── Integration Points ────────────────────────────────────────────────────────
"""
INTEGRATION POINTS:

1. In entity_extractor.py after entity extraction:

   from backend.core.relation_extractor import extract_relations
   
   def extract_entities(doc_data):
       # Existing entity extraction...
       
       # Extract relations
       nlp = _get_nlp()
       relations = extract_relations(doc_data, result, nlp)
       result["relations"] = relations
       
       return result

2. In graph_builder.py replace co-occurrence with relations:

   from backend.core.relation_extractor import relations_to_graph_edges
   
   def build_knowledge_graph(doc_data, entity_data):
       # Use extracted relations instead of co-occurrence
       relations = entity_data.get("relations", [])
       edges = relations_to_graph_edges(relations)
       
       # Build graph from edges...

3. In analytics.py add relation analytics:

   def get_platform_analytics():
       # Collect relations from all documents
       # Show top relations by type
       # Show relation confidence distribution
"""
