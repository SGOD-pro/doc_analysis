# Document Analytics Engine - Redesign Implementation Plan

## Architecture Review

### Current State
- **Entity Extraction**: spaCy NER with validation layer (good foundation)
- **Topic Discovery**: LDA/NMF/TF-IDF fallback (correct approach, poor tuning)
- **Knowledge Graph**: Sentence co-occurrence with pruning (needs relation extraction upgrade)
- **Search**: Custom TF-IDF (needs BM25 replacement)
- **Analytics**: Aggregator with quality filters (needs enhancement)

### Target Improvements (All Phases)

---

## PHASE 1: Entity Quality

### Files to Modify
1. `backend/core/entity_validator.py` - Enhanced validation rules
2. `backend/core/entity_config.json` - Domain dictionaries
3. `backend/core/entity_extractor.py` - Confidence scoring integration

### Changes Required

#### 1.1 entity_config.json - Add Domain Dictionaries
Add sections for:
- MODEL names (extended list)
- DATASET names
- METRIC names
- LEGAL_TERM examples
- MEDICAL_TERM examples
- FINANCIAL_TERM examples

#### 1.2 entity_validator.py - Enhanced Validation
Functions to add/modify:
- `is_valid_person()` - Reject "dk", "zn", single lowercase chars
- `is_valid_org()` - Reject common false positives like "Adam", "Linear"
- `is_valid_location()` - Already exists, enhance with more exclusions
- `calculate_confidence()` - Return confidence based on validation strength

#### 1.3 entity_extractor.py - Confidence Integration
- Update entity output to include confidence scores
- Propagate confidence to graph builder

---

## PHASE 2: Domain Detection

### New File: `backend/core/domain_detector.py`

### Supported Domains
- RESEARCH (ML/AI papers)
- LEGAL (contracts, regulations)
- MEDICAL (clinical, pharma)
- FINANCE (reports, filings)
- TECH (documentation, specs)
- GENERAL (fallback)

### Implementation
- Keyword scoring per domain
- Return `{domain, confidence}`
- Integrate with entity_extractor, topic_extractor, analytics

---

## PHASE 3: Knowledge Graph (Relation Extraction)

### Files to Modify
1. `backend/core/graph_builder.py` - Replace co-occurrence with dependency parsing

### Changes Required

#### 3.1 Add Dependency-Based Relation Extraction
Functions to add:
- `_extract_semantic_triples()` - Use spaCy dependency parse
- `_find_subject_verb_object()` - Parse sentence structure
- `_prune_low_confidence_edges()` - Edge filtering
- `_remove_isolated_nodes()` - Already exists, keep

#### 3.2 Triple Format
```json
{
  "source": "Google",
  "relation": "developed",
  "target": "Transformer",
  "confidence": 0.85
}
```

#### 3.3 Keep NetworkX
- No Neo4j introduction
- Preserve existing metrics (degree, betweenness, pagerank)

---

## PHASE 4: Topic Discovery

### Files to Modify
1. `backend/core/topic_extractor.py` - Replace LDA/NMF with TF-IDF + KMeans

### Changes Required

#### 4.1 Replace Topic Pipeline
Current: LDA → NMF → TF-IDF fallback
New: TF-IDF → KMeans clustering

Functions to modify:
- Remove `_run_lda()`, `_run_nmf()`
- Add `_run_kmeans_clustering()`
- Add `_determine_optimal_clusters()` - Elbow method or fixed heuristic
- Add `_extract_cluster_keywords()` - Top terms per cluster
- Add `_generate_deterministic_label()` - Rule-based naming

#### 4.2 Requirements
- Deterministic (same input = same output)
- Remove duplicate topics (Jaccard dedup already exists)
- Improve representative keyword extraction

---

## PHASE 5: Search (BM25)

### Files to Modify
1. `backend/core/index_builder.py` - Replace TF-IDF with BM25

### Changes Required

#### 5.1 BM25 Implementation
- Add `rank_bm25` library OR implement pure-Python BM25
- Functions to modify:
  - `search_documents()` - Replace cosine scoring with BM25
  - Keep index structure (preserve dashboard integration)

#### 5.2 Pure-Python BM25 (no extra dependency)
Implement:
```python
def bm25_score(term_freq, doc_len, avg_doc_len, idf, k1=1.5, b=0.75):
    ...
```

---

## PHASE 6: Analytics Enhancement

### Files to Modify
1. `backend/core/analytics.py` - Add new metrics

### Metrics to Add
- Domain confidence distribution
- Relation counts by type
- Graph density (already exists)
- Graph quality score (new)
- Average entity confidence (new)

---

## Testing Strategy

### Unit Tests
1. Entity validation edge cases
2. Domain detection accuracy
3. Triple extraction correctness
4. Topic clustering stability
5. BM25 ranking relevance

### Integration Tests
1. Full pipeline: PDF → Entities → Graph → Topics → Search
2. Cross-document aggregation
3. Dashboard data consistency

---

## Expected Improvements

| Metric | Before | After |
|--------|--------|-------|
| Wrong PERSON entities | ~15% | <3% |
| Wrong ORG entities | ~20% | <5% |
| Topic coherence | 0.3-0.4 | 0.6-0.7 |
| Graph precision | 0.4 | 0.75 |
| Search MRR | 0.5 | 0.75 |

---

## Implementation Order

1. **Phase 1** - Entity validation (foundation for all downstream)
2. **Phase 2** - Domain detector (needed for entity reclassification)
3. **Phase 3** - Knowledge graph relations (depends on entities)
4. **Phase 4** - Topic clustering (independent)
5. **Phase 5** - BM25 search (independent)
6. **Phase 6** - Analytics enhancement (aggregates all improvements)
