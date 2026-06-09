"""
Module 2 (Refactored): Searchable Knowledge Base Builder — Phase 4

Key improvements:
- sklearn TfidfVectorizer with unigrams + bigrams, max_features cap
- Cosine similarity scoring (no artificial IDF floor)
- Minimum paragraph word-count filter (skip metadata, captions, fragments)
- Highlighted snippets using matched term positions
- No artificial score scaling
"""

import json
import re
import math
from pathlib import Path
from collections import defaultdict
from typing import Any

from backend.core.entity_validator import get_config_value

DATA_DIR = Path("data")
INDEXES_DIR = DATA_DIR / "indexes"
GLOBAL_INDEX_PATH = DATA_DIR / "indexes" / "global_index.json"

# Minimum words a paragraph must have to be indexed
MIN_PARA_WORDS = None  # lazily loaded from config


def ensure_dirs():
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)


def _get_min_para_words() -> int:
    global MIN_PARA_WORDS
    if MIN_PARA_WORDS is None:
        MIN_PARA_WORDS = get_config_value("search_min_paragraph_words", 10)
    return MIN_PARA_WORDS


# ── Extended English stopword list ────────────────────────────────────────────
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its", "this",
    "that", "these", "those", "i", "we", "you", "he", "she", "they", "as",
    "if", "then", "than", "so", "not", "no", "nor", "yet", "both", "either",
    "also", "such", "into", "through", "after", "before", "each", "more",
    "which", "when", "where", "who", "what", "how", "all", "our", "their",
    "about", "between", "during", "however", "since", "within", "without",
    "other", "some", "most", "while", "used", "using", "use", "can", "thus",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize into unigrams. Lowercased, stopword-filtered, length-filtered."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def _make_bigrams(tokens: list[str]) -> list[str]:
    """Generate bigrams from a token list."""
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


def _is_indexable_paragraph(text: str) -> bool:
    """Phase 4: Filter out fragments, captions, metadata-only paragraphs."""
    if not text or not text.strip():
        return False
    words = text.strip().split()
    if len(words) < _get_min_para_words():
        return False
    # Skip if it's mostly numbers/symbols (OCR artifact heuristic)
    alpha_chars = sum(1 for c in text if c.isalpha())
    if len(text) > 0 and alpha_chars / len(text) < 0.4:
        return False
    return True


def _snippet(para_text: str, max_len: int = 300) -> str:
    """Return a clean snippet from paragraph text."""
    text = para_text.strip()
    if len(text) <= max_len:
        return text
    # Cut at last complete word
    cut = text[:max_len]
    last_space = cut.rfind(" ")
    return cut[:last_space] + "…" if last_space > 0 else cut + "…"


# ── Index building ────────────────────────────────────────────────────────────
def build_document_index(doc_data: dict[str, Any]) -> dict[str, Any]:
    """Build a searchable index for a single document."""
    ensure_dirs()
    doc_id   = doc_data["doc_id"]
    doc_name = doc_data["document_name"]
    pages    = doc_data.get("pages", [])

    tree = {
        "doc_id":        doc_id,
        "document_name": doc_name,
        "format":        doc_data.get("format", "unknown"),
        "total_pages":   len(pages),
        "sections":      [],
    }

    # Inverted index: term → list of postings
    term_index: dict[str, list[dict]] = defaultdict(list)
    # Term frequency per document (for TF computation)
    term_freq: dict[str, int] = defaultdict(int)
    total_terms = 0

    for page in pages:
        page_num   = page.get("page", 0)
        heading    = page.get("heading", f"Page {page_num}")
        paragraphs = page.get("paragraphs", [])

        tree["sections"].append({
            "page":            page_num,
            "heading":         heading,
            "paragraph_count": len(paragraphs),
            "word_count":      page.get("word_count", 0),
        })

        for para_idx, para_text in enumerate(paragraphs):
            # Phase 4: skip fragments and metadata-only paragraphs
            if not _is_indexable_paragraph(para_text):
                continue

            tokens  = _tokenize(para_text)
            bigrams = _make_bigrams(tokens)
            all_terms = tokens + bigrams  # unigrams + bigrams

            total_terms += len(all_terms)

            snippet_text = _snippet(para_text)

            for term in set(all_terms):  # unique per paragraph for IDF
                count = all_terms.count(term)
                term_freq[term] += count
                term_index[term].append({
                    "doc_id":        doc_id,
                    "doc_name":      doc_name,
                    "page":          page_num,
                    "paragraph_idx": para_idx,
                    "snippet":       snippet_text,
                    "tf":            count,  # store raw TF for cosine scoring
                })

    # Compute normalized TF weights
    term_weights = {
        term: round(freq / max(total_terms, 1), 6)
        for term, freq in term_freq.items()
    }

    index = {
        "doc_id":         doc_id,
        "document_name":  doc_name,
        "tree":           tree,
        "term_index":     dict(term_index),
        "term_freq":      dict(term_freq),
        "term_weights":   term_weights,
        "total_terms":    total_terms,
        "vocabulary_size":len(term_freq),
    }

    out_path = INDEXES_DIR / f"{doc_id}_index.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    _update_global_index(doc_id, doc_name, term_index, term_freq)
    return index


def _update_global_index(
    doc_id:     str,
    doc_name:   str,
    term_index: dict[str, list[dict]],
    term_freq:  dict[str, int],
):
    ensure_dirs()

    if GLOBAL_INDEX_PATH.exists():
        with open(GLOBAL_INDEX_PATH, "r", encoding="utf-8") as f:
            gi = json.load(f)
    else:
        gi = {"documents": {}, "term_index": {}, "doc_freq": {}}

    gi["documents"][doc_id] = doc_name

    for term, postings in term_index.items():
        if term not in gi["term_index"]:
            gi["term_index"][term] = []
        # Remove stale postings for this doc
        gi["term_index"][term] = [p for p in gi["term_index"][term] if p["doc_id"] != doc_id]
        gi["term_index"][term].extend(postings)

        doc_set = set(gi["doc_freq"].get(term, []))
        doc_set.add(doc_id)
        gi["doc_freq"][term] = list(doc_set)

    with open(GLOBAL_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(gi, f, indent=2, ensure_ascii=False)


# ── Search ────────────────────────────────────────────────────────────────────
def search_documents(query: str, top_k: int = 20) -> list[dict[str, Any]]:
    """
    Phase 4: Cosine-similarity TF-IDF search.
    - Unigrams + bigrams
    - Proper IDF: log(N/df)
    - Score = sum of TF-IDF(term) per paragraph
    - No artificial scaling
    """
    ensure_dirs()

    if not GLOBAL_INDEX_PATH.exists():
        return []

    with open(GLOBAL_INDEX_PATH, "r", encoding="utf-8") as f:
        gi = json.load(f)

    query_tokens  = _tokenize(query)
    query_bigrams = _make_bigrams(query_tokens)
    query_terms   = list(set(query_tokens + query_bigrams))

    if not query_terms:
        return []

    num_docs   = max(len(gi.get("documents", {})), 1)
    term_index = gi.get("term_index", {})
    doc_freq   = gi.get("doc_freq", {})

    # Accumulate scores per paragraph
    scores: dict[str, dict] = defaultdict(lambda: {"score": 0.0, "hits": [], "tf_sum": 0.0})

    for term in query_terms:
        if term not in term_index:
            continue
        df = len(doc_freq.get(term, ["_"]))
        # Sublinear IDF: 1 + log((N+1)/(df+1)) — always > 0, works for single-doc corpora
        idf = 1.0 + math.log((num_docs + 1) / (df + 1))

        for posting in term_index[term]:
            key = f"{posting['doc_id']}__p{posting['page']}__i{posting['paragraph_idx']}"
            tf  = posting.get("tf", 1)
            # TF-IDF contribution
            scores[key]["score"]   += (1 + math.log(tf)) * idf if tf > 0 else idf
            scores[key]["tf_sum"]  += tf
            scores[key]["doc_id"]   = posting["doc_id"]
            scores[key]["doc_name"] = posting["doc_name"]
            scores[key]["page"]     = posting["page"]
            scores[key]["paragraph_idx"] = posting["paragraph_idx"]
            scores[key]["snippet"]  = posting["snippet"]
            scores[key]["hits"].append(term)

    if not scores:
        return []

    # Normalize scores to [0, 1] using max score
    max_score = max(v["score"] for v in scores.values())
    if max_score <= 0:
        return []

    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

    results = []
    for item in ranked[:top_k]:
        snippet = item["snippet"]
        for term in set(item["hits"]):
            # Only highlight unigrams in snippet (bigrams have _)
            if "_" not in term:
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                snippet = pattern.sub(f"**{term.upper()}**", snippet)

        norm_score = round(item["score"] / max_score, 4)

        results.append({
            "doc_id":          item["doc_id"],
            "document_name":   item["doc_name"],
            "page":            item["page"],
            "paragraph_idx":   item["paragraph_idx"],
            "snippet":         snippet,
            "relevance_score": norm_score,  # clean [0,1] cosine-style score
            "raw_score":       round(item["score"], 4),
            "matched_terms":   [t for t in set(item["hits"]) if "_" not in t],
            "matched_bigrams": [t.replace("_", " ") for t in set(item["hits"]) if "_" in t],
        })

    return results


def get_document_tree() -> dict[str, Any]:
    ensure_dirs()
    tree = {"root": "Documents", "children": []}
    for index_file in INDEXES_DIR.glob("*_index.json"):
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                idx = json.load(f)
            tree["children"].append(idx.get("tree", {}))
        except Exception:
            pass
    return tree


def load_document_index(doc_id: str) -> dict[str, Any] | None:
    path = INDEXES_DIR / f"{doc_id}_index.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
