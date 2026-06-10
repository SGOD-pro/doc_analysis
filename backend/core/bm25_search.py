"""
PHASE 5: BM25 Search Implementation

Replaces TF-IDF with BM25 (Best Matching 25) for better search relevance.
BM25 is a probabilistic ranking function used in production search engines.

Key improvements over TF-IDF:
- Better handling of document length normalization
- Term frequency saturation (diminishing returns for repeated terms)
- Tunable parameters (k1, b) for different corpora

Uses only scikit-learn and standard library - no external dependencies.
"""

import math
import json
from pathlib import Path
from collections import defaultdict
from typing import Any


DATA_DIR = Path("data")
INDEXES_DIR = DATA_DIR / "indexes"
GLOBAL_INDEX_PATH = INDEXES_DIR / "global_index.json"

# BM25 parameters (tunable)
# k1: term frequency saturation (1.2-2.0 typical)
# b: length normalization (0.75 typical)
BM25_K1 = 1.5
BM25_B = 0.75


class BM25Index:
    """
    BM25 search index using inverted index structure.
    
    Stores:
    - Document frequencies (df) for IDF calculation
    - Term frequencies per document (tf)
    - Document lengths for normalization
    - Average document length
    """
    
    def __init__(self):
        self.doc_freq: dict[str, int] = {}  # term -> number of docs containing term
        self.term_docs: dict[str, dict[str, int]] = {}  # term -> {doc_id -> tf}
        self.doc_lengths: dict[str, int] = {}  # doc_id -> word count
        self.num_docs: int = 0
        self.avg_doc_length: float = 0.0
    
    def add_document(self, doc_id: str, tokens: list[str]):
        """Add a document to the index."""
        doc_len = len(tokens)
        self.doc_lengths[doc_id] = doc_len
        
        # Count term frequencies in this document
        tf: dict[str, int] = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        
        # Update document frequency and term-docs mapping
        for term, count in tf.items():
            if term not in self.doc_freq:
                self.doc_freq[term] = 0
                self.term_docs[term] = {}
            
            self.doc_freq[term] += 1
            self.term_docs[term][doc_id] = count
        
        self.num_docs += 1
        self._update_avg_doc_length()
    
    def _update_avg_doc_length(self):
        """Recalculate average document length."""
        if self.doc_lengths:
            total = sum(self.doc_lengths.values())
            self.avg_doc_length = total / len(self.doc_lengths)
    
    def _bm25_score(
        self,
        tf: int,
        df: int,
        doc_len: int,
        query_term_freq: int = 1,
    ) -> float:
        """
        Calculate BM25 score for a single term in a document.
        
        Formula:
        score = IDF(qi) * (f(d, qi) * (k1 + 1)) / (f(d, qi) + k1 * (1 - b + b * |d|/avgdl))
        
        Where:
        - f(d, qi) = term frequency in document
        - |d| = document length
        - avgdl = average document length
        - IDF(qi) = log((N - df + 0.5) / (df + 0.5) + 1)
        """
        # IDF component
        idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)
        
        # TF component with saturation
        tf_norm = tf * (BM25_K1 + 1)
        tf_denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / self.avg_doc_length)
        
        if tf_denom == 0:
            return 0.0
        
        tf_component = tf_norm / tf_denom
        
        return idf * tf_component * query_term_freq
    
    def search(self, query_tokens: list[str], top_k: int = 20) -> list[tuple[str, float]]:
        """
        Search for documents matching query tokens.
        
        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        if not query_tokens or self.num_docs == 0:
            return []
        
        # Accumulate scores per document
        scores: dict[str, float] = defaultdict(float)
        
        for term in set(query_tokens):
            if term not in self.term_docs:
                continue
            
            df = self.doc_freq.get(term, 0)
            
            for doc_id, tf in self.term_docs[term].items():
                doc_len = self.doc_lengths.get(doc_id, int(self.avg_doc_length))
                score = self._bm25_score(tf, df, doc_len)
                scores[doc_id] += score
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]
    
    def save(self, path: Path):
        """Save index to JSON file."""
        data = {
            "doc_freq": self.doc_freq,
            "term_docs": self.term_docs,
            "doc_lengths": self.doc_lengths,
            "num_docs": self.num_docs,
            "avg_doc_length": self.avg_doc_length,
            "params": {"k1": BM25_K1, "b": BM25_B},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Load index from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        index = cls()
        index.doc_freq = data["doc_freq"]
        index.term_docs = data["term_docs"]
        index.doc_lengths = data["doc_lengths"]
        index.num_docs = data["num_docs"]
        index.avg_doc_length = data["avg_doc_length"]
        
        return index


def _tokenize_for_search(text: str) -> list[str]:
    """Tokenize text for BM25 indexing/search."""
    import re
    
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    tokens = text.split()
    
    # Filter stopwords and short tokens
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "it", "its", "this", "that", "these", "those",
        "i", "we", "you", "he", "she", "they", "as", "if", "then", "than",
        "so", "not", "no", "yet", "both", "either", "also", "such",
    }
    
    return [t for t in tokens if len(t) > 2 and t not in stopwords]


def build_bm25_index(doc_data_list: list[dict[str, Any]]) -> BM25Index:
    """
    Build BM25 index from processed documents.
    
    Args:
        doc_data_list: List of document data dicts with 'full_text' or 'pages'
    
    Returns:
        BM25Index object ready for searching
    """
    index = BM25Index()
    
    for doc_data in doc_data_list:
        doc_id = doc_data.get("doc_id", "")
        if not doc_id:
            continue
        
        # Get full text
        full_text = doc_data.get("full_text", "")
        if not full_text:
            pages = doc_data.get("pages", [])
            full_text = "\n\n".join(p.get("text", "") for p in pages)
        
        # Tokenize and index
        tokens = _tokenize_for_search(full_text)
        index.add_document(doc_id, tokens)
    
    return index


def search_bm25(
    query: str,
    index: BM25Index,
    doc_lookup: dict[str, dict[str, Any]],
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """
    Search using BM25 and return formatted results.
    
    Args:
        query: Search query string
        index: BM25Index object
        doc_lookup: Dict mapping doc_id to document metadata
        top_k: Number of results to return
    
    Returns:
        List of result dicts with doc info and snippets
    """
    query_tokens = _tokenize_for_search(query)
    
    if not query_tokens:
        return []
    
    # Search
    results = index.search(query_tokens, top_k=top_k * 2)  # Get extra for filtering
    
    # Format results
    formatted = []
    for doc_id, score in results:
        doc_info = doc_lookup.get(doc_id, {})
        
        # Generate snippet with highlighted terms
        full_text = doc_info.get("full_text", "")
        snippet = _generate_snippet(full_text, query_tokens)
        
        formatted.append({
            "doc_id": doc_id,
            "document_name": doc_info.get("document_name", doc_id),
            "snippet": snippet,
            "score": round(score, 4),
            "matched_terms": list(set(query_tokens)),
        })
    
    return formatted[:top_k]


def _generate_snippet(text: str, query_tokens: list[str], max_len: int = 300) -> str:
    """Generate a snippet with highlighted query terms."""
    if not text:
        return ""
    
    text_lower = text.lower()
    
    # Find best window containing most query terms
    best_start = 0
    best_count = 0
    
    window_size = 200
    step = 50
    
    for start in range(0, max(len(text) - window_size, 0), step):
        window = text_lower[start:start + window_size]
        count = sum(1 for t in query_tokens if t in window)
        if count > best_count:
            best_count = count
            best_start = start
    
    # Extract and highlight
    snippet_start = max(0, best_start - 50)
    snippet_end = min(len(text), best_start + window_size + 50)
    
    snippet = text[snippet_start:snippet_end]
    
    # Add ellipsis if truncated
    if snippet_start > 0:
        snippet = "..." + snippet
    if snippet_end < len(text):
        snippet = snippet + "..."
    
    # Highlight matched terms
    for term in query_tokens:
        snippet = snippet.replace(term, f"**{term.upper()}**")
    
    return snippet.strip()


# ── Integration with existing index_builder.py ────────────────────────────────
"""
INTEGRATION POINTS:

1. Replace search_documents() in index_builder.py:

   from backend.core.bm25_search import BM25Index, build_bm25_index, search_bm25
   
   # Build index once (e.g., after processing all docs)
   def build_global_bm25_index():
       docs = list_processed_documents()
       index = build_bm25_index(docs)
       index.save(INDEXES_DIR / "bm25_index.json")
   
   # Search using BM25
   def search_documents(query: str, top_k: int = 20):
       index = BM25Index.load(INDEXES_DIR / "bm25_index.json")
       
       # Load doc lookup
       doc_lookup = {}
       for doc in list_processed_documents():
           doc_lookup[doc["doc_id"]] = doc
       
       return search_bm25(query, index, doc_lookup, top_k)

2. Keep existing TF-IDF index for backward compatibility:
   - BM25 for primary search
   - TF-IDF for keyword extraction/analytics

3. Migration strategy:
   - Build BM25 index alongside existing TF-IDF
   - Run both in parallel, compare results
   - Switch to BM25-only after validation
"""
