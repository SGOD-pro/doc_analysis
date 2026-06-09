"""
Module 4 (Production): Topic & Trend Extractor

Priority chain (all pure CPU, zero model downloads, domain-agnostic):
  1. LDA  — best quality for 10+ paragraphs
  2. NMF  — better for sparse/short docs
  3. TF-IDF keyword cluster — absolute last resort

BERTopic REMOVED: torch + sentence-transformers + hdbscan + umap-learn ≈ 3 GB.
LDA + NMF from scikit-learn is correct for lightweight local-first systems.

Phase improvements applied:
  Phase 3 — Domain-aware tokenizer (preserves GPT-4, HbA1c, MRI, EBITDA, etc.)
  Phase 4 — Deterministic topic naming without LLMs (template + uniqueness ranking)
  Phase 5 — Separated GENERAL_STOPS / DOMAIN_SENSITIVE_STOPS
  Phase 6 — Jaccard similarity deduplication (not raw overlap count)
  Phase 7 — Coherence is normalized [0–1], topics carry coherence + topic_weight
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Any

DATA_DIR  = Path("data")
TOPICS_DIR = DATA_DIR / "topics"


# ── Phase 5: Two-tier stopword lists ──────────────────────────────────────────
# Words that are noise in EVERY domain — safe to remove always.
GENERAL_STOPS: frozenset[str] = frozenset({
    "used", "using", "also", "more", "into", "each", "than", "other",
    "would", "could", "should", "however", "therefore", "thus", "hence",
    "show", "shows", "shown", "figure", "table", "section", "page",
    "paper", "work", "based", "approach", "proposed", "previous",
    "following", "given", "first", "second", "third",
    "number", "different", "large", "high", "low", "same",
    "allows", "apply", "applied", "include", "including", "requires",
    "note", "noted", "use", "may", "new", "well", "two", "three", "one",
    "due", "via", "per", "without", "within", "across", "along",
    "various", "certain", "specific", "general", "overall", "further",
    "type", "types", "level", "levels", "case", "cases",
})

# Words that carry domain meaning — only suppress when safe to do so.
# NOT added to vectorizer stops; used only for keyword-cloud filtering.
DOMAIN_SENSITIVE_STOPS: frozenset[str] = frozenset({
    "model", "models", "data", "dataset", "performance",
    "result", "results", "method", "methods",
    "system", "systems", "analysis", "process",
})

# Combined for vectorizer (LDA/NMF/TF-IDF)
_VECTORIZER_STOPS: frozenset[str] | None = None  # lazy-cached


def _get_stop_words() -> list[str]:
    """sklearn's 318 English stops + GENERAL_STOPS, cached after first call."""
    global _VECTORIZER_STOPS
    if _VECTORIZER_STOPS is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        _VECTORIZER_STOPS = ENGLISH_STOP_WORDS.union(GENERAL_STOPS)
    return list(_VECTORIZER_STOPS)  # type: ignore[arg-type]


# Phase 3: Domain-aware tokenizer
# Three dedicated patterns, applied in priority order:

# 1. Alphanumeric compound tokens: HbA1c, COVID-19, GPT-4, L1, P/E
_COMPOUND_RE = re.compile(
    r'[A-Za-z]{1,8}[\-/][A-Za-z0-9]{1,8}'  # hyphen/slash: GPT-4, P/E
    r'|[A-Za-z]+[0-9]+[A-Za-z][0-9A-Za-z]*'  # letter+digit+letter: HbA1c, COVID19
)

# 2. ALL-CAPS acronyms ≥ 2 chars: MRI, BLEU, CNN, SVM, JWT
_ACRONYM_RE = re.compile(r'\b[A-Z]{2,}\b')

# 3. Legal: Article 12, Section 15, Clause 4
_LEGAL_PHRASE_RE = re.compile(
    r'\b(Article|Section|Clause|Chapter|Schedule|Appendix)\s+\d+\b',
    re.IGNORECASE,
)

# 4. Normal words ≥ 3 chars (after compounds + acronyms already extracted)
_WORD_RE = re.compile(r'\b[a-zA-Z]{3,}\b')


def _tokenize_document(text: str) -> list[str]:
    """
    Domain-aware tokenizer. Applies dedicated patterns per token type.
    Order: legal phrases → compounds → acronyms → normal words.
    Avoids double-extraction by tracking consumed character positions.
    """
    tokens: list[str] = []
    consumed: set[int] = set()

    def add(m: re.Match, tok: str) -> None:
        positions = set(range(m.start(), m.end()))
        if not positions & consumed:
            tokens.append(tok.lower() if not any(c.isupper() for c in tok[1:]) else tok.lower())
            consumed.update(positions)

    # Legal phrases first (may contain uppercase words)
    for m in _LEGAL_PHRASE_RE.finditer(text):
        add(m, m.group())

    # Compound tokens: HbA1c, GPT-4, P/E
    for m in _COMPOUND_RE.finditer(text):
        positions = set(range(m.start(), m.end()))
        if not positions & consumed:
            tokens.append(m.group().lower())
            consumed.update(positions)

    # ALL-CAPS acronyms
    for m in _ACRONYM_RE.finditer(text):
        positions = set(range(m.start(), m.end()))
        if not positions & consumed:
            tokens.append(m.group().lower())
            consumed.update(positions)

    # Normal words
    for m in _WORD_RE.finditer(text):
        positions = set(range(m.start(), m.end()))
        if not positions & consumed and len(m.group()) >= 3:
            tokens.append(m.group().lower())
            consumed.update(positions)

    return tokens


def _tokenize_for_vectorizer(text: str) -> str:
    """
    Convert text to a normalized string that sklearn CountVectorizer/TfidfVectorizer
    can consume while benefiting from our domain-aware tokenization.
    """
    tokens = _tokenize_document(text)
    return " ".join(tokens)


# ── Utilities ──────────────────────────────────────────────────────────────────
def ensure_dirs() -> None:
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)


def _is_meaningful_paragraph(text: str, min_words: int = 15) -> bool:
    """
    Reject table/caption/metadata noise.
    15-word floor (was 20 — too aggressive, excluded real PDF paragraphs).
    50% alpha-char ratio rejects formula/table rows.
    """
    if not text or not text.strip():
        return False
    words = text.strip().split()
    if len(words) < min_words:
        return False
    alpha = sum(1 for c in text if c.isalpha())
    return (alpha / max(len(text), 1)) >= 0.5


def _adaptive_min_df(n_docs: int) -> int:
    """
    Adapts min_df to prevent vocabulary collapse on small corpora.
    Hardcoding min_df=2 kills LDA on single-document processing.
    """
    if n_docs < 10:
        return 1
    elif n_docs < 30:
        return 2
    return 3


# ── Phase 4: Deterministic topic naming ───────────────────────────────────────
# Curated bigram → readable label mapping (domain-agnostic, extendable)
_BIGRAM_LABELS: dict[str, str] = {
    # ML / NLP
    "attention transformer": "Transformer Architecture",
    "transformer attention": "Transformer Architecture",
    "neural network": "Neural Networks",
    "deep learning": "Deep Learning",
    "machine learning": "Machine Learning",
    "natural language": "Natural Language Processing",
    "language model": "Language Modeling",
    "encoder decoder": "Encoder-Decoder Architecture",
    "recurrent neural": "Recurrent Networks",
    "self attention": "Self-Attention",
    "transfer learning": "Transfer Learning",
    "image classification": "Image Classification",
    "object detection": "Object Detection",
    "gradient descent": "Optimization",
    # Medical
    "blood glucose": "Diabetes Management",
    "heart rate": "Cardiovascular Health",
    "clinical trial": "Clinical Research",
    "patient outcome": "Patient Outcomes",
    "drug treatment": "Drug Treatment",
    "adverse event": "Adverse Events",
    "randomized controlled": "Clinical Trials",
    # Finance
    "credit risk": "Credit Risk",
    "market risk": "Market Risk",
    "interest rate": "Interest Rate",
    "cash flow": "Cash Flow",
    "revenue growth": "Revenue Growth",
    "earnings report": "Earnings Report",
    "risk management": "Risk Management",
    "portfolio management": "Portfolio Management",
    # Legal
    "intellectual property": "Intellectual Property",
    "breach contract": "Contract Breach",
    "due diligence": "Due Diligence",
    "governing law": "Governing Law",
    "indemnification clause": "Indemnification",
    "force majeure": "Force Majeure",
    # Technical
    "software development": "Software Development",
    "system architecture": "System Architecture",
    "api endpoint": "API Design",
    "cloud computing": "Cloud Computing",
    "data pipeline": "Data Pipeline",
    "database schema": "Database Design",
    "security vulnerability": "Security",
    "microservice architecture": "Microservices",
}


def _name_topic(keywords: list[str]) -> str:
    """
    Deterministic, LLM-free topic naming.

    Strategy (in order):
    1. Check top-3 keywords as bigrams against curated domain label map.
    2. Find the 2 most unique (rarest / most distinctive) keywords.
    3. Join them with a space — not '&' or '/' which reads as a list.

    Result: "Encoder Decoder" not "Encoder / Decoder / Layers".
    """
    if not keywords:
        return "General"

    # Step 1: curated bigram lookup (top 3 keywords)
    top3 = [k.lower().strip() for k in keywords[:3]]
    for i in range(len(top3)):
        for j in range(i + 1, len(top3)):
            bigram = f"{top3[i]} {top3[j]}"
            rev_bigram = f"{top3[j]} {top3[i]}"
            if bigram in _BIGRAM_LABELS:
                return _BIGRAM_LABELS[bigram]
            if rev_bigram in _BIGRAM_LABELS:
                return _BIGRAM_LABELS[rev_bigram]

    # Step 2: prefer bigrams (2-word phrases) over unigrams — more informative
    bigrams = [k for k in keywords[:6] if " " in k]
    unigrams = [k for k in keywords[:6] if " " not in k]

    chosen: list[str] = []
    seen_roots: set[str] = set()

    # Prefer bigrams first, then unigrams
    for kw in (bigrams + unigrams):
        root = kw.split()[0].lower()
        if root not in seen_roots and kw.lower() not in seen_roots:
            # Title-case: "attention mechanism" → "Attention Mechanism"
            chosen.append(" ".join(w.capitalize() for w in kw.split()))
            seen_roots.add(root)
            for part in kw.lower().split():
                seen_roots.add(part)
        if len(chosen) == 2:
            break

    if not chosen:
        return keywords[0].title()

    return " ".join(chosen)  # e.g. "Attention Mechanism Encoder Decoder"


# ── Phase 7: Coherence scoring ────────────────────────────────────────────────
def _compute_coherence(component, top_indices) -> float:
    """
    Normalized topic coherence: proportion of total topic mass concentrated
    in the top-N terms.

    Range: [0.0, 1.0] — higher = more focused topic.
    Was always 0.0 in previous version (bug: used full mean not top-term sum).
    """
    total = component.sum()
    if total == 0.0:
        return 0.0
    top_mass = float(component[top_indices].sum())
    return round(top_mass / total, 4)


def _topic_weight(coherence: float, document_count: int, total_docs: int) -> float:
    """
    Blended topic importance score for ranking and pie-chart sizing.
    Combines coherence (quality) with document coverage (breadth).
    Range: [0.0, 1.0]
    """
    if total_docs == 0:
        return coherence
    coverage = document_count / total_docs
    # 60% coherence quality, 40% document coverage
    return round(0.6 * coherence + 0.4 * coverage, 4)


# ── Phase 6: Jaccard deduplication ────────────────────────────────────────────
def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity coefficient between two keyword sets."""
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _deduplicate_topics(topics: list[dict]) -> list[dict]:
    """
    Remove semantically redundant topics using Jaccard similarity.
    Threshold: 0.5 (50% Jaccard overlap = duplicate).

    Improves on raw overlap count which was biased toward small keyword sets.
    Example removed pair:
      A: ["attention", "transformer", "encoder"]
      B: ["transformer", "attention", "decoder"]
      Jaccard = 2/4 = 0.5 → duplicate, B removed.
    """
    seen_kws: list[set[str]] = []
    unique: list[dict] = []

    for topic in topics:
        kws = set(k.lower() for k in topic.get("keywords", [])[:8])
        if not any(_jaccard(kws, s) >= 0.5 for s in seen_kws):
            unique.append(topic)
            seen_kws.append(kws)

    return unique


# ── LDA ───────────────────────────────────────────────────────────────────────
def _run_lda(documents: list[str], n_topics: int) -> list[dict] | None:
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        # Pre-tokenize with domain-aware tokenizer
        tokenized = [_tokenize_for_vectorizer(d) for d in documents]

        min_df = _adaptive_min_df(len(tokenized))
        stop_words = _get_stop_words()

        vect = CountVectorizer(
            stop_words=stop_words,
            min_df=min_df,
            max_df=0.95,
            ngram_range=(1, 2),
            max_features=3000,
            # Simple whitespace split — our tokenizer already handled complexity
            token_pattern=r'\S+',
        )
        dtm = vect.fit_transform(tokenized)

        if dtm.shape[0] < 3 or dtm.shape[1] < 8:
            return None

        n = min(n_topics, max(2, dtm.shape[0] // 3))
        lda = LatentDirichletAllocation(
            n_components=n,
            max_iter=30,
            learning_method="online",
            learning_offset=50.0,
            random_state=42,
            n_jobs=-1,
        )
        lda.fit(dtm)

        feature_names = vect.get_feature_names_out()
        total_docs = dtm.shape[0]
        results: list[dict] = []

        for tid, component in enumerate(lda.components_):
            top_indices = component.argsort()[:-11:-1]
            keywords = [feature_names[i] for i in top_indices]
            doc_count = int((dtm[:, top_indices].sum(axis=1) > 0).sum())
            coherence = _compute_coherence(component, top_indices)
            results.append({
                "topic_id":      tid,
                "topic":         _name_topic(keywords),
                "keywords":      keywords,
                "coherence":     coherence,
                "topic_weight":  _topic_weight(coherence, doc_count, total_docs),
                "document_count": doc_count,
                "method":        "LDA",
            })

        results.sort(key=lambda x: x["topic_weight"], reverse=True)
        return _deduplicate_topics(results) or None

    except Exception as e:
        print(f"[Topics] LDA failed: {e}")
        return None


# ── NMF ───────────────────────────────────────────────────────────────────────
def _run_nmf(documents: list[str], n_topics: int) -> list[dict] | None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF

        tokenized = [_tokenize_for_vectorizer(d) for d in documents]
        min_df = _adaptive_min_df(len(tokenized))
        stop_words = _get_stop_words()

        vect = TfidfVectorizer(
            stop_words=stop_words,
            min_df=min_df,
            max_df=0.95,
            ngram_range=(1, 2),
            max_features=3000,
            token_pattern=r'\S+',
        )
        dtm = vect.fit_transform(tokenized)

        if dtm.shape[0] < 2 or dtm.shape[1] < 5:
            return None

        n = min(n_topics, max(2, dtm.shape[0] // 2))
        nmf = NMF(n_components=n, random_state=42, max_iter=300)
        nmf.fit(dtm)

        feature_names = vect.get_feature_names_out()
        total_docs = dtm.shape[0]
        results: list[dict] = []

        for tid, component in enumerate(nmf.components_):
            top_indices = component.argsort()[:-11:-1]
            keywords = [feature_names[i] for i in top_indices]
            doc_count = int((dtm[:, top_indices].sum(axis=1) > 0).sum())
            coherence = _compute_coherence(component, top_indices)
            results.append({
                "topic_id":      tid,
                "topic":         _name_topic(keywords),
                "keywords":      keywords,
                "coherence":     coherence,
                "topic_weight":  _topic_weight(coherence, doc_count, total_docs),
                "document_count": doc_count,
                "method":        "NMF",
            })

        results.sort(key=lambda x: x["topic_weight"], reverse=True)
        return _deduplicate_topics(results) or None

    except Exception as e:
        print(f"[Topics] NMF failed: {e}")
        return None


# ── TF-IDF keyword-cluster fallback ───────────────────────────────────────────
def _run_tfidf_fallback(documents: list[str], n_topics: int) -> list[dict]:
    """
    Pure TF-IDF score grouping — no clustering algorithms.
    Ranks terms by global TF-IDF score, assigns them to topic buckets.
    Safe for corpora as small as 1 document.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        tokenized = [_tokenize_for_vectorizer(d) for d in documents]
        stop_words = _get_stop_words()

        vect = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=(1, 2),
            max_features=500,
            min_df=1,
            token_pattern=r'\S+',
        )
        matrix = vect.fit_transform(tokenized)
        feature_names = vect.get_feature_names_out()

        # Global TF-IDF importance
        scores = matrix.sum(axis=0).A1
        top_idx = scores.argsort()[::-1][:n_topics * 7]
        top_terms = [feature_names[i] for i in top_idx]

        topics: list[dict] = []
        chunk = 7
        for i in range(0, min(len(top_terms), n_topics * chunk), chunk):
            cluster = top_terms[i:i + chunk]
            if cluster:
                topics.append({
                    "topic_id":      i // chunk,
                    "topic":         _name_topic(cluster),
                    "keywords":      cluster,
                    "coherence":     0.0,
                    "topic_weight":  0.0,
                    "document_count": len(documents),
                    "method":        "keyword-cluster",
                })
        return _deduplicate_topics(topics[:n_topics])

    except Exception as e:
        print(f"[Topics] TF-IDF fallback failed: {e}")
        # Absolute last resort — pure frequency count
        all_text = " ".join(documents)
        tokens = _tokenize_document(all_text)
        all_stops = _get_stop_words()
        filtered = [t for t in tokens if t not in all_stops and len(t) >= 4]
        counter = Counter(filtered)
        top_words = [w for w, _ in counter.most_common(30)]
        topics = []
        for i in range(0, min(len(top_words), 30), 6):
            cluster = top_words[i:i + 6]
            if cluster:
                topics.append({
                    "topic_id":      i // 6,
                    "topic":         _name_topic(cluster),
                    "keywords":      cluster,
                    "coherence":     0.0,
                    "topic_weight":  0.0,
                    "document_count": 1,
                    "method":        "frequency",
                })
        return topics[:n_topics]


# ── Page trends (TF-IDF per page) ────────────────────────────────────────────
def _extract_page_trends(pages: list[dict]) -> list[dict]:
    """Per-page TF-IDF trend — shows how topic focus shifts across document."""
    page_texts: list[str] = []
    page_nums:  list[int] = []

    for page in pages:
        text = page.get("text", "").strip()
        if text:
            page_texts.append(_tokenize_for_vectorizer(text))
            page_nums.append(page.get("page", 0))

    if not page_texts:
        return []

    if len(page_texts) < 2:
        # Single page: frequency fallback
        all_stops = set(_get_stop_words())
        trends = []
        for page in pages:
            tokens = _tokenize_document(page.get("text", ""))
            filtered = [t for t in tokens if t not in all_stops and len(t) >= 4]
            top = [{"keyword": w, "count": c}
                   for w, c in Counter(filtered).most_common(8)]
            trends.append({"page": page.get("page", 0), "top_keywords": top})
        return trends

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        stop_words = _get_stop_words()
        vect = TfidfVectorizer(
            stop_words=stop_words,
            max_features=300,
            token_pattern=r'\S+',
        )
        matrix = vect.fit_transform(page_texts)
        feature_names = vect.get_feature_names_out()

        trends = []
        for i, page_num in enumerate(page_nums):
            row = matrix[i].toarray()[0]
            top_idx = row.argsort()[::-1][:8]
            top_kws = [
                {"keyword": feature_names[j], "score": round(float(row[j]), 4)}
                for j in top_idx if row[j] > 0
            ]
            trends.append({"page": page_num, "top_keywords": top_kws})
        return trends

    except Exception:
        # Frequency fallback
        all_stops = set(_get_stop_words())
        trends = []
        for page_num, text in zip(page_nums, page_texts):
            tokens = text.split()
            filtered = [t for t in tokens if t not in all_stops and len(t) >= 4]
            top = [{"keyword": w, "count": c}
                   for w, c in Counter(filtered).most_common(8)]
            trends.append({"page": page_num, "top_keywords": top})
        return trends


# ── Main ──────────────────────────────────────────────────────────────────────
def extract_topics(doc_data: dict[str, Any]) -> dict[str, Any]:
    ensure_dirs()

    doc_id   = doc_data["doc_id"]
    doc_name = doc_data["document_name"]
    pages    = doc_data.get("pages", [])

    topic_docs: list[str] = [
        para
        for page in pages
        for para in page.get("paragraphs", [])
        if _is_meaningful_paragraph(para, min_words=15)
    ]

    n_paras = len(topic_docs)
    # Adaptive topic count: small docs get fewer, larger docs more
    if n_paras < 10:
        n_topics_target = 3
    elif n_paras < 30:
        n_topics_target = 5
    else:
        n_topics_target = 7

    topics: list[dict] | None = None
    method_used: str | None = None

    if n_paras >= 5:
        topics = _run_lda(topic_docs, n_topics_target)
        if topics:
            method_used = "LDA"

    if topics is None and n_paras >= 3:
        topics = _run_nmf(topic_docs, n_topics_target)
        if topics:
            method_used = "NMF"

    if topics is None:
        fallback_docs = topic_docs if topic_docs else [doc_data.get("full_text", "")]
        topics = _run_tfidf_fallback(fallback_docs, n_topics_target)
        method_used = "keyword-cluster"

    topics = topics or []

    # Global keyword cloud — domain-aware tokens, skip GENERAL + DOMAIN_SENSITIVE stops
    full_text = doc_data.get("full_text", "")
    all_tokens = _tokenize_document(full_text)
    all_stops = _get_stop_words() + list(DOMAIN_SENSITIVE_STOPS)
    all_stops_set = set(all_stops)
    counter = Counter(t for t in all_tokens if t not in all_stops_set and len(t) >= 3)
    top_keywords = [
        {"keyword": w, "count": c}
        for w, c in counter.most_common(30)
    ]

    result: dict[str, Any] = {
        "doc_id":          doc_id,
        "document_name":   doc_name,
        "total_topics":    len(topics),
        "method_used":     method_used,
        "topics":          topics,
        "top_keywords":    top_keywords,
        "page_trends":     _extract_page_trends(pages),
        "paragraphs_used": n_paras,
    }

    out_path = TOPICS_DIR / f"{doc_id}_topics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ── Load / Aggregate ──────────────────────────────────────────────────────────
def load_topics(doc_id: str) -> dict[str, Any] | None:
    path = TOPICS_DIR / f"{doc_id}_topics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_all_topics() -> dict[str, Any]:
    ensure_dirs()
    all_keywords: Counter = Counter()
    all_topics:   list[dict] = []
    doc_count = 0

    for tf in TOPICS_DIR.glob("*_topics.json"):
        try:
            data = json.loads(tf.read_text(encoding="utf-8"))
            doc_count += 1
            for kw in data.get("top_keywords", []):
                all_keywords[kw["keyword"]] += kw.get("count", 1)
            for topic in data.get("topics", []):
                t = dict(topic)
                t["source_doc"]  = data.get("document_name", "")
                t["method_used"] = data.get("method_used", topic.get("method", "unknown"))
                all_topics.append(t)
        except Exception:
            pass

    # topic_distribution uses topic_weight for meaningful sizing (not count=1 for all)
    topic_weights: dict[str, float] = {}
    topic_counts:  Counter = Counter()
    for t in all_topics:
        name = t["topic"]
        topic_counts[name] += 1
        topic_weights[name] = max(topic_weights.get(name, 0.0), t.get("topic_weight", 0.0))

    topic_distribution = [
        {
            "topic":        name,
            "count":        topic_counts[name],
            "topic_weight": round(topic_weights[name], 4),
        }
        for name, _ in topic_counts.most_common(20)
    ]

    return {
        "total_documents_analyzed": doc_count,
        "total_topics":             len(all_topics),
        "top_keywords":             [{"keyword": k, "count": c}
                                     for k, c in all_keywords.most_common(30)],
        "topic_distribution":       topic_distribution,
        "all_topics":               all_topics[:50],
    }