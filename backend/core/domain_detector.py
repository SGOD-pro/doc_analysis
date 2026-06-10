"""
Phase 2: Domain Detector

Detects document domain using keyword scoring.
Supported domains: RESEARCH, LEGAL, MEDICAL, FINANCE, TECH, GENERAL

Returns: {"domain": "...", "confidence": 0.0}
"""

import json
import re
from pathlib import Path
from typing import TypedDict
from collections import Counter


class DomainResult(TypedDict):
    domain: str
    confidence: float
    scores: dict[str, float]


# Domain keyword dictionaries
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "RESEARCH": [
        # ML/AI research terms
        "neural network", "deep learning", "machine learning", "transformer",
        "attention mechanism", "encoder", "decoder", "embedding", "latent",
        "training", "validation", "test set", "benchmark", "dataset",
        "model architecture", "hyperparameter", "loss function", "optimizer",
        "gradient descent", "backpropagation", "fine-tuning", "pre-trained",
        "transfer learning", "supervised", "unsupervised", "reinforcement learning",
        "accuracy", "precision", "recall", "f1 score", "auc", "roc",
        "conference", "paper", "arxiv", "peer-reviewed", "citation",
        "experiment", "evaluation", "baseline", "state-of-the-art", "novel",
        "methodology", "ablation study", "empirical", "quantitative",
    ],
    "LEGAL": [
        # Legal terms
        "agreement", "contract", "party", "parties", "herein", "thereof",
        "whereas", "indemnification", "liability", "warranty", "representation",
        "governing law", "jurisdiction", "venue", "arbitration", "mediation",
        "confidentiality", "non-disclosure", "proprietary", "intellectual property",
        "patent", "trademark", "copyright", "license", "licensor", "licensee",
        "termination", "breach", "remedy", "damages", "injunction",
        "force majeure", "act of god", "severability", "amendment",
        "assignment", "successor", "binding", "enforceable", "validity",
        "plaintiff", "defendant", "court", "judge", "attorney", "counsel",
        "statute", "regulation", "compliance", "legal", "law", "clause",
    ],
    "MEDICAL": [
        # Medical/clinical terms
        "patient", "diagnosis", "treatment", "therapy", "medication",
        "clinical trial", "randomized", "placebo", "double-blind", "cohort",
        "symptom", "syndrome", "disease", "disorder", "condition",
        "pathology", "etiology", "prognosis", "epidemiology", "prevalence",
        "incidence", "mortality", "morbidity", "survival rate", "remission",
        "pharmacology", "dosage", "administration", "adverse event", "side effect",
        "therapeutic", "efficacy", "safety", "tolerability", "bioavailability",
        "biomarker", "assay", "specimen", "sample", "laboratory",
        "hospital", "clinic", "physician", "nurse", "healthcare",
        "medical", "health", "clinical", "pharmaceutical", "drug",
    ],
    "FINANCE": [
        # Financial terms
        "revenue", "earnings", "profit", "loss", "income", "expense",
        "asset", "liability", "equity", "debt", "capital", "cash flow",
        "balance sheet", "income statement", "financial statement",
        "audit", "accounting", "fiscal year", "quarterly", "annual report",
        "stock", "share", "dividend", "ipo", "market cap", "valuation",
        "investment", "portfolio", "return", "roi", "roe", "roa",
        "risk", "volatility", "beta", "alpha", "sharpe ratio",
        "merger", "acquisition", "leveraged", "derivative", "hedge",
        "bank", "lender", "borrower", "interest rate", "bond", "yield",
        "ebitda", "pe ratio", "price-to-earnings", "book value",
    ],
    "TECH": [
        # Technical/documentation terms
        "api", "endpoint", "request", "response", "http", "https",
        "authentication", "authorization", "token", "oauth", "jwt",
        "database", "schema", "query", "index", "table", "row", "column",
        "server", "client", "service", "microservice", "container",
        "deployment", "ci/cd", "pipeline", "build", "release", "version",
        "software", "hardware", "system", "architecture", "component",
        "interface", "module", "library", "framework", "sdk", "toolkit",
        "documentation", "specification", "implementation", "configuration",
        "bug", "issue", "fix", "patch", "update", "upgrade", "migration",
        "cloud", "aws", "azure", "gcp", "kubernetes", "docker",
    ],
}

# Normalize keywords to lowercase for matching
DOMAIN_KEYWORDS_LOWER: dict[str, set[str]] = {
    domain: {kw.lower() for kw in keywords}
    for domain, keywords in DOMAIN_KEYWORDS.items()
}

# Domain priority for tie-breaking (higher = more specific)
DOMAIN_PRIORITY: dict[str, int] = {
    "LEGAL": 5,
    "MEDICAL": 5,
    "FINANCE": 5,
    "RESEARCH": 4,
    "TECH": 3,
    "GENERAL": 1,
}


def _count_keyword_matches(text: str) -> dict[str, int]:
    """Count keyword matches per domain."""
    text_lower = text.lower()
    # Tokenize into words and bigrams for better matching
    words = re.findall(r'\b[a-z][a-z\-/]*[a-z]\b|\b[a-z]\b', text_lower)
    
    # Create bigrams
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    all_tokens = set(words + bigrams)
    
    counts: dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS_LOWER.items():
        count = sum(1 for token in all_tokens if token in keywords)
        # Also check for multi-word phrases directly in text
        for kw in keywords:
            if ' ' in kw and kw in text_lower:
                count += 1
        counts[domain] = count
    
    return counts


def _calculate_confidence(counts: dict[str, int], total_words: int) -> tuple[str, float]:
    """Calculate domain and confidence from keyword counts."""
    if not counts or total_words == 0:
        return "GENERAL", 0.5
    
    # Find max count
    max_count = max(counts.values())
    
    if max_count == 0:
        return "GENERAL", 0.5
    
    # Get all domains with max count
    top_domains = [d for d, c in counts.items() if c == max_count]
    
    # Break ties by priority
    if len(top_domains) > 1:
        top_domains.sort(key=lambda d: DOMAIN_PRIORITY.get(d, 0), reverse=True)
    
    domain = top_domains[0]
    
    # Calculate confidence based on:
    # 1. Count magnitude (more matches = higher confidence)
    # 2. Gap between top and second place (larger gap = higher confidence)
    
    sorted_counts = sorted(counts.values(), reverse=True)
    second_count = sorted_counts[1] if len(sorted_counts) > 1 else 0
    
    # Magnitude factor: log scale, capped at 1.0
    import math
    magnitude_factor = min(1.0, math.log(max_count + 1) / math.log(50))
    
    # Gap factor: how much higher is top vs second
    if max_count > 0:
        gap_factor = (max_count - second_count) / max_count
    else:
        gap_factor = 0
    
    # Weighted combination
    confidence = 0.6 * magnitude_factor + 0.4 * gap_factor
    confidence = max(0.5, min(1.0, confidence))  # Clamp to [0.5, 1.0]
    
    return domain, round(confidence, 3)


def detect_domain(text: str) -> DomainResult:
    """
    Detect document domain using keyword scoring.
    
    Args:
        text: Document text to analyze
        
    Returns:
        DomainResult with domain, confidence, and per-domain scores
    """
    if not text or not text.strip():
        return DomainResult(
            domain="GENERAL",
            confidence=0.5,
            scores={d: 0.0 for d in DOMAIN_KEYWORDS.keys()}
        )
    
    # Count keyword matches
    counts = _count_keyword_matches(text)
    
    # Normalize scores to [0, 1] range
    max_count = max(counts.values()) if counts else 1
    scores = {
        domain: round(count / max(max_count, 1), 3)
        for domain, count in counts.items()
    }
    
    # Calculate domain and confidence
    total_words = len(text.split())
    domain, confidence = _calculate_confidence(counts, total_words)
    
    return DomainResult(
        domain=domain,
        confidence=confidence,
        scores=scores
    )


def detect_domain_from_file(file_path: Path | str) -> DomainResult:
    """Detect domain from a file path."""
    path = Path(file_path)
    if not path.exists():
        return DomainResult(
            domain="GENERAL",
            confidence=0.5,
            scores={d: 0.0 for d in DOMAIN_KEYWORDS.keys()}
        )
    
    text = path.read_text(encoding="utf-8", errors="ignore")
    return detect_domain(text)


# Cache for domain detection results
_domain_cache: dict[str, DomainResult] = {}


def get_domain_for_document(doc_data: dict) -> DomainResult:
    """
    Get domain for a processed document.
    Uses caching to avoid re-computation.
    
    Args:
        doc_data: Processed document data with 'doc_id' and 'full_text'
        
    Returns:
        DomainResult
    """
    doc_id = doc_data.get("doc_id", "")
    
    if doc_id in _domain_cache:
        return _domain_cache[doc_id]
    
    full_text = doc_data.get("full_text", "")
    result = detect_domain(full_text)
    
    _domain_cache[doc_id] = result
    return result


def clear_cache():
    """Clear the domain detection cache."""
    _domain_cache.clear()


def get_domain_keywords(domain: str) -> list[str]:
    """Get the keyword list for a specific domain."""
    return DOMAIN_KEYWORDS.get(domain, [])


def add_domain_keyword(domain: str, keyword: str):
    """Add a keyword to a domain's dictionary."""
    if domain not in DOMAIN_KEYWORDS:
        DOMAIN_KEYWORDS[domain] = []
    if keyword not in DOMAIN_KEYWORDS[domain]:
        DOMAIN_KEYWORDS[domain].append(keyword)
        DOMAIN_KEYWORDS_LOWER[domain].add(keyword.lower())


def save_domain_config(path: Path | str = None):
    """Save current domain configuration to JSON."""
    if path is None:
        path = Path(__file__).parent / "domain_config.json"
    
    config = {
        "domain_keywords": DOMAIN_KEYWORDS,
        "domain_priority": DOMAIN_PRIORITY,
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_domain_config(path: Path | str = None):
    """Load domain configuration from JSON."""
    global DOMAIN_KEYWORDS, DOMAIN_KEYWORDS_LOWER, DOMAIN_PRIORITY
    
    if path is None:
        path = Path(__file__).parent / "domain_config.json"
    
    if not Path(path).exists():
        return
    
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    if "domain_keywords" in config:
        DOMAIN_KEYWORDS = config["domain_keywords"]
        DOMAIN_KEYWORDS_LOWER = {
            domain: {kw.lower() for kw in keywords}
            for domain, keywords in DOMAIN_KEYWORDS.items()
        }
    
    if "domain_priority" in config:
        DOMAIN_PRIORITY = config["domain_priority"]
