"""
PHASE 2: Domain Detector

Detects document domain using keyword scoring + entity distribution.
No ML model required - uses curated domain dictionaries.

Supported domains:
- RESEARCH (academic papers, arXiv, conferences)
- LEGAL (contracts, regulations, legal documents)
- MEDICAL (clinical trials, medical research, healthcare)
- FINANCE (financial reports, earnings, market analysis)
- TECH (technical documentation, software, APIs)
- GENERAL (fallback for mixed/unclear domains)

Output format:
{
    "domain": "...",
    "confidence": 0.0,
    "scores": {...}  # per-domain scores for transparency
}
"""

import re
from typing import Any
from collections import Counter


# ── Domain Keyword Dictionaries ────────────────────────────────────────────────
# Each domain has signature keywords that indicate its presence

RESEARCH_KEYWORDS = {
    # Academic structure
    "abstract", "introduction", "methodology", "experiments", "results",
    "discussion", "conclusion", "references", "bibliography", "acknowledgments",
    # Research verbs
    "propose", "demonstrate", "evaluate", "compare", "outperform",
    "achieve", "introduce", "present", "investigate", "analyze",
    # Academic terms
    "state-of-the-art", "baseline", "benchmark", "dataset", "corpus",
    "training", "validation", "test set", "cross-validation",
    # Publication markers
    "arxiv", "preprint", "peer-reviewed", "journal", "conference",
    "proceedings", "volume", "issue", "doi",
    # Common in ML/AI papers
    "transformer", "attention", "neural network", "deep learning",
    "model", "architecture", "layer", "embedding", "encoder", "decoder",
}

LEGAL_KEYWORDS = {
    # Legal document structure
    "whereas", "therefore", "herein", "hereof", "hereto", "hereby",
    "notwithstanding", "aforementioned", "pursuant to", "in accordance with",
    # Legal parties
    "party", "parties", "plaintiff", "defendant", "petitioner", "respondent",
    "appellant", "appellee", "licensor", "licensee",
    # Legal concepts
    "agreement", "contract", "clause", "provision", "amendment",
    "termination", "breach", "liability", "indemnification", "warranty",
    "representation", "covenant", "obligation", "right", "duty",
    # Legal actions
    "agree", "consent", "waive", "execute", "deliver", "perform",
    # Document types
    "exhibit", "schedule", "appendix", "attachment", "addendum",
}

MEDICAL_KEYWORDS = {
    # Medical structure
    "patient", "subjects", "participants", "cohort", "randomized",
    "double-blind", "placebo-controlled", "clinical trial",
    # Medical outcomes
    "efficacy", "safety", "tolerability", "adverse event", "side effect",
    "response rate", "remission", "relapse", "progression-free survival",
    # Medical measurements
    "dose", "dosage", "mg/kg", "administration", "pharmacokinetics",
    "biomarker", "endpoint", "primary endpoint", "secondary endpoint",
    # Medical conditions
    "diagnosis", "treatment", "therapy", "prognosis", "symptom",
    "disease", "disorder", "condition", "syndrome",
    # Medical organizations
    "fda", "ema", "clinicaltrials.gov", "irb", "ethics committee",
}

FINANCE_KEYWORDS = {
    # Financial metrics
    "revenue", "earnings", "net income", "gross profit", "ebitda",
    "eps", "pe ratio", "price-to-earnings", "dividend", "yield",
    "roe", "roa", "roi", "margin", "cash flow",
    # Financial statements
    "balance sheet", "income statement", "cash flow statement",
    "quarterly report", "annual report", "10-k", "10-q", "8-k",
    # Market terms
    "market cap", "market capitalization", "shares outstanding",
    "stock price", "trading volume", "beta", "volatility",
    # Corporate actions
    "merger", "acquisition", "spin-off", "ipo", "offering",
    "buyback", "split", "consolidation",
    # Economic indicators
    "gdp", "inflation", "interest rate", "fed", "central bank",
}

TECH_KEYWORDS = {
    # Software development
    "api", "sdk", "library", "framework", "module", "package",
    "repository", "commit", "pull request", "merge", "branch",
    # Architecture
    "microservice", "container", "kubernetes", "docker", "serverless",
    "cloud", "aws", "azure", "gcp", "deployment", "infrastructure",
    # Programming
    "code", "function", "class", "method", "interface", "protocol",
    "algorithm", "data structure", "complexity", "optimization",
    # Systems
    "database", "sql", "nosql", "cache", "queue", "message broker",
    "load balancer", "proxy", "cdn", "latency", "throughput",
    # Web/Mobile
    "frontend", "backend", "rest", "graphql", "http", "https",
    "responsive", "mobile", "web application", "single-page",
}


# ── Entity-Based Domain Signals ────────────────────────────────────────────────
# Certain entity types strongly indicate specific domains

DOMAIN_ENTITY_SIGNALS = {
    "RESEARCH": {"MODEL", "METRIC", "BENCHMARK", "DATASET", "VENUE", "JOURNAL"},
    "LEGAL": {"LAW", "LEGAL_TERM"},
    "MEDICAL": {"MEDICAL_TERM"},
    "FINANCE": {"FINANCIAL_TERM"},
    "TECH": {"PRODUCT", "FUNCTION", "TOKEN"},
}


# ── Domain Detection Logic ─────────────────────────────────────────────────────
def _count_keyword_matches(text: str, keywords: set[str]) -> int:
    """Count how many domain-specific keywords appear in text."""
    text_lower = text.lower()
    count = 0
    for kw in keywords:
        if kw in text_lower:
            count += 1
    return count


def _get_entity_domain_signal(entity_types: list[str]) -> dict[str, float]:
    """
    Calculate domain scores based on entity type distribution.
    Returns dict of domain -> score.
    """
    type_counts = Counter(entity_types)
    domain_scores: dict[str, float] = {}
    
    for domain, signal_types in DOMAIN_ENTITY_SIGNALS.items():
        score = sum(type_counts.get(t, 0) for t in signal_types)
        domain_scores[domain] = score
    
    return domain_scores


def detect_domain(doc_data: dict[str, Any]) -> dict[str, Any]:
    """
    Detect the primary domain of a document.
    
    Uses two signals:
    1. Keyword matching against domain dictionaries
    2. Entity type distribution
    
    Args:
        doc_data: Processed document data with 'full_text' and optionally 'entities'
    
    Returns:
        {
            "domain": str,           # Detected domain
            "confidence": float,     # 0.0-1.0 confidence score
            "scores": dict,          # Raw scores per domain
            "signals": dict          # Breakdown of keyword vs entity signals
        }
    """
    full_text = doc_data.get("full_text", "")
    entities = doc_data.get("entities", [])
    
    # Extract entity types if available
    entity_types = []
    if isinstance(entities, list):
        for ent in entities:
            if isinstance(ent, dict) and "type" in ent:
                entity_types.append(ent["type"])
    
    # ── Signal 1: Keyword matching ────────────────────────────────────────────
    keyword_scores = {
        "RESEARCH": _count_keyword_matches(full_text, RESEARCH_KEYWORDS),
        "LEGAL": _count_keyword_matches(full_text, LEGAL_KEYWORDS),
        "MEDICAL": _count_keyword_matches(full_text, MEDICAL_KEYWORDS),
        "FINANCE": _count_keyword_matches(full_text, FINANCE_KEYWORDS),
        "TECH": _count_keyword_matches(full_text, TECH_KEYWORDS),
    }
    
    # ── Signal 2: Entity distribution ─────────────────────────────────────────
    entity_scores = _get_entity_domain_signal(entity_types)
    
    # Normalize entity scores to comparable scale with keyword scores
    max_keyword_score = max(keyword_scores.values()) if keyword_scores else 1
    if max_keyword_score > 0:
        # Scale entity scores to ~30% weight relative to keywords
        for domain in entity_scores:
            entity_scores[domain] = entity_scores.get(domain, 0) * (max_keyword_score / 10)
    
    # ── Combine signals ───────────────────────────────────────────────────────
    combined_scores: dict[str, float] = {}
    all_domains = set(keyword_scores.keys()) | set(entity_scores.keys())
    
    for domain in all_domains:
        kw_score = keyword_scores.get(domain, 0)
        ent_score = entity_scores.get(domain, 0)
        combined_scores[domain] = kw_score + ent_score
    
    # ── Determine winner and confidence ───────────────────────────────────────
    if not combined_scores or max(combined_scores.values()) == 0:
        return {
            "domain": "GENERAL",
            "confidence": 0.5,
            "scores": {"GENERAL": 1.0},
            "signals": {
                "keyword_scores": keyword_scores,
                "entity_scores": entity_scores,
            },
        }
    
    sorted_domains = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    winner_domain = sorted_domains[0][0]
    winner_score = sorted_domains[0][1]
    
    # Calculate confidence based on margin over second place
    if len(sorted_domains) > 1:
        second_score = sorted_domains[1][1]
        # Confidence increases with larger margin
        margin = winner_score - second_score
        total = winner_score + second_score if winner_score + second_score > 0 else 1
        base_confidence = winner_score / sum(combined_scores.values())
        margin_bonus = min(margin / total, 0.3)  # Cap margin bonus at 0.3
        confidence = min(base_confidence + margin_bonus, 1.0)
    else:
        # Only one domain detected - high confidence but not absolute
        confidence = min(0.7 + (winner_score / 100) * 0.3, 1.0)
    
    # Round scores for output
    rounded_scores = {k: round(v, 4) for k, v in combined_scores.items()}
    
    return {
        "domain": winner_domain,
        "confidence": round(confidence, 4),
        "scores": rounded_scores,
        "signals": {
            "keyword_scores": keyword_scores,
            "entity_scores": {k: round(v, 4) for k, v in entity_scores.items()},
        },
    }


def detect_domain_from_text(text: str) -> dict[str, Any]:
    """
    Convenience function to detect domain from raw text only.
    
    Args:
        text: Raw document text
    
    Returns:
        Same format as detect_domain()
    """
    return detect_domain({"full_text": text, "entities": []})


# ── Integration Points ────────────────────────────────────────────────────────
"""
INTEGRATION POINTS:

1. In document_processor.py after text extraction:
   
   from backend.core.domain_detector import detect_domain
   
   def process_document(file_path, doc_id):
       result = ...  # existing extraction
       domain_info = detect_domain(result)
       result["domain"] = domain_info["domain"]
       result["domain_confidence"] = domain_info["confidence"]
       return result

2. In entity_extractor.py for domain-aware extraction:
   
   from backend.core.domain_detector import detect_domain
   
   def extract_entities(doc_data):
       # First detect domain
       domain_info = detect_domain(doc_data)
       domain = domain_info["domain"]
       
       # Use domain to adjust extraction behavior
       # e.g., load domain-specific patterns
   
   Pass domain info to downstream modules.

3. In topic_extractor.py for domain-aware topic naming:
   
   def extract_topics(doc_data):
       domain = doc_data.get("domain", "GENERAL")
       # Use domain to select appropriate topic naming templates

4. In analytics.py for domain-based filtering:
   
   def get_platform_analytics():
       # Group documents by domain
       # Show domain-specific insights
"""
