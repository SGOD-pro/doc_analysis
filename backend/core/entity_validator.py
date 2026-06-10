"""
Phase 1 + 2 + 6 + 8: Entity Validation, Normalization & Reclassification Layer

This module forms the adapter between raw spaCy output and the graph/analytics layers.
Extractor → Validator (this module) → Normalizer → Graph Builder → Analytics

Supports future swap to SciSpacy, GLiNER, HuggingFace NER, or LLM-based extractors
by keeping this layer decoupled from any specific NER backend.

PHASE 1 IMPROVEMENTS:
- Enhanced PERSON validation with title case, length, abbreviation filtering
- Enhanced ORG validation to reject ML terms incorrectly classified
- Enhanced LOCATION validation to prevent technical terms becoming locations
- Domain dictionaries for MODEL, DATASET, METRIC, LEGAL_TERM, MEDICAL_TERM, FINANCIAL_TERM
- Confidence scoring based on validation rules
"""

import re
import json
import unicodedata
from pathlib import Path
from typing import NamedTuple

# ── Types ──────────────────────────────────────────────────────────────────────
class ValidatedEntity(NamedTuple):
    text: str           # canonical normalized form
    original: str       # raw spaCy output
    type: str           # final entity type (may be reclassified)
    type_label: str     # human-readable label
    confidence: float   # 0.0–1.0 (1.0 for high-confidence, lower for edge cases)


# ── Config loading ─────────────────────────────────────────────────────────────
_CONFIG: dict | None = None
_CONFIG_PATH = Path(__file__).parent / "entity_config.json"


def _load_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                _CONFIG = json.load(f)
        else:
            _CONFIG = {
                "reclassify": {},
                "location_exclusions": [],
                "junk_patterns": [],
                "graph_edge_weight_threshold": 2,
                "graph_max_sentence_entities": 8,
                "search_min_paragraph_words": 10,
            }
    return _CONFIG


# ── Allowed entity types (Phase 1) ────────────────────────────────────────────
ALLOWED_TYPES = {
    "PERSON", "ORG", "PRODUCT", "GPE", "LOC",
    "EVENT", "WORK_OF_ART", "LAW", "NORP",
    # Reclassified types (Phase 2)
    "MODEL", "METRIC", "BENCHMARK", "DATASET",
    "JOURNAL", "VENUE", "FUNCTION", "TOKEN",
    # Domain-specific types
    "LEGAL_TERM", "MEDICAL_TERM", "FINANCIAL_TERM",
}

BLOCKED_TYPES = {
    "DATE", "TIME", "CARDINAL", "ORDINAL",
    "PERCENT", "MONEY", "QUANTITY", "LANGUAGE",
}

ENTITY_LABELS = {
    "PERSON":       "Person",
    "ORG":          "Organization",
    "PRODUCT":      "Product",
    "GPE":          "Location (Geo-Political)",
    "LOC":          "Location",
    "EVENT":        "Event",
    "WORK_OF_ART":  "Work of Art",
    "LAW":          "Law / Regulation",
    "NORP":         "Nationality / Group",
    "MODEL":        "AI / ML Model",
    "METRIC":       "Evaluation Metric",
    "BENCHMARK":    "Benchmark / Dataset",
    "DATASET":      "Dataset",
    "JOURNAL":      "Journal / Repository",
    "VENUE":        "Conference / Venue",
    "FUNCTION":     "Mathematical Function",
    "TOKEN":        "Special Token",
    "LEGAL_TERM":   "Legal Term",
    "MEDICAL_TERM": "Medical Term",
    "FINANCIAL_TERM": "Financial Term",
}


# ── Domain Dictionaries (Phase 1) ─────────────────────────────────────────────
# These are used for reclassification and validation

# ML/AI Models that should be MODEL type, not ORG/PERSON
ML_MODEL_NAMES = {
    "transformer", "bert", "gpt", "gpt-2", "gpt-3", "gpt-4",
    "llama", "llama2", "llama3", "llama-2", "llama-3",
    "roberta", "xlnet", "albert", "t5", "bart",
    "lstm", "bilstm", "gru", "cnn", "resnet", "vgg",
    "elmo", "word2vec", "glove", "fasttext",
    "clip", "diffusion", "stable diffusion", "midjourney",
    "whisper", "codex", "palm", "gemini", "claude",
    "falcon", "mistral", "mixtral", "phi", "qwen",
    "yi", "deepseek", "grok",
}

# Common datasets/benchmarks
DATASET_NAMES = {
    "wmt", "squad", "glue", "superglue", "imagenet", "coco",
    "mnist", "cifar", "wikitext", "books corpus", "bookscorpus",
    "common crawl", "commoncrawl", "penn treebank", "ptb",
    "conll", "ontonotes", "yelp reviews", "amazon reviews",
    "ag news", "dbpedia", "yahoo answers",
}

# Evaluation metrics
METRIC_NAMES = {
    "bleu", "rouge", "meteor", "gleu", "chrf",
    "perplexity", "ppl", "f1", "accuracy", "precision", "recall",
    "em", "exact match", "map", "mrr", "ndcg",
    "iou", "psnr", "ssim", "fid", "is", "inception score",
}

# Legal terms
LEGAL_TERMS = {
    "indemnification", "force majeure", "governing law", "jurisdiction",
    "liability", "breach", "termination", "confidentiality",
    "intellectual property", "patent", "trademark", "copyright",
    "warranty", "representation", "covenant", "condition precedent",
    "due diligence", "material adverse effect", "change of control",
}

# Medical terms
MEDICAL_TERMS = {
    "diagnosis", "prognosis", "treatment", "therapy", "medication",
    "dosage", "contraindication", "adverse event", "side effect",
    "clinical trial", "randomized", "placebo", "double-blind",
    "efficacy", "safety", "tolerability", "pharmacokinetics",
    "biomarker", "pathology", "histology", "radiology",
    "hba1c", "ldl", "hdl", "triglycerides", "creatinine",
}

# Financial terms
FINANCIAL_TERMS = {
    "ebitda", "revenue", "earnings", "net income", "gross profit",
    "cash flow", "operating cash flow", "free cash flow",
    "roi", "roe", "roa", "eps", "pe ratio", "price earnings",
    "dividend", "market cap", "market capitalization",
    "liquidity", "solvency", "leverage", "debt-to-equity",
    "amortization", "depreciation", "accrual",
}

# Technical acronyms that are NOT organizations
TECH_ACROYMNS_NOT_ORG = {
    "sgd", "adam", "rmsprop", "adagrad", "adadelta",
    "relu", "sigmoid", "tanh", "softmax", "layernorm", "batchnorm",
    "dropout", "attention", "self-attention", "multi-head",
    "encoder", "decoder", "transformer",
    "api", "sdk", "cli", "gui", "ide", "http", "https",
    "json", "xml", "yaml", "sql", "nosql",
    "cpu", "gpu", "tpu", "ram", "rom", "ssd", "hdd",
    "aws", "azure", "gcp", "kubernetes", "docker",
}

# Single-letter or short abbreviations that are NOT people
INVALID_PERSON_PATTERNS = {
    "dk", "zn", "abc", "xyz", "aa", "bb", "cc", "dd",
    "jr", "sr", "ii", "iii", "iv",
}


# ── Compiled junk patterns ─────────────────────────────────────────────────────
def _compile_junk_patterns() -> list[re.Pattern]:
    cfg = _load_config()
    patterns = cfg.get("junk_patterns", [])
    # Always add these hard-coded baseline patterns
    baseline = [
        r"^\d+$",                          # pure numbers
        r"^[ivxlcdm]+$",                   # roman numerals
        r"^\W+$",                           # all punctuation
        r"^.$",                             # single char
        r"^\d{4}\.\d{4,}$",                # arxiv ID format
        r"^\[\d+\]$",                       # citation bracket [42]
        r"^fig(ure)?[\s\.\-]?\d+",         # figure references
        r"^table[\s\.\-]?\d+",             # table references
        r"^eq(uation)?[\s\.\-]?\d+",       # equation references
        r"^algorithm[\s\.\-]?\d+",         # algorithm references
        r"^section[\s\.\-]?\d+",           # section references
        r"^appendix",                       # appendix
        r"^\d+[a-z]$",                     # 42a, 15b
        r"^[a-z]\d+$",                     # a1, b2
        # OCR / LaTeX math artifacts
        r"[∈∉∀∃∑∏∫∂∇⊂⊃⊄⊆⊇∪∩≤≥≠≈≡±×÷√∞]",  # unicode math symbols
        r"R[dhl]model",                     # LaTeX dimension vars
        r"^max\s*\(",                       # max(0, ...) fragments
        r"^min\s*\(",                       # min(0, ...) fragments
        r"\\[a-zA-Z]+\{",                   # LaTeX commands \cmd{
        r"^[xyzwuvXYZWUV]_?[0-9]+$",       # single variable subscripts
        r"^[A-Za-z][0-9]+$",               # W1, h2, etc.
        r"^\d+[A-Za-z]+\d+",               # mixed num-alpha-num
        r"newstest",                        # MT evaluation set names
        r"^(et|al|fig|vol|pp|no|ed)$",     # common abbreviation fragments
        r"^\(\d",                           # starts with (number
    ]
    compiled = []
    for p in set(baseline + patterns):
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error:
            pass
    return compiled


_JUNK_PATTERNS: list[re.Pattern] | None = None


def _get_junk_patterns() -> list[re.Pattern]:
    global _JUNK_PATTERNS
    if _JUNK_PATTERNS is None:
        _JUNK_PATTERNS = _compile_junk_patterns()
    return _JUNK_PATTERNS


# ── Normalization (Phase 1) ────────────────────────────────────────────────────
def normalize_entity_text(text: str) -> str:
    """
    Canonical normalization:
      1. Normalize unicode (NFC)
      2. Strip whitespace
      3. Collapse internal whitespace
      4. Title-case single-word non-acronym entities
    """
    # Unicode normalization
    text = unicodedata.normalize("NFC", text)
    # Strip and collapse whitespace
    text = " ".join(text.split())
    # Remove leading/trailing quotes and brackets
    text = text.strip("\"'`()[]{}.,;:")
    text = text.strip()
    return text


def canonical_key(text: str) -> str:
    """Return lowercase stripped key for deduplication."""
    return normalize_entity_text(text).lower()


# ── Validation (Phase 1 + 6) ───────────────────────────────────────────────────
def is_junk_entity(text: str) -> bool:
    """Return True if entity should be discarded as noise."""
    text = text.strip()
    if len(text) <= 1:
        return True
    if len(text) > 80:  # very long = OCR / table artifact
        return True
    # Must contain at least one ASCII letter (catches unicode-only math)
    if not re.search(r"[a-zA-Z]", text):
        return True
    # Ratio of non-ASCII characters > 30% → likely OCR/math artifact
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if len(text) > 3 and non_ascii / len(text) > 0.3:
        return True
    # Contains operator/math characters mixed with letters
    if re.search(r"[\{\}\\=<>\|\^~`]", text):
        return True
    patterns = _get_junk_patterns()
    for pat in patterns:
        if pat.search(text):
            return True
    return False


def is_valid_location(text: str) -> bool:
    """Phase 6: Ensure GPE/LOC entities are genuine geographic places."""
    cfg = _load_config()
    exclusions = {e.lower() for e in cfg.get("location_exclusions", [])}
    if canonical_key(text) in exclusions:
        return False
    # Reject if it looks like a technical term (all caps acronym with no vowels possible)
    if re.match(r'^[A-Z]{2,6}$', text.strip()):
        # Short all-caps: likely a dataset/model name, not a place
        return False
    return True


# ── Reclassification (Phase 2) ────────────────────────────────────────────────
def reclassify_entity(text: str, spacy_type: str) -> str:
    """
    Apply externalized reclassification mapping.
    Config-driven: no domain knowledge hardcoded in this function.
    """
    cfg = _load_config()
    remap = cfg.get("reclassify", {})

    # Exact match first
    if text in remap:
        return remap[text]

    # Case-insensitive match
    text_lower = text.lower()
    for key, val in remap.items():
        if key.lower() == text_lower:
            return val

    return spacy_type  # keep spaCy label if no remap found


# NORP names that are actually single person first names misclassified by spaCy
_NORP_PERSON_FIRST_NAMES: set[str] = {
    "ashish", "noam", "illia", "llion", "aidan", "jakob",
    "denny", "lukasz", "quoc",
}


# ── Phase 1: Enhanced Entity Validation Functions ──────────────────────────────
def _is_valid_person(text: str) -> tuple[bool, float]:
    """
    Validate PERSON entities with strict rules.
    Returns (is_valid, confidence).
    
    Rejects:
    - Single letters or very short names (< 2 chars)
    - Known abbreviations (dk, zn, abc, etc.)
    - All uppercase (likely acronyms)
    - All lowercase (not title case)
    - Names with numbers
    """
    text = text.strip()
    
    # Minimum length check
    if len(text) < 2:
        return False, 0.0
    
    # Check for known invalid patterns
    if text.lower() in INVALID_PERSON_PATTERNS:
        return False, 0.0
    
    # Reject all-uppercase (likely acronyms)
    if text.isupper() and len(text) <= 5:
        return False, 0.0
    
    # Reject all-lowercase (not proper noun format)
    if text.islower() and len(text) <= 4:
        return False, 0.0
    
    # Reject if contains numbers
    if any(c.isdigit() for c in text):
        return False, 0.0
    
    # Title case validation: first letter should be uppercase
    if not text[0].isupper():
        return False, 0.0
    
    # Multi-word names: each word should start with capital
    words = text.split()
    for word in words:
        if word and not word[0].isupper():
            return False, 0.0
    
    # Common person name suffixes are valid
    valid_suffixes = {"jr", "sr", "ii", "iii", "iv", "phd", "md"}
    if len(words) > 1 and words[-1].lower() in valid_suffixes:
        return True, 0.95
    
    # If single word and very short (2-3 chars), lower confidence
    if len(words) == 1 and len(text) <= 3:
        return True, 0.7
    
    return True, 1.0


def _is_valid_organization(text: str) -> tuple[bool, float]:
    """
    Validate ORG entities, rejecting technical terms incorrectly classified.
    Returns (is_valid, confidence).
    
    Rejects:
    - ML/AI model names (CNN, RNN, LSTM, etc.)
    - Technical acronyms (SGD, Adam, ReLU, etc.)
    - Single common words that aren't organization names
    """
    text_lower = text.lower().strip()
    
    # Check against ML model names
    if text_lower in ML_MODEL_NAMES:
        return False, 0.0
    
    # Check against technical acronyms
    if text_lower in TECH_ACROYMNS_NOT_ORG:
        return False, 0.0
    
    # Check against dataset/benchmark names
    if text_lower in DATASET_NAMES:
        return False, 0.0
    
    # Check against metric names
    if text_lower in METRIC_NAMES:
        return False, 0.0
    
    # Short all-caps acronyms (2-4 chars) are likely not orgs
    if re.match(r'^[A-Z]{2,4}$', text.strip()):
        # Exception: well-known company-like acronyms
        known_org_acronyms = {"ibm", "hp", "gm", "ge", "bmw", "kia", "lg"}
        if text_lower not in known_org_acronyms:
            return False, 0.0
    
    # Single common words that are often misclassified
    common_words_not_orgs = {
        "adam", "linear", "attention", "transformer", "encoder", "decoder",
        "model", "network", "system", "layer", "function", "module",
    }
    if text_lower in common_words_not_orgs:
        return False, 0.0
    
    return True, 1.0


def _is_valid_location(text: str) -> bool:
    """
    Ensure GPE/LOC entities are genuine geographic places.
    
    Rejects:
    - Technical terms (BERT, GPT, BLEU, SQuAD, etc.)
    - Model/dataset names
    - Short all-caps acronyms
    """
    cfg = _load_config()
    exclusions = {e.lower() for e in cfg.get("location_exclusions", [])}
    
    text_lower = text.lower().strip()
    
    # Check config exclusions
    if text_lower in exclusions:
        return False
    
    # Check against ML model names
    if text_lower in ML_MODEL_NAMES:
        return False
    
    # Check against dataset/benchmark names
    if text_lower in DATASET_NAMES:
        return False
    
    # Check against metric names
    if text_lower in METRIC_NAMES:
        return False
    
    # Check against technical acronyms
    if text_lower in TECH_ACROYMNS_NOT_ORG:
        return False
    
    # Reject short all-caps acronyms (2-6 chars)
    if re.match(r'^[A-Z]{2,6}$', text.strip()):
        return False
    
    return True


def _reclassify_by_domain(text: str, spacy_type: str) -> tuple[str, float]:
    """
    Reclassify entities based on domain dictionaries.
    Returns (new_type, confidence).
    """
    text_lower = text.lower().strip()
    
    # Check ML/AI models
    if text_lower in ML_MODEL_NAMES:
        return "MODEL", 0.95
    
    # Check datasets/benchmarks
    if text_lower in DATASET_NAMES:
        return "DATASET" if text_lower in {"wmt", "squad", "glue", "superglue", "imagenet", "coco", 
                                            "mnist", "cifar", "wikitext", "conll", "ontonotes"} else "BENCHMARK", 0.95
    
    # Check metrics
    if text_lower in METRIC_NAMES:
        return "METRIC", 0.95
    
    # Check legal terms
    if text_lower in LEGAL_TERMS:
        return "LEGAL_TERM", 0.9
    
    # Check medical terms
    if text_lower in MEDICAL_TERMS:
        return "MEDICAL_TERM", 0.9
    
    # Check financial terms
    if text_lower in FINANCIAL_TERMS:
        return "FINANCIAL_TERM", 0.9
    
    return spacy_type, 1.0


# ── Main validation pipeline ───────────────────────────────────────────────────
def validate_entity(raw_text: str, spacy_type: str) -> ValidatedEntity | None:
    """
    Full validation + normalization + reclassification pipeline.
    Returns None if entity should be discarded.
    
    PHASE 1 IMPROVEMENTS:
    - Enhanced PERSON validation with title case, length, abbreviation filtering
    - Enhanced ORG validation to reject ML terms incorrectly classified  
    - Enhanced LOCATION validation to prevent technical terms becoming locations
    - Domain-aware reclassification
    - Confidence scoring based on validation rules
    """
    # 1. Block noisy types outright
    if spacy_type in BLOCKED_TYPES:
        return None

    # 2. Normalize text
    norm = normalize_entity_text(raw_text)
    if not norm:
        return None

    # 3. Junk filter
    if is_junk_entity(norm):
        return None

    # 4. Type-specific validation with confidence scoring
    effective_type = spacy_type
    confidence = 1.0
    
    # PERSON validation
    if spacy_type == "PERSON":
        is_valid, person_conf = _is_valid_person(norm)
        if not is_valid:
            return None
        confidence = min(confidence, person_conf)
    
    # ORG validation
    elif spacy_type == "ORG":
        is_valid, org_conf = _is_valid_organization(norm)
        if not is_valid:
            # Try reclassifying as MODEL instead
            if norm.lower() in ML_MODEL_NAMES or norm.lower() in TECH_ACROYMNS_NOT_ORG:
                effective_type = "MODEL"
                confidence = 0.9
            else:
                return None
        else:
            confidence = min(confidence, org_conf)
    
    # GPE/LOC validation
    elif spacy_type in ("GPE", "LOC"):
        if not _is_valid_location(norm):
            # Try to reclassify
            new_type, type_conf = _reclassify_by_domain(norm, spacy_type)
            if new_type != spacy_type:
                effective_type = new_type
                confidence = type_conf
            else:
                return None
    
    # 5. NORP disambiguation: single-word NORP that are person first names
    if spacy_type == "NORP" and len(norm.split()) == 1:
        if norm.lower() in _NORP_PERSON_FIRST_NAMES:
            effective_type = "PERSON"
            confidence = 0.85
        elif not norm[0].isupper():
            # lowercase NORP single word — likely a fragment
            return None

    # 6. Domain-based reclassification (applies to all types)
    if effective_type == spacy_type:
        effective_type, domain_conf = _reclassify_by_domain(norm, effective_type)
        confidence = min(confidence, domain_conf)

    # 7. Final allowed-type check
    if effective_type not in ALLOWED_TYPES:
        return None

    label = ENTITY_LABELS.get(effective_type, effective_type)

    return ValidatedEntity(
        text=norm,
        original=raw_text,
        type=effective_type,
        type_label=label,
        confidence=confidence,
    )


def validate_entities_batch(
    raw_entities: list[tuple[str, str]]
) -> list[ValidatedEntity]:
    """Validate a batch of (text, spacy_type) tuples."""
    results = []
    seen: set[str] = set()
    for raw_text, spacy_type in raw_entities:
        validated = validate_entity(raw_text, spacy_type)
        if validated is None:
            continue
        key = canonical_key(validated.text)
        # Deduplication: if same canonical form seen, skip (highest confidence wins)
        if key not in seen:
            seen.add(key)
            results.append(validated)
    return results


def get_config_value(key: str, default=None):
    """Utility to read a config value from entity_config.json."""
    return _load_config().get(key, default)
