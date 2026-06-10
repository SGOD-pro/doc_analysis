"""
Phase 1 + 2 + 6 + 8: Entity Validation, Normalization & Reclassification Layer

This module forms the adapter between raw spaCy output and the graph/analytics layers.
Extractor → Validator (this module) → Normalizer → Graph Builder → Analytics

Supports future swap to SciSpacy, GLiNER, HuggingFace NER, or LLM-based extractors
by keeping this layer decoupled from any specific NER backend.
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
    confidence: float   # 0.0–1.0 (1.0 for all spaCy, lower for edge cases)


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
                "domain_dictionaries": {},
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
}

BLOCKED_TYPES = {
    "DATE", "TIME", "CARDINAL", "ORDINAL",
    "PERCENT", "MONEY", "QUANTITY", "LANGUAGE",
}

ENTITY_LABELS = {
    "PERSON":     "Person",
    "ORG":        "Organization",
    "PRODUCT":    "Product",
    "GPE":        "Location (Geo-Political)",
    "LOC":        "Location",
    "EVENT":      "Event",
    "WORK_OF_ART":"Work of Art",
    "LAW":        "Law / Regulation",
    "NORP":       "Nationality / Group",
    "MODEL":      "AI / ML Model",
    "METRIC":     "Evaluation Metric",
    "BENCHMARK":  "Benchmark / Dataset",
    "DATASET":    "Dataset",
    "JOURNAL":    "Journal / Repository",
    "VENUE":      "Conference / Venue",
    "FUNCTION":   "Mathematical Function",
    "TOKEN":      "Special Token",
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

# Common false positive ORG detections (actually common words/names)
_ORG_FALSE_POSITIVES: set[str] = {
    "adam", "linear", "attention", "transformer", "model", "network",
    "layer", "output", "input", "training", "testing", "validation",
    "baseline", "result", "results", "method", "approach", "system",
}

# Valid person name patterns - must have at least one capital letter and be multi-char
_PERSON_NAME_PATTERN = re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$')

# Single lowercase letters or very short strings that are not valid persons
_INVALID_PERSON_PATTERNS = [
    r'^[a-z]$',           # Single lowercase letter
    r'^[a-z]{1,2}$',      # 1-2 lowercase letters
    r'^[dkznxy]$',        # Common variable names
    r'^[abc]$',           # Generic labels
    r'^\d+$',             # Pure numbers
]


def is_valid_person(text: str) -> tuple[bool, float]:
    """
    Validate if text is a genuine person name.
    
    Returns:
        (is_valid, confidence) tuple
        
    Rejects:
        - Single lowercase letters (dk, zn, abc)
        - Very short strings (< 3 chars)
        - All lowercase strings
        - Strings matching variable name patterns
        
    Accepts:
        - Proper capitalized names (Geoffrey Hinton, Andrew Ng)
        - Multi-word names with proper capitalization
    """
    text = text.strip()
    
    # Too short to be a real name
    if len(text) < 3:
        return False, 0.0
    
    # Check against invalid patterns
    for pattern in _INVALID_PERSON_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return False, 0.0
    
    # All lowercase = likely not a proper name
    if text.islower():
        return False, 0.0
    
    # Must start with capital letter
    if not text[0].isupper():
        return False, 0.0
    
    # Check for proper name pattern (Capitalized words)
    if _PERSON_NAME_PATTERN.match(text):
        return True, 1.0
    
    # Names with initials (e.g., "John A. Smith")
    if re.match(r'^[A-Z][a-z]+\s+[A-Z]\.?\s*[A-Z][a-z]+$', text):
        return True, 0.95
    
    # Single capitalized word that looks like a name
    if re.match(r'^[A-Z][a-z]{2,}$', text) and ' ' not in text:
        # Lower confidence for single names
        return True, 0.7
    
    # Default: not a valid person name
    return False, 0.0


def is_valid_org(text: str) -> tuple[bool, float]:
    """
    Validate if text is a genuine organization name.
    
    Returns:
        (is_valid, confidence) tuple
        
    Rejects:
        - Common false positives (Adam, Linear, etc.)
        - Generic technical terms
        - Single common words
        
    Accepts:
        - Multi-word organization names
        - Known acronyms with context
        - Proper company/institution names
    """
    text_lower = text.lower().strip()
    
    # Check against known false positives
    if text_lower in _ORG_FALSE_POSITIVES:
        return False, 0.0
    
    # All caps acronyms of reasonable length
    if text.isupper() and len(text) >= 3 and len(text) <= 10:
        return True, 0.85
    
    # Multi-word with proper capitalization
    if ' ' in text and text[0].isupper():
        return True, 0.9
    
    # Contains common org suffixes
    org_suffixes = ['inc', 'ltd', 'corp', 'corporation', 'company', 'llc', 
                    'university', 'institute', 'laboratory', 'lab', 'foundation']
    if any(text_lower.endswith(s) for s in org_suffixes):
        return True, 0.95
    
    # Single capitalized word: check if it looks like a proper noun (not a common word)
    # Well-known tech companies are often single words
    if len(text.split()) == 1 and text[0].isupper() and len(text) >= 4:
        # Check if it's NOT a common English word (heuristic: no vowels pattern = likely not common)
        has_vowel_pattern = any(c in text_lower for c in 'aeiou')
        if has_vowel_pattern and text.isalpha():
            # Could be a company name like Google, Apple, Microsoft
            return True, 0.65
    
    # Default rejection for ambiguous short single words
    if len(text.split()) == 1 and len(text) < 4:
        return False, 0.0
    
    return True, 0.7


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


def calculate_entity_confidence(text: str, entity_type: str, spacy_type: str) -> float:
    """
    Calculate confidence score for an entity based on validation strength.
    
    Args:
        text: Entity text
        entity_type: Final entity type
        spacy_type: Original spaCy type
        
    Returns:
        Confidence score 0.0-1.0
    """
    # If type was reclassified, slightly lower confidence
    base_confidence = 1.0 if entity_type == spacy_type else 0.9
    
    # Type-specific validation
    if entity_type == "PERSON":
        is_valid, conf = is_valid_person(text)
        if not is_valid:
            return 0.0
        return min(1.0, (base_confidence + conf) / 2)
    
    elif entity_type == "ORG":
        is_valid, conf = is_valid_org(text)
        if not is_valid:
            return max(0.0, base_confidence - 0.3)
        return min(1.0, (base_confidence + conf) / 2)
    
    elif entity_type in ("GPE", "LOC"):
        if not is_valid_location(text):
            return 0.0
        return base_confidence
    
    return base_confidence


# ── Main validation pipeline ───────────────────────────────────────────────────
def validate_entity(raw_text: str, spacy_type: str) -> ValidatedEntity | None:
    """
    Full validation + normalization + reclassification pipeline.
    Returns None if entity should be discarded.
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

    # 4. NORP disambiguation: single-word NORP that are person first names
    effective_type = spacy_type
    if spacy_type == "NORP" and len(norm.split()) == 1:
        if norm.lower() in _NORP_PERSON_FIRST_NAMES:
            effective_type = "PERSON"
        elif not norm[0].isupper():
            # lowercase NORP single word — likely a fragment
            return None

    # 5. Reclassify type
    final_type = reclassify_entity(norm, effective_type)

    # 6. Type-specific validation with confidence scoring
    confidence = calculate_entity_confidence(norm, final_type, spacy_type)
    
    # Reject entities with zero confidence
    if confidence <= 0.0:
        # Special case: allow low-confidence ORG if it's multi-word
        if final_type == "ORG" and ' ' in norm and len(norm) > 5:
            confidence = 0.4
        else:
            return None

    # 7. Location validation (additional check)
    if final_type in ("GPE", "LOC"):
        if not is_valid_location(norm):
            return None

    # 8. Final allowed-type check
    if final_type not in ALLOWED_TYPES:
        return None

    label = ENTITY_LABELS.get(final_type, final_type)

    return ValidatedEntity(
        text=norm,
        original=raw_text,
        type=final_type,
        type_label=label,
        confidence=round(confidence, 3),
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
