"""
Phase 8: Automated Topic Extraction Validation
Tests across 5 domain types: research, finance, legal, medical, technical.

Run with:
    cd /home/swyra/Desktop/doc_analytics
    uv run python tests/test_topic_extractor.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.core.topic_extractor import (
    extract_topics,
    _run_lda,
    _run_nmf,
    _run_tfidf_fallback,
    _deduplicate_topics,
    _jaccard,
    _name_topic,
    _tokenize_document,
    _compute_coherence,
)

import numpy as np

# ── Sample documents per domain ───────────────────────────────────────────────
RESEARCH_PARAS = [
    "The Transformer architecture relies entirely on attention mechanisms, dispensing with recurrence and convolutions.",
    "Multi-head self-attention allows the model to attend to different representation subspaces at different positions.",
    "The encoder maps an input sequence of symbol representations to a sequence of continuous representations.",
    "The decoder generates an output sequence of symbols one element at a time, attending to the encoder output.",
    "We train our models on the WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs.",
    "Training on 8 NVIDIA P100 GPUs with the base model taking 0.4 seconds per training step, totaling 100K steps.",
    "BLEU score improvements of 2 points over previous state-of-the-art models were achieved on the WMT benchmark.",
    "Positional encodings use sine and cosine functions to inject position information into the model.",
    "Layer normalization is applied before each sub-layer, and residual connections are used around each sub-layer.",
    "The feed-forward networks consist of two linear transformations with a ReLU activation in between.",
    "Dropout is applied to the output of each sub-layer, before it is added to the sub-layer input and normalized.",
    "The self-attention mechanism enables each position to attend to all positions in the previous decoder layer.",
    "Cross-attention layers allow the decoder to attend over the encoder output, integrating context information.",
    "We used the Adam optimizer with a warmup schedule for the learning rate over the first training steps.",
    "Model performance on machine translation was evaluated using tokenized BLEU score on the newstest2014 dataset.",
]

FINANCE_PARAS = [
    "The company's EBITDA margin improved by 3.2 percentage points year-over-year, reaching 24.5% in Q3 2023.",
    "Revenue growth of 18% was driven primarily by strong performance in the enterprise software segment.",
    "Operating cash flow increased to $2.4 billion, supporting continued capital allocation towards share buybacks.",
    "The board approved a dividend increase of 15 cents per share, reflecting confidence in the balance sheet.",
    "Credit risk exposure was reduced through portfolio diversification across investment-grade corporate bonds.",
    "Interest rate sensitivity analysis shows the portfolio would decline 3.2% for each 100 basis point increase.",
    "Loan default rates in the consumer lending segment remained below 1.5%, well within historical norms.",
    "The P/E ratio of 22x reflects premium valuation relative to industry peers at 18x forward earnings.",
    "ROI on capital expenditures averaged 14.2% across the past three fiscal years, exceeding the hurdle rate.",
    "Leverage ratios improved as net debt declined from $8.1B to $6.4B through disciplined debt repayment.",
    "Market risk was hedged using interest rate swaps and foreign currency forward contracts.",
    "Earnings per share of $4.52 exceeded analyst consensus estimates of $4.31 by approximately 5%.",
]

MEDICAL_PARAS = [
    "Patients with HbA1c levels above 7.5% were randomized to receive intensive glucose-lowering therapy.",
    "LDL cholesterol reduction of 45% was achieved in the treatment arm versus 12% in the placebo group.",
    "The clinical trial enrolled 1,240 patients with Type 2 diabetes across 18 academic medical centers.",
    "MRI imaging revealed significant reduction in hippocampal atrophy in patients receiving the intervention.",
    "Blood glucose monitoring was performed using continuous glucose monitoring devices at 5-minute intervals.",
    "Adverse events including nausea and headache were reported in 18% of treated patients versus 9% controls.",
    "Insulin sensitivity improved significantly after 12 weeks of lifestyle modification combined with metformin.",
    "WBC count normalization occurred within 4 weeks in 78% of patients responding to the immunotherapy protocol.",
    "RBC transfusion requirements decreased by 40% following erythropoietin-stimulating agent administration.",
    "CT scan findings correlated with surgical outcomes in 91% of cases across the retrospective cohort.",
    "Randomized controlled trials provide the highest level of evidence for evaluating treatment efficacy.",
    "Patient outcomes were assessed at 6-month and 12-month follow-up intervals using validated quality-of-life scales.",
]

LEGAL_PARAS = [
    "Article 12 of this Agreement sets forth the indemnification obligations of each Party to the other.",
    "Section 15 requires the licensor to provide 30 days written notice before terminating the license grant.",
    "The governing law of this contract shall be the laws of the State of Delaware, excluding conflict of law rules.",
    "Intellectual property rights in all deliverables shall vest exclusively in the Client upon full payment.",
    "The arbitration clause in Clause 4 mandates binding arbitration before the American Arbitration Association.",
    "Force majeure events shall excuse performance obligations for the duration of the qualifying event.",
    "Confidentiality obligations survive termination of the Agreement for a period of five years thereafter.",
    "Due diligence review of the target company's financial statements revealed no material misrepresentations.",
    "The non-compete covenant restricts the party from engaging in competing business activities for 24 months.",
    "Breach of contract claims require demonstration of a valid agreement, breach, causation, and damages.",
    "Representations and warranties are made as of the Effective Date and survive closing for 18 months.",
]

TECHNICAL_PARAS = [
    "The REST API exposes endpoints secured with OAuth 2.0 bearer tokens issued by the identity provider.",
    "JWT tokens are signed using RS256 and include claims for user identity, roles, and expiration time.",
    "Docker containers are orchestrated by Kubernetes, enabling horizontal scaling and rolling deployments.",
    "The GraphQL schema defines types for User, Product, and Order, with resolvers backed by PostgreSQL.",
    "Microservice architecture decouples the authentication service from the business logic layer.",
    "Redis caching reduces database query latency from 45ms to 3ms for frequently accessed records.",
    "CI/CD pipelines execute unit tests, integration tests, and security scans on every pull request.",
    "Infrastructure as Code using Terraform provisions cloud resources reproducibly across environments.",
    "The message queue using Apache Kafka decouples event producers from downstream consumers.",
    "Load balancing across three availability zones ensures 99.95% uptime SLA for the production API.",
    "Database migrations are managed through version-controlled schema files applied by Flyway.",
    "TLS 1.3 is enforced for all service-to-service communication within the internal network.",
]


def make_fake_doc(doc_id: str, doc_name: str, paragraphs: list[str]) -> dict:
    return {
        "doc_id": doc_id,
        "document_name": doc_name,
        "full_text": " ".join(paragraphs),
        "pages": [
            {
                "page": i + 1,
                "text": para,
                "paragraphs": [para],
            }
            for i, para in enumerate(paragraphs)
        ],
    }


# ── Helper assertions ─────────────────────────────────────────────────────────
def assert_topics_valid(topics: list, domain: str) -> None:
    assert topics is not None, f"[{domain}] Topics is None"
    assert len(topics) > 0, f"[{domain}] No topics generated"
    for t in topics:
        assert t["topic"], f"[{domain}] Empty topic name in {t}"
        assert t["keywords"], f"[{domain}] Empty keywords in {t}"
        assert len(t["keywords"]) >= 3, f"[{domain}] Fewer than 3 keywords in {t}"
        assert 0.0 <= t.get("coherence", 0.0) <= 1.0, f"[{domain}] coherence out of range: {t}"
        assert 0.0 <= t.get("topic_weight", 0.0) <= 1.0, f"[{domain}] topic_weight out of range: {t}"


def assert_no_duplicates(topics: list, domain: str) -> None:
    names = [t["topic"] for t in topics]
    assert len(names) == len(set(names)), f"[{domain}] Duplicate topic names: {names}"


def assert_no_crashes(fn, *args, label: str = "") -> None:
    try:
        result = fn(*args)
        assert result is not None or True  # just must not throw
    except Exception as e:
        raise AssertionError(f"[{label}] Unexpected crash: {e}")


# ── Test 1: Research paper ────────────────────────────────────────────────────
def test_research_paper():
    doc = make_fake_doc("test_research", "Attention Is All You Need", RESEARCH_PARAS)
    result = extract_topics(doc)
    topics = result["topics"]

    assert_topics_valid(topics, "Research")
    assert_no_duplicates(topics, "Research")
    assert result["method_used"] in ("LDA", "NMF", "keyword-cluster")

    # Expect attention/transformer to appear
    all_kws = " ".join(" ".join(t["keywords"]) for t in topics).lower()
    assert any(w in all_kws for w in ["attention", "transformer", "encoder", "decoder"]), \
        f"Research paper should mention attention/transformer. Keywords: {all_kws}"

    print(f"  ✅ Research: {len(topics)} topics via {result['method_used']}")
    for t in topics:
        print(f"     [{t['coherence']:.3f} / {t['topic_weight']:.3f}] {t['topic']} → {t['keywords'][:4]}")


# ── Test 2: Finance report ────────────────────────────────────────────────────
def test_finance_report():
    doc = make_fake_doc("test_finance", "Q3 2023 Earnings Report", FINANCE_PARAS)
    result = extract_topics(doc)
    topics = result["topics"]

    assert_topics_valid(topics, "Finance")
    assert_no_duplicates(topics, "Finance")

    all_kws = " ".join(" ".join(t["keywords"]) for t in topics).lower()
    assert any(w in all_kws for w in ["revenue", "credit", "market", "ebitda", "roi", "risk", "cash"]), \
        f"Finance doc should mention financial terms. Keywords: {all_kws}"

    print(f"  ✅ Finance: {len(topics)} topics via {result['method_used']}")
    for t in topics:
        print(f"     [{t['coherence']:.3f} / {t['topic_weight']:.3f}] {t['topic']} → {t['keywords'][:4]}")


# ── Test 3: Medical report ────────────────────────────────────────────────────
def test_medical_report():
    doc = make_fake_doc("test_medical", "Clinical Trial Report", MEDICAL_PARAS)
    result = extract_topics(doc)
    topics = result["topics"]

    assert_topics_valid(topics, "Medical")
    assert_no_duplicates(topics, "Medical")

    all_kws = " ".join(" ".join(t["keywords"]) for t in topics).lower()
    assert any(w in all_kws for w in ["patient", "clinical", "glucose", "insulin", "treatment", "trial"]), \
        f"Medical doc should mention clinical terms. Keywords: {all_kws}"

    print(f"  ✅ Medical: {len(topics)} topics via {result['method_used']}")
    for t in topics:
        print(f"     [{t['coherence']:.3f} / {t['topic_weight']:.3f}] {t['topic']} → {t['keywords'][:4]}")


# ── Test 4: Legal contract ────────────────────────────────────────────────────
def test_legal_contract():
    doc = make_fake_doc("test_legal", "Software License Agreement", LEGAL_PARAS)
    result = extract_topics(doc)
    topics = result["topics"]

    assert_topics_valid(topics, "Legal")
    assert_no_duplicates(topics, "Legal")

    print(f"  ✅ Legal: {len(topics)} topics via {result['method_used']}")
    for t in topics:
        print(f"     [{t['coherence']:.3f} / {t['topic_weight']:.3f}] {t['topic']} → {t['keywords'][:4]}")


# ── Test 5: Technical documentation ──────────────────────────────────────────
def test_technical_doc():
    doc = make_fake_doc("test_technical", "API Architecture Guide", TECHNICAL_PARAS)
    result = extract_topics(doc)
    topics = result["topics"]

    assert_topics_valid(topics, "Technical")
    assert_no_duplicates(topics, "Technical")

    all_kws = " ".join(" ".join(t["keywords"]) for t in topics).lower()
    assert any(w in all_kws for w in ["api", "jwt", "oauth", "kubernetes", "docker", "service", "microservice"]), \
        f"Technical doc should mention infra terms. Keywords: {all_kws}"

    print(f"  ✅ Technical: {len(topics)} topics via {result['method_used']}")
    for t in topics:
        print(f"     [{t['coherence']:.3f} / {t['topic_weight']:.3f}] {t['topic']} → {t['keywords'][:4]}")


# ── Test 6: Tokenizer correctness ─────────────────────────────────────────────
def test_tokenizer():
    text = "GPT-4 and BERT outperform SVM on NLP tasks. HbA1c > 7.5% indicates poor glucose control. P/E ratio 22x."
    tokens = _tokenize_document(text)
    token_set = set(t.lower() for t in tokens)

    # These should be preserved
    assert "gpt-4" in token_set, f"GPT-4 not found in {token_set}"
    assert "bert" in token_set or "BERT".lower() in token_set
    assert "hba1c" in token_set or "HbA1c".lower() in token_set
    assert "svm" in token_set
    assert "p/e" in token_set or "nlp" in token_set

    print(f"  ✅ Tokenizer: preserved domain tokens {[t for t in tokens if len(t) >= 2]}")


# ── Test 7: Jaccard deduplication ────────────────────────────────────────────
def test_jaccard_dedup():
    from backend.core.topic_extractor import _jaccard, _deduplicate_topics

    a = {"attention", "transformer", "encoder"}
    b = {"transformer", "attention", "decoder"}
    j = _jaccard(a, b)
    # intersection=2, union=4, jaccard=0.5
    assert abs(j - 0.5) < 1e-6, f"Expected jaccard=0.5, got {j}"

    topics = [
        {"topic": "A", "keywords": ["attention", "transformer", "encoder", "layer"], "coherence": 0.9, "topic_weight": 0.8, "document_count": 5},
        {"topic": "B", "keywords": ["transformer", "attention", "decoder", "layer"], "coherence": 0.8, "topic_weight": 0.7, "document_count": 4},
        {"topic": "C", "keywords": ["medical", "clinical", "patient", "trial"], "coherence": 0.7, "topic_weight": 0.6, "document_count": 3},
    ]
    deduped = _deduplicate_topics(topics)
    # A and B have jaccard(top8)=3/5=0.6 >= 0.5 → B should be removed
    assert len(deduped) == 2, f"Expected 2 after dedup, got {len(deduped)}: {[t['topic'] for t in deduped]}"
    assert deduped[0]["topic"] == "A"
    assert deduped[1]["topic"] == "C"
    print(f"  ✅ Jaccard dedup: {len(topics)} → {len(deduped)} topics (B removed as duplicate of A)")


# ── Test 8: Topic naming ─────────────────────────────────────────────────────
def test_topic_naming():
    cases = [
        (["attention", "transformer", "encoder"], "Transformer Architecture"),
        (["credit", "risk", "loan", "default"], "Credit Risk"),
        (["blood", "glucose", "insulin", "diabetes"], "Diabetes Management"),
        (["encoder", "decoder", "sequence", "layer"], "Encoder-Decoder Architecture"),
        (["neural", "network", "recurrent", "hidden"], "Neural Networks"),
    ]
    for keywords, expected in cases:
        result = _name_topic(keywords)
        assert result == expected, f"Expected '{expected}', got '{result}' for {keywords}"
        print(f"  ✅ Topic name: {keywords[:2]} → '{result}'")


# ── Test 9: Coherence range ──────────────────────────────────────────────────
def test_coherence_range():
    component = np.array([0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01])
    top_indices = np.array([0, 1, 2, 3])
    score = _compute_coherence(component, top_indices)
    assert 0.0 <= score <= 1.0, f"Coherence out of range: {score}"
    # 0.95 of mass is in top 4 — coherence should be ~0.95
    assert score > 0.9, f"Expected high coherence ~0.95, got {score}"
    print(f"  ✅ Coherence: top-4 mass proportion = {score}")


# ── Runner ────────────────────────────────────────────────────────────────────
TESTS = [
    ("Tokenizer Correctness",      test_tokenizer),
    ("Jaccard Deduplication",      test_jaccard_dedup),
    ("Topic Naming Determinism",   test_topic_naming),
    ("Coherence Range",            test_coherence_range),
    ("Research Paper",             test_research_paper),
    ("Finance Report",             test_finance_report),
    ("Medical Report",             test_medical_report),
    ("Legal Contract",             test_legal_contract),
    ("Technical Documentation",    test_technical_doc),
]

if __name__ == "__main__":
    passed = 0
    failed = 0
    for name, fn in TESTS:
        print(f"\n{'─'*60}")
        print(f"TEST: {name}")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  💥 ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{passed+failed} passed  |  {failed} failed")
    sys.exit(0 if failed == 0 else 1)
