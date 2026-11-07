"""
Deterministic unit tests for Evidence provenance, citations, and Claim structures.
Tests structural provenance separately from semantic support.
"""

import pytest
from langchain_core.documents import Document

from retrieval.evidence import (
    Claim,
    ConflictingFinding,
    CorroboratingFinding,
    Evidence,
    EvidenceAggregator,
    ResearchGap,
    SupportStatus,
    build_citation,
    build_citations,
)


def test_evidence_from_document_preserves_metadata():
    doc = Document(
        page_content="The Transformer relies entirely on an attention mechanism.",
        metadata={
            "document_id": "doc_1_paper1_transformers",
            "document_name": "paper1_transformers.pdf",
            "page": 0,  # 0-indexed in raw metadata
            "chunk_id": "doc_1_paper1_transformers_p0_c0",
            "parent_id": "doc_1_paper1_transformers_p0",
        },
    )

    ev = Evidence.from_document(doc, support_status=SupportStatus.SUPPORTED)

    assert ev.document_id == "doc_1_paper1_transformers"
    assert ev.document_name == "paper1_transformers.pdf"
    assert ev.page == 1  # Converted to 1-indexed for display
    assert ev.chunk_id == "doc_1_paper1_transformers_p0_c0"
    assert ev.parent_id == "doc_1_paper1_transformers_p0"
    assert ev.support_status == SupportStatus.SUPPORTED
    assert "Transformer relies" in ev.evidence_text


def test_build_citation_formatting_and_rejection():
    ev = Evidence(
        evidence_text="Sample text",
        document_id="doc_bert",
        document_name="paper2_bert.pdf",
        page=2,
    )

    # Bracket style
    assert build_citation(ev, style="bracket") == "[paper2_bert.pdf, p.2]"
    # Inline style
    assert build_citation(ev, style="inline") == "paper2_bert.pdf · p.2"

    # Must reject non-Evidence inputs (prevent hallucinated citations)
    with pytest.raises(TypeError):
        build_citation("paper2_bert.pdf: p.2")  # type: ignore


def test_build_citations_deduplication():
    ev1 = Evidence(evidence_text="Text 1", document_id="doc1", document_name="paper1.pdf", page=1)
    ev2 = Evidence(evidence_text="Text 2", document_id="doc1", document_name="paper1.pdf", page=1)  # Duplicate source/page
    ev3 = Evidence(evidence_text="Text 3", document_id="doc2", document_name="paper2.pdf", page=3)

    citations = build_citations([ev1, ev2, ev3])
    assert len(citations) == 2
    assert citations[0] == "[paper1.pdf, p.1]"
    assert citations[1] == "[paper2.pdf, p.3]"


def test_claim_structure_and_citations():
    ev = Evidence(evidence_text="Attention maps queries to keys", document_id="doc1", document_name="paper1.pdf", page=2)
    claim = Claim(
        text="Attention computes compatibility scores.",
        evidence=[ev],
        support_status=SupportStatus.SUPPORTED,
    )

    assert claim.text == "Attention computes compatibility scores."
    assert len(claim.get_citations()) == 1
    assert claim.get_citations()[0] == "[paper1.pdf, p.2]"


def test_evidence_aggregator_deduplication_and_grouping():
    ev1 = Evidence(evidence_text="A", document_id="doc1", document_name="p1.pdf", page=1, chunk_id="c1")
    ev2 = Evidence(evidence_text="A", document_id="doc1", document_name="p1.pdf", page=1, chunk_id="c1")  # Duplicate
    ev3 = Evidence(evidence_text="B", document_id="doc2", document_name="p2.pdf", page=2, chunk_id="c2")

    aggregator = EvidenceAggregator([ev1, ev2, ev3])
    assert len(aggregator.get_all_evidence()) == 3

    deduped = aggregator.deduplicate()
    assert len(deduped) == 2
    assert [e.chunk_id for e in deduped] == ["c1", "c2"]

    grouped = aggregator.group_by_document(by="document_name")
    assert "p1.pdf" in grouped
    assert "p2.pdf" in grouped
    assert len(grouped["p1.pdf"]) == 1


def test_cross_paper_finding_structures():
    ev1 = Evidence(evidence_text="T1", document_id="d1", document_name="p1.pdf", page=1)
    ev2 = Evidence(evidence_text="T2", document_id="d2", document_name="p2.pdf", page=1)

    corr = CorroboratingFinding(
        finding="Both architectures avoid recurrence.",
        supporting_papers=["p1.pdf", "p2.pdf"],
        evidence=[ev1, ev2],
    )
    assert len(corr.get_citations()) == 2
    assert corr.supporting_papers == ["p1.pdf", "p2.pdf"]

    conf = ConflictingFinding(
        topic="Pre-training speed",
        papers=["p1.pdf", "p2.pdf"],
        explanation="P1 reports 2x speedup while P2 reports 1.2x.",
        evidence=[ev1, ev2],
    )
    assert conf.topic == "Pre-training speed"

    gap = ResearchGap(
        finding="Neither paper explores long-context retrieval > 32k tokens.",
        relevant_papers=["p1.pdf", "p2.pdf"],
        evidence=[ev1, ev2],
    )
    assert len(gap.relevant_papers) == 2


def test_claim_default_status_is_not_enough_evidence():
    claim = Claim(text="Default claim without evidence")
    assert claim.support_status == SupportStatus.NOT_ENOUGH_EVIDENCE
    assert claim.evidence == []
    assert claim.get_citations() == []


def test_claim_verifier_token_overlap_semantics():
    from retrieval.evidence import ClaimVerifier

    ev1 = Evidence(
        evidence_text="Self-attention mechanisms replace recurrent layers, allowing significant parallelization.",
        document_id="doc1",
        document_name="transformers.pdf",
        page=1,
    )
    ev2 = Evidence(
        evidence_text="Adam optimizer is configured with beta1 0.9 and beta2 0.98.",
        document_id="doc1",
        document_name="transformers.pdf",
        page=3,
    )

    verifier = ClaimVerifier(support_threshold=0.5, partial_threshold=0.25)

    # 1. High overlap -> SUPPORTED
    claim_supported = "Self-attention mechanisms replace recurrent layers."
    status, matched_ev = verifier.verify(claim_supported, [ev1, ev2])
    assert status == SupportStatus.SUPPORTED
    assert len(matched_ev) == 1
    assert matched_ev[0].document_name == "transformers.pdf"
    assert matched_ev[0].page == 1

    # 2. Moderate/partial overlap -> PARTIALLY_SUPPORTED
    claim_partial = "Self-attention mechanisms require massive compute cluster clusters."
    status, matched_ev = verifier.verify(claim_partial, [ev1, ev2])
    assert status == SupportStatus.PARTIALLY_SUPPORTED
    assert len(matched_ev) == 1

    # 3. Zero overlap -> UNSUPPORTED
    claim_unsupported = "Convolutional filters are applied across depth dimensions for visual features."
    status, matched_ev = verifier.verify(claim_unsupported, [ev1, ev2])
    assert status == SupportStatus.UNSUPPORTED
    assert len(matched_ev) == 0

    # 4. Empty candidate evidence -> NOT_ENOUGH_EVIDENCE
    status, matched_ev = verifier.verify("Any claim text", [])
    assert status == SupportStatus.NOT_ENOUGH_EVIDENCE
    assert len(matched_ev) == 0


def test_claim_verifier_custom_function():
    from retrieval.evidence import ClaimVerifier

    ev = Evidence(
        evidence_text="Sample text",
        document_id="doc1",
        document_name="paper.pdf",
        page=1,
    )

    # Custom oracle verifier
    def mock_nli(claim_text: str, evidence: Evidence) -> SupportStatus:
        if "entail" in claim_text:
            return SupportStatus.SUPPORTED
        return SupportStatus.UNSUPPORTED

    verifier = ClaimVerifier(verifier_fn=mock_nli)
    status_yes, ev_yes = verifier.verify("This statement entails evidence", [ev])
    assert status_yes == SupportStatus.SUPPORTED
    assert len(ev_yes) == 1

    status_no, ev_no = verifier.verify("This statement contradicts", [ev])
    assert status_no == SupportStatus.UNSUPPORTED
    assert len(ev_no) == 0


def test_claim_verifier_threshold_validation():
    from retrieval.evidence import ClaimVerifier

    # partial_threshold > support_threshold must raise ValueError
    with pytest.raises(ValueError, match="cannot exceed"):
        ClaimVerifier(support_threshold=0.3, partial_threshold=0.6)

    # support_threshold > 1.0 must raise ValueError
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        ClaimVerifier(support_threshold=1.5, partial_threshold=0.5)

    # partial_threshold < 0.0 must raise ValueError
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        ClaimVerifier(support_threshold=0.5, partial_threshold=-0.1)


def test_adversarial_negation_limitation_documented():
    """
    DOCUMENTED HEURISTIC LIMITATION:
    Token overlap fails to detect polarity/negation.
    Evidence directly negates the claim, but because the content words match almost 100%,
    the lexical heuristic incorrectly flags the claim as SUPPORTED.
    This test establishes and documents this known limitation.
    """
    from retrieval.evidence import ClaimVerifier

    ev = Evidence(
        evidence_text="Transformers do not use recurrent layers for sequence modeling.",
        document_id="doc1",
        document_name="transformers.pdf",
        page=1,
    )
    claim_negated = "Transformers use recurrent layers for sequence modeling."

    verifier = ClaimVerifier()
    status, matched = verifier.verify(claim_negated, [ev])
    # The heuristic marks this SUPPORTED because content words match: {"transformers", "use", "recurrent", "layers", "sequence", "modeling"}
    assert status == SupportStatus.SUPPORTED
    assert len(matched) == 1

    # In contrast, an NLI / semantic verifier function correctly detects contradiction:
    def semantic_negation_verifier(claim: str, evidence: Evidence) -> SupportStatus:
        if " not " in evidence.evidence_text and " not " not in claim:
            return SupportStatus.UNSUPPORTED
        return SupportStatus.SUPPORTED

    semantic_verifier = ClaimVerifier(verifier_fn=semantic_negation_verifier)
    sem_status, sem_matched = semantic_verifier.verify(claim_negated, [ev])
    assert sem_status == SupportStatus.UNSUPPORTED
    assert len(sem_matched) == 0


def test_adversarial_numerical_mismatch_limitation_documented():
    """
    DOCUMENTED HEURISTIC LIMITATION:
    Token overlap cannot verify numerical equality.
    Evidence reports 62.1% but claim asserts 95.4%.
    The non-numerical content words match strongly, so lexical overlap flags it as supported.
    """
    from retrieval.evidence import ClaimVerifier

    ev = Evidence(
        evidence_text="The proposed model achieves a test accuracy of 62.1 percent on the benchmark dataset.",
        document_id="doc1",
        document_name="benchmark.pdf",
        page=4,
    )
    claim_wrong_num = "The proposed model achieves a test accuracy of 95.4 percent on the benchmark dataset."

    verifier = ClaimVerifier()
    status, matched = verifier.verify(claim_wrong_num, [ev])
    # Documents that token overlap cannot distinguish factual numbers
    assert status == SupportStatus.SUPPORTED
    assert len(matched) == 1


def test_adversarial_entity_substitution_limitation_documented():
    """
    DOCUMENTED HEURISTIC LIMITATION:
    When the surrounding template words are identical, substituting the primary entity
    (ResNet vs Transformer) still yields high overlap ratio.
    """
    from retrieval.evidence import ClaimVerifier

    ev = Evidence(
        evidence_text="Transformer achieves state of the art results on large scale machine translation tasks.",
        document_id="doc1",
        document_name="translation.pdf",
        page=2,
    )
    claim_wrong_entity = "ResNet achieves state of the art results on large scale machine translation tasks."

    verifier = ClaimVerifier(support_threshold=0.8, partial_threshold=0.4)
    # Total content words in claim: 8. Matched: 7. Ratio = 7/8 = 0.875
    ratio, _ = verifier.compute_overlap(claim_wrong_entity, ev.evidence_text)
    assert ratio >= 0.8
    # Documents that surface overlap alone cannot verify named entity correctness
    status, _ = verifier.verify(claim_wrong_entity, [ev])
    assert status == SupportStatus.SUPPORTED


def test_adversarial_genuine_paraphrase_limitation_documented():
    """
    DOCUMENTED HEURISTIC LIMITATION:
    A valid semantic paraphrase using distinct vocabulary produces low lexical overlap,
    causing the heuristic to falsely mark the claim as UNSUPPORTED.
    """
    from retrieval.evidence import ClaimVerifier

    ev = Evidence(
        evidence_text="Processing longer inputs demands execution duration proportional to the square of token count.",
        document_id="doc1",
        document_name="complexity.pdf",
        page=3,
    )
    claim_paraphrase = "The network computational cost escalates quadratically with sequence length."

    verifier = ClaimVerifier(support_threshold=0.5, partial_threshold=0.25)
    status, matched = verifier.verify(claim_paraphrase, [ev])
    # Almost zero content word overlap despite semantic equivalence
    assert status == SupportStatus.UNSUPPORTED
    assert len(matched) == 0


def test_empty_candidate_evidence_vs_unsupported_evidence():
    from retrieval.evidence import ClaimVerifier

    verifier = ClaimVerifier()

    # 1. Truly empty candidate list -> NOT_ENOUGH_EVIDENCE
    status_empty, ev_empty = verifier.verify("Any claim", [])
    assert status_empty == SupportStatus.NOT_ENOUGH_EVIDENCE
    assert ev_empty == []

    # 2. Non-empty candidate list with irrelevant content -> UNSUPPORTED
    unrelated_ev = Evidence(
        evidence_text="Chocolate chip cookies require butter, sugar, flour, and baking soda.",
        document_id="doc_recipes",
        document_name="recipes.pdf",
        page=1,
    )
    status_unsupported, ev_unsupported = verifier.verify("Transformers utilize multi-head self-attention.", [unrelated_ev])
    assert status_unsupported == SupportStatus.UNSUPPORTED
    assert ev_unsupported == []


def test_filter_supporting_evidence_with_verifier_fn():
    from retrieval.evidence import ClaimVerifier

    ev1 = Evidence(evidence_text="Text 1", document_id="d1", document_name="p1.pdf", page=1)
    ev2 = Evidence(evidence_text="Text 2", document_id="d2", document_name="p2.pdf", page=2)

    def selective_verifier(text: str, ev: Evidence) -> SupportStatus:
        if ev.document_id == "d1":
            return SupportStatus.SUPPORTED
        return SupportStatus.UNSUPPORTED

    verifier = ClaimVerifier(verifier_fn=selective_verifier)
    filtered = verifier.filter_supporting_evidence("Some query", [ev1, ev2])
    assert len(filtered) == 1
    assert filtered[0].document_id == "d1"
