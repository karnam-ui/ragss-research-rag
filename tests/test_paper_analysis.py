import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeListLLM
from retrieval.evidence import Evidence, SupportStatus
from retrieval.paper_analysis import PaperAnalysis, PaperAnalyzer, analyze_papers
from retrieval.retriver import Retriever


def test_paper_analyzer_clean_abstention_offline():
    class EmptyMockRetriever:
        parents = []

        def retrieve(self, query, k=4, document_id=None):
            return []

    # If the LLM is invoked during clean abstention, FakeListLLM will raise IndexError because responses is empty!
    fake_llm = FakeListLLM(responses=[])
    analyzer = PaperAnalyzer(retriever=EmptyMockRetriever(), llm=fake_llm)

    analysis = analyzer.analyze_paper(document_id="doc_missing", document_name="missing.pdf")
    assert isinstance(analysis, PaperAnalysis)
    assert analysis.document_id == "doc_missing"
    assert analysis.document_name == "missing.pdf"
    assert analysis.research_problem == "Not available in the provided evidence."
    assert analysis.methodology == "Not available in the provided evidence."
    assert analysis.major_claims == "Not available in the provided evidence."
    assert analysis.evidence == "Not available in the provided evidence."
    assert analysis.limitations == "Not available in the provided evidence."
    assert analysis.citations == []
    assert analysis.source_evidence == []
    assert analysis.claims == []


def test_paper_analyzer_grounded_claims_offline():
    class MockPaperRetriever:
        parents = []

        def retrieve(self, query, k=4, document_id=None):
            return [
                Document(
                    page_content="We introduce the Transformer, a model architecture relying entirely on an attention mechanism.",
                    metadata={
                        "document_id": "doc_1_transformer",
                        "document_name": "transformer.pdf",
                        "page": 0,
                        "source": "transformer.pdf",
                    },
                )
            ]

    llm_output = (
        "TITLE: Attention Is All You Need\n"
        "RESEARCH PROBLEM: Recurrent sequence models cannot be parallelized.\n"
        "METHODOLOGY: Transformer architecture based entirely on attention mechanisms.\n"
        "MAJOR CLAIMS:\n"
        "- Transformer model relies entirely on an attention mechanism.\n"
        "- Transformer achieves sub-linear quantum speedups on NP-hard graphs.\n"
        "EVIDENCE: State-of-the-art translation quality on WMT 2014.\n"
        "LIMITATIONS: Quadratic complexity with sequence length.\n"
    )
    fake_llm = FakeListLLM(responses=[llm_output])
    analyzer = PaperAnalyzer(retriever=MockPaperRetriever(), llm=fake_llm)

    analysis = analyzer.analyze_paper(document_id="doc_1_transformer", document_name="transformer.pdf")
    assert analysis.title == "Attention Is All You Need"
    assert len(analysis.citations) == 1
    assert analysis.citations[0]["page"] == 1

    assert len(analysis.claims) == 2
    # Claim 1 matches the evidence
    claim1 = analysis.claims[0]
    assert "Transformer model relies" in claim1.text
    assert claim1.support_status == SupportStatus.SUPPORTED
    assert len(claim1.evidence) == 1
    assert claim1.get_citations() == ["[transformer.pdf, p.1]"]

    # Claim 2 has zero overlap with the evidence
    claim2 = analysis.claims[1]
    assert "quantum speedups" in claim2.text
    assert claim2.support_status == SupportStatus.UNSUPPORTED
    assert len(claim2.evidence) == 0
    assert claim2.get_citations() == []


def test_paper_analysis_claim_evidence_isolation_invariant():
    """
    Ensures that when a paper has multiple evidence chunks, each generated claim
    is associated ONLY with the specific chunk that corroborates it, NOT the entire pool.
    """
    from retrieval.evidence import ClaimVerifier

    class MultiChunkRetriever:
        parents = []

        def retrieve(self, query, k=4, document_id=None):
            return [
                Document(
                    page_content="On WMT 2014 English-to-German translation, the model achieves 28.4 BLEU.",
                    metadata={"document_id": "doc1", "document_name": "paper.pdf", "page": 7, "source": "paper.pdf"},
                ),
                Document(
                    page_content="The encoder consists of a stack of 6 identical layers with multi-head self-attention.",
                    metadata={"document_id": "doc1", "document_name": "paper.pdf", "page": 2, "source": "paper.pdf"},
                ),
            ]

    llm_output = (
        "TITLE: Transformer Paper\n"
        "RESEARCH PROBLEM: N/A\n"
        "METHODOLOGY: N/A\n"
        "MAJOR CLAIMS:\n"
        "- Model achieves 28.4 BLEU on English to German translation.\n"
        "- The encoder consists of a stack of 6 identical layers.\n"
        "- Model trains using backpropagation across photonic circuits.\n"
        "EVIDENCE: N/A\n"
        "LIMITATIONS: N/A\n"
    )

    fake_llm = FakeListLLM(responses=[llm_output])
    analyzer = PaperAnalyzer(retriever=MultiChunkRetriever(), llm=fake_llm)

    analysis = analyzer.analyze_paper(document_id="doc1", document_name="paper.pdf")
    assert len(analysis.claims) == 3

    # Claim 1 (BLEU score) -> attached ONLY chunk 1 (page 8, display)
    c1 = analysis.claims[0]
    assert "28.4 BLEU" in c1.text
    assert c1.support_status == SupportStatus.SUPPORTED
    assert len(c1.evidence) == 1
    assert c1.evidence[0].page == 8  # 0-indexed page 7 + 1

    # Claim 2 (6 identical layers) -> attached ONLY chunk 2 (page 3, display)
    c2 = analysis.claims[1]
    assert "6 identical layers" in c2.text
    assert c2.support_status == SupportStatus.SUPPORTED
    assert len(c2.evidence) == 1
    assert c2.evidence[0].page == 3  # 0-indexed page 2 + 1

    # Claim 3 (photonic circuits) -> uncorroborated, empty evidence attached
    c3 = analysis.claims[2]
    assert "photonic" in c3.text
    assert c3.support_status == SupportStatus.UNSUPPORTED
    assert len(c3.evidence) == 0


def test_paper_analysis_custom_verifier_fn():
    """
    Verifies that a custom verifier callable injected into PaperAnalyzer is genuinely invoked.
    """
    from retrieval.evidence import ClaimVerifier

    class SimpleRetriever:
        parents = []

        def retrieve(self, query, k=4, document_id=None):
            return [
                Document(
                    page_content="Model is trained without recurrence.",
                    metadata={"document_id": "doc1", "document_name": "paper.pdf", "page": 0, "source": "paper.pdf"},
                )
            ]

    llm_output = (
        "TITLE: Test\n"
        "RESEARCH PROBLEM: N/A\n"
        "METHODOLOGY: N/A\n"
        "MAJOR CLAIMS:\n"
        "- Model is trained with recurrence.\n"
        "EVIDENCE: N/A\n"
        "LIMITATIONS: N/A\n"
    )

    def strict_nli(claim: str, ev: Evidence) -> SupportStatus:
        # Detects contradiction where surface tokens match
        if "without recurrence" in ev.evidence_text and "with recurrence" in claim and "without" not in claim:
            return SupportStatus.UNSUPPORTED
        return SupportStatus.SUPPORTED

    fake_llm = FakeListLLM(responses=[llm_output])
    custom_verifier = ClaimVerifier(verifier_fn=strict_nli)
    analyzer = PaperAnalyzer(retriever=SimpleRetriever(), llm=fake_llm, verifier=custom_verifier)

    analysis = analyzer.analyze_paper(document_id="doc1", document_name="paper.pdf")
    assert len(analysis.claims) == 1
    # Custom NLI correctly flagged this as UNSUPPORTED despite identical lexical tokens
    assert analysis.claims[0].support_status == SupportStatus.UNSUPPORTED
    assert len(analysis.claims[0].evidence) == 0


@pytest.mark.integration
def test_paper_analysis():
    sample_dir = PROJECT_ROOT / "tests" / "sample_papers"
    sample_files = sorted(list(sample_dir.glob("*.pdf")))
    assert len(sample_files) >= 3, "Expected at least 3 sample PDF papers."

    file_paths = [str(f) for f in sample_files]
    print(f"Testing paper analysis on {len(file_paths)} papers:")
    for p in file_paths:
        print(f" - {Path(p).name}")

    # Use a separate test Chroma directory to avoid interfering with production/dev storage
    test_db_dir = PROJECT_ROOT / "tests" / "test_paper_analysis_db"
    retriever = Retriever(
        file_paths=file_paths,
        persist_directory=str(test_db_dir),
        collection_name="test_paper_analysis_collection",
        rebuild_index=True,
    )

    print("\nRunning analyze_papers()...")
    analyses = analyze_papers(file_paths=file_paths, retriever=retriever)

    # 1. Verify all 3 papers received an analysis
    assert len(analyses) == len(file_paths), (
        f"Expected {len(file_paths)} analyses, got {len(analyses)}"
    )
    print(f"\n✓ Generated {len(analyses)} paper analyses.")

    expected_doc_names = {Path(p).name for p in file_paths}
    analyzed_doc_names = set()

    for idx, analysis in enumerate(analyses, 1):
        assert isinstance(analysis, PaperAnalysis), "Output must be instance of PaperAnalysis"

        # 2. Verify correct document_name and document_id
        assert analysis.document_name in expected_doc_names, (
            f"Unexpected document_name: {analysis.document_name}"
        )
        assert analysis.document_id.startswith("doc_"), (
            f"Invalid document_id format: {analysis.document_id}"
        )
        analyzed_doc_names.add(analysis.document_name)

        # 3. Verify evidence and citations do not mix across papers
        for citation in analysis.citations:
            assert citation["document_id"] == analysis.document_id, (
                f"Citation document_id mismatch: {citation['document_id']} != {analysis.document_id}"
            )
            assert citation["document_name"] == analysis.document_name, (
                f"Citation document_name mismatch: {citation['document_name']} != {analysis.document_name}"
            )
            assert "page" in citation, "Citation must contain page number"
            assert "source" in citation, "Citation must contain source path"

        # 4. Verify all structured fields are present and non-empty
        assert analysis.title, "Title field cannot be empty"
        assert analysis.research_problem, "Research problem field cannot be empty"
        assert analysis.methodology, "Methodology field cannot be empty"
        assert analysis.major_claims, "Major claims field cannot be empty"
        assert analysis.evidence, "Evidence field cannot be empty"
        assert analysis.limitations, "Limitations field cannot be empty"

        print(f"\n========================================================")
        print(f"Paper Analysis [{idx}/{len(analyses)}]: {analysis.document_name}")
        print(f"Document ID:      {analysis.document_id}")
        print(f"Title:            {analysis.title}")
        print(f"Research Problem: {analysis.research_problem}")
        print(f"Methodology:      {analysis.methodology}")
        print(f"Major Claims:     {analysis.major_claims}")
        print(f"Evidence:         {analysis.evidence}")
        print(f"Limitations:      {analysis.limitations}")
        print(f"Citations:        {analysis.citations}")
        print(f"========================================================")

    assert analyzed_doc_names == expected_doc_names, (
        f"Missing analyses for: {expected_doc_names - analyzed_doc_names}"
    )
    print("\n✓ All assertions passed: Isolated paper analyses generated successfully with complete metadata.")


if __name__ == "__main__":
    test_paper_analysis()
