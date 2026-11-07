import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from langchain_core.language_models.fake import FakeListLLM
from retrieval.cross_paper_analysis import CrossPaperAnalysis, CrossPaperAnalyzer, analyze_cross_paper
from retrieval.evidence import Evidence, SupportStatus
from retrieval.paper_analysis import PaperAnalysis, analyze_papers
from retrieval.retriver import Retriever


def test_cross_paper_analyzer_empty_and_single_offline():
    fake_llm = FakeListLLM(responses=[])
    analyzer = CrossPaperAnalyzer(llm=fake_llm)

    # 1. Zero papers
    res_empty = analyzer.analyze([])
    assert res_empty.analyzed_papers == []
    assert "No papers provided" in res_empty.similarities
    assert res_empty.source_evidence == []

    # 2. Single paper (no LLM call should be made)
    ev = Evidence(
        evidence_text="Sample limitation",
        document_id="doc1",
        document_name="p1.pdf",
        page=1,
    )
    single_analysis = PaperAnalysis(
        document_id="doc1",
        document_name="p1.pdf",
        title="Single Paper",
        research_problem="Problem",
        methodology="Method",
        major_claims="Claims",
        evidence="Evidence text",
        limitations="Quadratic computational complexity with context length.",
        source_evidence=[ev],
    )
    res_single = analyzer.analyze([single_analysis])
    assert len(res_single.analyzed_papers) == 1
    assert "Only one paper provided" in res_single.similarities
    assert len(res_single.research_gaps_structured) == 1
    assert res_single.research_gaps_structured[0].relevant_papers == ["p1.pdf"]
    assert res_single.research_gaps_structured[0].get_citations() == ["[p1.pdf, p.1]"]


def test_cross_paper_analyzer_multi_paper_grounding_offline():
    ev1 = Evidence(
        evidence_text="Transformers outperform recurrent models across language benchmarks.",
        document_id="doc_1_bert",
        document_name="bert.pdf",
        page=2,
    )
    ev2 = Evidence(
        evidence_text="Transformers demonstrate state of the art results over recurrent architectures.",
        document_id="doc_2_gpt",
        document_name="gpt.pdf",
        page=4,
    )

    paper1 = PaperAnalysis(
        document_id="doc_1_bert",
        document_name="bert.pdf",
        title="BERT Pre-training",
        research_problem="Bidirectional representations",
        methodology="Masked language modeling",
        major_claims="Bidirectional pre-training yields superior representations.",
        evidence="Outperformed all models on GLUE.",
        limitations="Pre-training is computationally demanding.",
        source_evidence=[ev1],
    )
    paper2 = PaperAnalysis(
        document_id="doc_2_gpt",
        document_name="gpt.pdf",
        title="GPT Pre-training",
        research_problem="Generative language modeling",
        methodology="Autoregressive causal modeling",
        major_claims="Generative pre-training transfers effectively.",
        evidence="State of the art on multiple zero-shot tasks.",
        limitations="Unidirectional attention bottleneck.",
        source_evidence=[ev2],
    )

    llm_output = (
        "SIMILARITIES:\nBoth papers leverage transformer architectures for language representations.\n\n"
        "DIFFERENCES:\nBERT uses masked bidirectional pre-training while GPT uses autoregressive modeling.\n\n"
        "CORROBORATING FINDINGS:\n"
        "- Both bert.pdf and gpt.pdf establish that transformers outperform recurrent models.\n\n"
        "CONFLICTING FINDINGS:\n"
        "- Bidirectional vs causal objective superiority in bert.pdf and gpt.pdf.\n\n"
        "RESEARCH GAPS:\n"
        "- Neither bert.pdf nor gpt.pdf addresses context lengths beyond 512 tokens.\n\n"
        "OVERALL SYNTHESIS:\nTransformer pre-training is the foundational paradigm for NLP."
    )

    fake_llm = FakeListLLM(responses=[llm_output])
    analyzer = CrossPaperAnalyzer(llm=fake_llm)

    cross = analyzer.analyze([paper1, paper2])
    assert len(cross.analyzed_papers) == 2
    assert "BERT uses masked" in cross.differences

    # Corroborating findings
    assert len(cross.corroborating_findings_structured) == 1
    corr = cross.corroborating_findings_structured[0]
    assert "transformers outperform recurrent" in corr.finding
    assert "bert.pdf" in corr.supporting_papers
    assert "gpt.pdf" in corr.supporting_papers
    assert len(corr.get_citations()) >= 1

    # Conflicting findings
    assert len(cross.conflicting_findings_structured) == 1
    conf = cross.conflicting_findings_structured[0]
    assert "bert.pdf" in conf.papers
    assert "gpt.pdf" in conf.papers

    # Research gaps
    assert len(cross.research_gaps_structured) == 1
    gap = cross.research_gaps_structured[0]
    assert "512 tokens" in gap.finding


def test_cross_paper_evidence_isolation_invariant():
    """
    CRITICAL REGRESSION TEST:
    Ensures that each finding receives ONLY evidence selected specifically for that finding,
    rather than inheriting the entire paper's evidence pool (preventing evidence pollution).
    Also confirms uncorroborated findings receive empty evidence lists.
    """
    ev1_adam = Evidence(
        evidence_text="We train using the Adam optimizer with beta1 0.9 and beta2 0.98.",
        document_id="doc1",
        document_name="paper1.pdf",
        page=5,
    )
    ev1_att = Evidence(
        evidence_text="Multi-head attention maps queries and keys across 8 parallel heads.",
        document_id="doc1",
        document_name="paper1.pdf",
        page=2,
    )
    ev2_adam = Evidence(
        evidence_text="Adam optimizer learning rate was warmed up linearly over 10000 steps.",
        document_id="doc2",
        document_name="paper2.pdf",
        page=6,
    )
    ev2_tok = Evidence(
        evidence_text="WordPiece tokenization vocabulary contains 30000 token types.",
        document_id="doc2",
        document_name="paper2.pdf",
        page=3,
    )

    p1 = PaperAnalysis(
        document_id="doc1",
        document_name="paper1.pdf",
        title="Paper 1",
        research_problem="",
        methodology="",
        major_claims="",
        evidence="",
        limitations="",
        source_evidence=[ev1_adam, ev1_att],
    )
    p2 = PaperAnalysis(
        document_id="doc2",
        document_name="paper2.pdf",
        title="Paper 2",
        research_problem="",
        methodology="",
        major_claims="",
        evidence="",
        limitations="",
        source_evidence=[ev2_adam, ev2_tok],
    )

    llm_output = (
        "SIMILARITIES:\nShared optimization algorithms.\n\n"
        "DIFFERENCES:\nDifferent attention models.\n\n"
        "CORROBORATING FINDINGS:\n"
        "- Both paper1 and paper2 utilize the Adam optimizer for model training.\n"
        "- Both paper1 and paper2 train quantum error correcting circuits on trapped ions.\n\n"
        "CONFLICTING FINDINGS:\nNone identified from the provided evidence.\n\n"
        "RESEARCH GAPS:\nNone identified from the provided evidence.\n\n"
        "OVERALL SYNTHESIS:\nSummary."
    )

    fake_llm = FakeListLLM(responses=[llm_output])
    analyzer = CrossPaperAnalyzer(llm=fake_llm)
    result = analyzer.analyze([p1, p2])

    assert len(result.corroborating_findings_structured) == 2

    # Finding 1 (Adam optimizer) must attach ONLY ev1_adam and ev2_adam
    finding_adam = result.corroborating_findings_structured[0]
    assert "Adam optimizer" in finding_adam.finding
    adam_pages = {ev.page for ev in finding_adam.evidence}
    assert adam_pages == {5, 6}  # ev1_adam (p.5) and ev2_adam (p.6)
    # MUST NOT contain attention (p.2) or tokenization (p.3)
    assert 2 not in adam_pages
    assert 3 not in adam_pages

    # Finding 2 (Quantum circuits - hallucinated / uncorroborated by text)
    # MUST receive EMPTY evidence (0 chunks) rather than dumping all 4 chunks
    finding_quantum = result.corroborating_findings_structured[1]
    assert "quantum" in finding_quantum.finding
    assert len(finding_quantum.evidence) == 0
    assert finding_quantum.get_citations() == []


@pytest.mark.integration
def test_cross_paper_pipeline():
    sample_dir = PROJECT_ROOT / "tests" / "sample_papers"
    sample_files = sorted(list(sample_dir.glob("*.pdf")))
    assert len(sample_files) >= 3, "Expected at least 3 sample PDF papers."

    file_paths = [str(f) for f in sample_files]
    print(f"Testing Cross-Paper Analysis across {len(file_paths)} papers:")
    for p in file_paths:
        print(f" - {Path(p).name}")

    # Step 1: Generate individual PaperAnalysis objects
    print("\n--- Step 1: Paper-Level Analysis ---")
    test_db_dir = PROJECT_ROOT / "tests" / "test_cross_paper_db"
    retriever = Retriever(
        file_paths=file_paths,
        persist_directory=str(test_db_dir),
        collection_name="test_cross_paper_collection",
        rebuild_index=True,
    )
    paper_analyses = analyze_papers(file_paths=file_paths, retriever=retriever)
    assert len(paper_analyses) == 3, f"Expected 3 paper analyses, got {len(paper_analyses)}"
    print(f"✓ Generated {len(paper_analyses)} paper analyses.")

    # Step 2: Cross-Paper Analysis across all 3 papers
    print("\n--- Step 2: Multi-Paper Synthesis ---")
    analyzer = CrossPaperAnalyzer()
    cross_result = analyzer.analyze(paper_analyses)

    assert isinstance(cross_result, CrossPaperAnalysis), "Output must be an instance of CrossPaperAnalysis"
    assert len(cross_result.analyzed_papers) == 3, "All 3 papers must be listed in analyzed_papers"

    expected_doc_names = {Path(p).name for p in file_paths}
    actual_doc_names = {p["document_name"] for p in cross_result.analyzed_papers}
    assert actual_doc_names == expected_doc_names, (
        f"Analyzed papers mismatch: {actual_doc_names} != {expected_doc_names}"
    )

    # Check that all sections are populated
    assert cross_result.similarities, "Similarities section cannot be empty"
    assert cross_result.differences, "Differences section cannot be empty"
    assert cross_result.corroborating_findings, "Corroborating findings section cannot be empty"
    assert cross_result.conflicting_findings, "Conflicting findings section cannot be empty"
    assert cross_result.research_gaps, "Research gaps section cannot be empty"
    assert cross_result.overall_synthesis, "Overall synthesis section cannot be empty"

    print("\n========================================================")
    print("CROSS-PAPER RESEARCH SYNTHESIS")
    print("========================================================")
    print(f"Analyzed Papers ({len(cross_result.analyzed_papers)}):")
    for p in cross_result.analyzed_papers:
        print(f" - {p['document_name']} ({p['document_id']}): {p['title']}")
    print(f"\n[1] SIMILARITIES:\n{cross_result.similarities}\n")
    print(f"[2] DIFFERENCES:\n{cross_result.differences}\n")
    print(f"[3] CORROBORATING FINDINGS:\n{cross_result.corroborating_findings}\n")
    print(f"[4] CONFLICTING FINDINGS:\n{cross_result.conflicting_findings}\n")
    print(f"[5] RESEARCH GAPS:\n{cross_result.research_gaps}\n")
    print(f"[6] OVERALL SYNTHESIS:\n{cross_result.overall_synthesis}\n")
    print("========================================================")
    print("✓ Full 3-paper cross-paper analysis verified successfully!")

    # Step 3: Verify pipeline adaptability with different number of papers (e.g. 2 papers)
    print("\n--- Step 3: Verifying Adaptability with 2 Papers ---")
    two_papers = paper_analyses[:2]
    subset_result = analyze_cross_paper(two_papers)
    assert len(subset_result.analyzed_papers) == 2
    assert subset_result.similarities
    assert subset_result.differences
    print("✓ 2-paper subset synthesis succeeded!")

    # Step 4: Verify single-paper edge case
    print("\n--- Step 4: Verifying 1-Paper Edge Case ---")
    one_paper = [paper_analyses[0]]
    single_result = analyze_cross_paper(one_paper)
    assert len(single_result.analyzed_papers) == 1
    assert "Only one paper" in single_result.similarities
    print("✓ Single-paper handling verified successfully!")


if __name__ == "__main__":
    test_cross_paper_pipeline()
