import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from main import build_parser, main
from retrieval.evidence import Claim, Evidence, SupportStatus
from retrieval.paper_analysis import PaperAnalysis
from retrieval.cross_paper_analysis import CrossPaperAnalysis, CorroboratingFinding


def test_cli_parser_structure():
    parser = build_parser()
    subparser_actions = [
        action for action in parser._actions
        if action.__class__.__name__ == "_SubParsersAction"
    ]
    assert len(subparser_actions) == 1
    choices = subparser_actions[0].choices
    assert "query" in choices
    assert "analyze" in choices
    assert "compare" in choices
    assert "evaluate" in choices
    assert "ingest" in choices


def test_cli_main_no_args(capsys):
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "usage: ragss" in captured.out


def test_cli_help_flag(capsys):
    exit_code = main(["--help"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "RAGSS: Research Paper Evidence Retrieval" in captured.out


def test_cli_argparse_syntax_error(capsys):
    """
    argparse convention: Unrecognized argument triggers parser error and returns exit code 2.
    """
    exit_code = main(["query", "test question", "--unknown-bogus-flag"])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "unrecognized arguments" in captured.err


def test_cli_query_empty_error(capsys):
    exit_code = main(["query", "   "])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Query text cannot be empty" in captured.err


def test_cli_query_empty_error_json(capsys):
    exit_code = main(["query", "   ", "--json"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Query text cannot be empty" in captured.err
    # stdout MUST remain parseable JSON
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert "Query text cannot be empty" in payload["error"]


def test_cli_query_invalid_k(capsys):
    exit_code = main(["query", "Valid query", "--k", "0"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Retrieval K must be a positive integer" in captured.err


def test_cli_query_missing_docs_error(capsys):
    exit_code = main(["query", "Valid query", "--docs", "non_existent_file.pdf"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Path does not exist" in captured.err


def test_cli_query_missing_docs_json(capsys):
    exit_code = main(["query", "Valid query", "--docs", "non_existent_file.pdf", "--json"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Path does not exist" in captured.err
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert "Path does not exist" in payload["error"]


def test_cli_query_no_llm_preview(capsys):
    mock_doc = MagicMock()
    mock_doc.page_content = "Self-attention mechanism replaces recurrence."
    mock_doc.metadata = {
        "document_id": "doc_1",
        "document_name": "paper.pdf",
        "page": 0,
    }

    mock_retriever_inst = MagicMock()
    mock_retriever_inst.retrieve.return_value = [mock_doc]

    with patch("retrieval.retriver.Retriever", return_value=mock_retriever_inst), \
         patch("retrieval.pipeline.BasicRetriever") as mock_pipeline:
        exit_code = main(["query", "What is attention?", "--no-llm", "--json"])
        assert exit_code == 0
        # BasicRetriever must NOT be instantiated when --no-llm is active
        mock_pipeline.assert_not_called()
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["mode"] == "retrieval_only"
        assert payload["results_count"] == 1
        assert payload["citations"][0]["document_id"] == "doc_1"
        assert payload["citations"][0]["page"] == 1


def test_cli_query_with_mock_pipeline(capsys):
    mock_pipeline_inst = MagicMock()
    mock_pipeline_inst.generate.return_value = {
        "answer": "Attention calculates scalar products.",
        "citations": [{"document_name": "paper.pdf", "page": 1, "source": "paper.pdf"}],
    }

    with patch("retrieval.retriver.Retriever"), \
         patch("retrieval.pipeline.BasicRetriever", return_value=mock_pipeline_inst):
        exit_code = main(["query", "How does attention work?", "--json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert "Attention calculates" in payload["answer"]
        assert len(payload["citations"]) == 1


def test_cli_analyze_validation_errors(capsys, tmp_path):
    # 1. Missing file
    code_missing = main(["analyze", str(tmp_path / "missing.pdf")])
    assert code_missing == 1
    assert "Path does not exist" in capsys.readouterr().err

    # 2. Non-PDF file
    text_file = tmp_path / "notes.txt"
    text_file.write_text("Not a pdf")
    code_non_pdf = main(["analyze", str(text_file)])
    assert code_non_pdf == 1
    assert "File is not a PDF" in capsys.readouterr().err


def test_cli_analyze_errors_json(capsys, tmp_path):
    code_missing = main(["analyze", str(tmp_path / "missing.pdf"), "--json"])
    assert code_missing == 1
    captured = capsys.readouterr()
    assert "Path does not exist" in captured.err
    payload = json.loads(captured.out)
    assert payload["status"] == "error"


def test_cli_analyze_with_mocks(capsys, tmp_path):
    sample_pdf = Path(__file__).resolve().parent / "sample_papers" / "paper1_transformers.pdf"
    output_file = tmp_path / "analysis.json"

    dummy_analysis = PaperAnalysis(
        document_id="doc_1",
        document_name="paper1_transformers.pdf",
        title="Attention Is All You Need",
        research_problem="Sequence modeling without recurrence",
        methodology="Transformer",
        major_claims="Replaces RNNs",
        evidence="28.4 BLEU",
        limitations="Quadratic memory",
        citations=[{"document_name": "paper1.pdf", "page": 1}],
        claims=[
            Claim(
                text="Replaces RNNs",
                support_status=SupportStatus.SUPPORTED,
                evidence=[],
            )
        ],
    )

    with patch("retrieval.retriver.Retriever"), \
         patch("retrieval.paper_analysis.analyze_papers", return_value=[dummy_analysis]):
        exit_code = main([
            "analyze",
            str(sample_pdf),
            "--json",
            "--output",
            str(output_file),
        ])
        assert exit_code == 0
        assert output_file.exists()
        saved_data = json.loads(output_file.read_text(encoding="utf-8"))
        assert len(saved_data) == 1
        assert saved_data[0]["title"] == "Attention Is All You Need"
        assert saved_data[0]["claims"][0]["support_status"] == "SUPPORTED"

        # Check stdout is also valid JSON status
        captured = capsys.readouterr()
        status_payload = json.loads(captured.out)
        assert status_payload["status"] == "success"
        assert status_payload["paper_count"] == 1


def test_cli_compare_validation_errors(capsys):
    sample_pdf = Path(__file__).resolve().parent / "sample_papers" / "paper1_transformers.pdf"
    # Single paper without --allow-single must exit 1
    exit_code = main(["compare", str(sample_pdf)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Compare requires at least 2 papers" in captured.err


def test_cli_compare_errors_json(capsys):
    sample_pdf = Path(__file__).resolve().parent / "sample_papers" / "paper1_transformers.pdf"
    exit_code = main(["compare", str(sample_pdf), "--json"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Compare requires at least 2 papers" in captured.err
    payload = json.loads(captured.out)
    assert payload["status"] == "error"


def test_cli_compare_with_mocks(capsys, tmp_path):
    sample_pdf1 = Path(__file__).resolve().parent / "sample_papers" / "paper1_transformers.pdf"
    sample_pdf2 = Path(__file__).resolve().parent / "sample_papers" / "paper2_bert.pdf"
    output_file = tmp_path / "compare.json"

    dummy_cross = CrossPaperAnalysis(
        similarities="Shared attention mechanisms.",
        differences="Bidirectional vs causal.",
        corroborating_findings="Both outperform RNNs.",
        conflicting_findings="None identified.",
        research_gaps="Context length.",
        overall_synthesis="Transformers dominant.",
        analyzed_papers=[{"document_name": "paper1.pdf"}, {"document_name": "paper2.pdf"}],
        corroborating_findings_structured=[
            CorroboratingFinding(
                finding="Both outperform RNNs.",
                supporting_papers=["paper1.pdf", "paper2.pdf"],
                evidence=[Evidence(evidence_text="E", document_id="d1", document_name="paper1.pdf", page=1)],
            )
        ],
    )

    with patch("retrieval.retriver.Retriever"), \
         patch("retrieval.paper_analysis.analyze_papers", return_value=[]), \
         patch("retrieval.cross_paper_analysis.analyze_cross_paper", return_value=dummy_cross):
        exit_code = main([
            "compare",
            str(sample_pdf1),
            str(sample_pdf2),
            "--json",
            "--output",
            str(output_file),
        ])
        assert exit_code == 0
        assert output_file.exists()
        saved = json.loads(output_file.read_text(encoding="utf-8"))
        assert len(saved["analyzed_papers"]) == 2
        assert len(saved["corroborating_findings"]) == 1
        assert saved["corroborating_findings"][0]["citations"] == ["[paper1.pdf, p.1]"]

        captured = capsys.readouterr()
        status_payload = json.loads(captured.out)
        assert status_payload["status"] == "success"


def test_cli_evaluate_fixture(tmp_path):
    out_json = tmp_path / "bench_results.json"
    exit_code = main([
        "evaluate",
        "--mode",
        "fixture",
        "--k",
        "3",
        "--output-json",
        str(out_json),
    ])
    assert exit_code == 0
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["mode"] == "fixture"
    assert data["mean_recall_at_3"] == 1.0


def test_cli_evaluate_json_stdout(capsys):
    exit_code = main(["evaluate", "--mode", "fixture", "--k", "3", "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "success"
    assert payload["mean_recall_at_3"] == 1.0


def test_cli_evaluate_missing_dataset(capsys):
    exit_code = main(["evaluate", "--dataset", "non_existent_dataset.jsonl"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Dataset file not found" in captured.err


def test_cli_ingest_with_mock(capsys):
    sample_pdf = Path(__file__).resolve().parent / "sample_papers" / "paper1_transformers.pdf"
    mock_retriever = MagicMock()
    mock_retriever.get_indexed_papers.return_value = [
        {"document_name": "paper1_transformers.pdf", "document_id": "doc_1_paper1_transformers"}
    ]

    with patch("retrieval.retriver.Retriever", return_value=mock_retriever):
        exit_code = main(["ingest", str(sample_pdf), "--json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["status"] == "success"
        assert payload["count"] == 1


def test_cli_unexpected_internal_error(capsys):
    with patch("retrieval.retriver.Retriever", side_effect=RuntimeError("Corrupt SQLite database")):
        exit_code = main(["query", "What is attention?", "--json"])
        assert exit_code == 2
        captured = capsys.readouterr()
        assert "Internal Error" in captured.err
        payload = json.loads(captured.out)
        assert payload["status"] == "internal_error"
        assert "Corrupt SQLite database" in payload["error"]


def test_cli_keyboard_interrupt(capsys):
    with patch("retrieval.retriver.Retriever", side_effect=KeyboardInterrupt):
        exit_code = main(["query", "What is attention?", "--json"])
        assert exit_code == 130
        captured = capsys.readouterr()
        assert "Operation cancelled" in captured.err
        payload = json.loads(captured.out)
        assert payload["status"] == "cancelled"
