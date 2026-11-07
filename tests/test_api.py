import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import app, sessions, SessionState


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health_and_config(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    cfg_res = client.get("/api/config")
    assert cfg_res.status_code == 200
    data = cfg_res.json()
    assert "known_limitations" in data
    assert len(data["known_limitations"]) >= 4
    assert any("Parent documents" in lim for lim in data["known_limitations"])
    assert any("lexical token-overlap" in lim for lim in data["known_limitations"])
    assert "chunk_size" in data


def test_upload_validation_errors(client):
    # 1. Non-PDF file
    res = client.post(
        "/api/upload",
        files=[("files", ("notes.txt", b"plain text", "text/plain"))],
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data["files"]) == 1
    assert data["files"][0]["valid"] is False
    assert "Only PDF" in data["files"][0]["error"]

    # 2. Empty PDF file (0 bytes)
    res = client.post(
        "/api/upload",
        files=[("files", ("empty.pdf", b"", "application/pdf"))],
    )
    assert res.status_code == 200
    data = res.json()
    assert data["files"][0]["valid"] is False
    assert "empty" in data["files"][0]["error"].lower()

    # 3. Corrupt PDF file (missing %PDF magic bytes)
    res = client.post(
        "/api/upload",
        files=[("files", ("corrupt.pdf", b"NOT_A_PDF_CONTENT", "application/pdf"))],
    )
    assert res.status_code == 200
    data = res.json()
    assert data["files"][0]["valid"] is False
    assert "%PDF" in data["files"][0]["error"]


def test_valid_pdf_upload_and_removal(client):
    valid_pdf_bytes = b"%PDF-1.4\n%Fake PDF binary stream\n%%EOF"
    res = client.post(
        "/api/upload",
        files=[
            ("files", ("paper1.pdf", valid_pdf_bytes, "application/pdf")),
            ("files", ("paper2.pdf", valid_pdf_bytes, "application/pdf")),
        ],
    )
    assert res.status_code == 200
    data = res.json()
    session_id = data["session_id"]
    assert len(data["files"]) == 2
    assert all(f["valid"] for f in data["files"])

    # Verify staged files listing
    staged_res = client.get(f"/api/uploaded?session_id={session_id}")
    assert staged_res.status_code == 200
    assert len(staged_res.json()["files"]) == 2

    # Remove one file
    del_res = client.delete(f"/api/upload/paper1.pdf?session_id={session_id}")
    assert del_res.status_code == 200
    assert len(del_res.json()["files"]) == 1
    assert del_res.json()["files"][0]["filename"] == "paper2.pdf"


def test_duplicate_submission_rejection(client):
    session = SessionState("test_dup_session")
    session.staged_files["test.pdf"] = PROJECT_ROOT / "test.pdf"
    session.processing = True
    sessions["test_dup_session"] = session

    res = client.post(
        "/api/analyze",
        json={"session_id": "test_dup_session", "k": 4, "no_llm": True},
    )
    assert res.status_code == 409
    assert "already in progress" in res.json()["detail"]


def test_analyze_empty_session_error(client):
    res = client.post(
        "/api/analyze",
        json={"session_id": "empty_session_xyz", "k": 4, "no_llm": True},
    )
    assert res.status_code == 400
    assert "No PDF files staged" in res.json()["detail"]


def test_analyze_single_pdf_offline(client):
    valid_pdf_bytes = b"%PDF-1.4\n%Fake PDF binary stream\n%%EOF"
    upload_res = client.post(
        "/api/upload",
        files=[("files", ("single_paper.pdf", valid_pdf_bytes, "application/pdf"))],
    )
    session_id = upload_res.json()["session_id"]

    # Mock Retriever so we don't load live SentenceTransformer weights during unit test
    mock_doc = Document(
        page_content="We present an empirical study on retrieval augmented generation for scientific literature.",
        metadata={
            "document_id": "doc_1_single",
            "document_name": "single_paper.pdf",
            "chunk_id": "doc_1_chunk_0",
            "page": 0,
            "source": "single_paper.pdf",
        },
    )

    class MockOfflineRetriever:
        parents = [mock_doc]
        parent_by_id = {"doc_1_chunk_0": mock_doc}

        def get_indexed_papers(self):
            return [{"document_id": "doc_1_single", "document_name": "single_paper.pdf"}]

        def retrieve_with_diagnostics(self, query, k=4, candidate_k=None, document_id=None, rrf_k=60):
            diag = {
                "query": query,
                "stages": {
                    "dense": {"ran": True, "retrieved_count": 1, "timing_ms": 1.2, "error": None},
                    "bm25": {"ran": True, "retrieved_count": 1, "timing_ms": 0.5, "error": None},
                    "rrf": {"ran": True, "fused_count": 1, "timing_ms": 0.2, "k": 60},
                    "cross_encoder": {"ran": True, "reranked_count": 1, "fallback": False, "timing_ms": 2.1, "error": None},
                    "parent_expansion": {"ran": True, "expanded_count": 1, "fallback_count": 0, "timing_ms": 0.1},
                },
                "filters": {"document_id": document_id, "k": k, "candidate_k": candidate_k, "rrf_k": rrf_k},
                "timings": {"dense_ms": 1.2, "bm25_ms": 0.5, "rrf_ms": 0.2, "rerank_ms": 2.1, "parent_expansion_ms": 0.1, "total_ms": 4.1},
                "abstention_reason": None,
                "chunks": [
                    {
                        "chunk_id": "doc_1_chunk_0",
                        "parent_id": None,
                        "document_id": "doc_1_single",
                        "document_name": "single_paper.pdf",
                        "source": "single_paper.pdf",
                        "page": 0,
                        "chunk_text": mock_doc.page_content,
                        "dense_rank": 1,
                        "dense_score": 0.92,
                        "bm25_rank": 1,
                        "bm25_score": 3.45,
                        "rrf_rank": 1,
                        "rrf_score": 0.032,
                        "cross_encoder_rank": 1,
                        "cross_encoder_score": 0.88,
                        "final_rank": 1,
                        "expanded": False,
                        "parent_content": None,
                        "metadata": mock_doc.metadata,
                    }
                ],
            }
            return [mock_doc], diag

        def retrieve(self, query, k=4, candidate_k=None, document_id=None, rrf_k=60):
            docs, _ = self.retrieve_with_diagnostics(query, k=k, candidate_k=candidate_k, document_id=document_id, rrf_k=rrf_k)
            return docs

    with patch("api.app.Retriever", return_value=MockOfflineRetriever()):
        res = client.post(
            "/api/analyze",
            json={"session_id": session_id, "k": 4, "no_llm": True},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["total_submitted"] == 1
        assert data["successful_count"] == 1
        assert len(data["papers"]) == 1
        # Cross-paper analysis must NOT run when only one paper is analyzed
        assert data["cross_analysis"] is None

        paper = data["papers"][0]
        assert paper["document_id"] == "doc_1_single"
        assert paper["document_name"] == "single_paper.pdf"
        assert paper["success"] is True
        assert paper["diagnostics"] is not None
        assert len(paper["diagnostics"]["chunks"]) == 1
        assert paper["diagnostics"]["chunks"][0]["chunk_id"] == "doc_1_chunk_0"
        assert paper["diagnostics"]["chunks"][0]["dense_score"] == 0.92


def test_analyze_multi_pdf_with_cross_analysis_offline(client):
    valid_pdf_bytes = b"%PDF-1.4\n%Fake PDF binary stream\n%%EOF"
    upload_res = client.post(
        "/api/upload",
        files=[
            ("files", ("paper_a.pdf", valid_pdf_bytes, "application/pdf")),
            ("files", ("paper_b.pdf", valid_pdf_bytes, "application/pdf")),
        ],
    )
    session_id = upload_res.json()["session_id"]

    doc_a = Document(
        page_content="Paper A explores transformer attention mechanisms.",
        metadata={"document_id": "doc_a", "document_name": "paper_a.pdf", "chunk_id": "chunk_a", "page": 0, "source": "paper_a.pdf"},
    )
    doc_b = Document(
        page_content="Paper B investigates recurrent neural networks.",
        metadata={"document_id": "doc_b", "document_name": "paper_b.pdf", "chunk_id": "chunk_b", "page": 0, "source": "paper_b.pdf"},
    )

    class MockMultiRetriever:
        parents = [doc_a, doc_b]

        def get_indexed_papers(self):
            return [
                {"document_id": "doc_a", "document_name": "paper_a.pdf"},
                {"document_id": "doc_b", "document_name": "paper_b.pdf"},
            ]

        def retrieve_with_diagnostics(self, query, k=4, candidate_k=None, document_id=None, rrf_k=60):
            doc = doc_a if document_id == "doc_a" else doc_b
            diag = {
                "query": query,
                "stages": {
                    "dense": {"ran": True, "retrieved_count": 1, "timing_ms": 1.0, "error": None},
                    "bm25": {"ran": True, "retrieved_count": 1, "timing_ms": 0.4, "error": None},
                    "rrf": {"ran": True, "fused_count": 1, "timing_ms": 0.2, "k": 60},
                    "cross_encoder": {"ran": False, "reranked_count": 0, "fallback": False, "timing_ms": 0.0, "error": None},
                    "parent_expansion": {"ran": True, "expanded_count": 0, "fallback_count": 1, "timing_ms": 0.1},
                },
                "filters": {"document_id": document_id, "k": k, "candidate_k": candidate_k, "rrf_k": rrf_k},
                "timings": {"dense_ms": 1.0, "bm25_ms": 0.4, "rrf_ms": 0.2, "rerank_ms": 0.0, "parent_expansion_ms": 0.1, "total_ms": 1.7},
                "abstention_reason": None,
                "chunks": [],
            }
            return [doc], diag

        def retrieve(self, query, k=4, candidate_k=None, document_id=None, rrf_k=60):
            docs, _ = self.retrieve_with_diagnostics(query, k=k, candidate_k=candidate_k, document_id=document_id, rrf_k=rrf_k)
            return docs

    with patch("api.app.Retriever", return_value=MockMultiRetriever()):
        res = client.post(
            "/api/analyze",
            json={"session_id": session_id, "k": 4, "no_llm": True},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["successful_count"] == 2
        assert len(data["papers"]) == 2

        # Cross-paper analysis must run when >= 2 papers succeed
        assert data["cross_analysis"] is not None
        ca = data["cross_analysis"]
        assert ca["success"] is True
        assert len(ca["analyzed_papers"]) == 2
        assert "SIMILARITIES" in ca["similarities"].upper() or len(ca["similarities"]) > 0


def test_cross_analyze_rejection_for_insufficient_papers(client):
    res = client.post(
        "/api/cross-analyze",
        json={"session_id": "single_paper_session", "no_llm": True},
    )
    assert res.status_code == 400
    assert "requires at least two" in res.json()["detail"].lower()


def test_query_endpoint_diagnostics(client):
    mock_doc = Document(
        page_content="Evidence excerpt on attention mechanisms.",
        metadata={
            "document_id": "doc_query",
            "document_name": "query_paper.pdf",
            "chunk_id": "chunk_q1",
            "page": 2,
            "source": "query_paper.pdf",
        },
    )

    class MockQueryRetriever:
        def retrieve_with_diagnostics(self, query, k=4, candidate_k=None, document_id=None, rrf_k=60):
            return [mock_doc], {
                "query": query,
                "stages": {
                    "dense": {"ran": True, "retrieved_count": 1, "timing_ms": 1.5, "error": None},
                    "bm25": {"ran": True, "retrieved_count": 1, "timing_ms": 0.8, "error": None},
                    "rrf": {"ran": True, "fused_count": 1, "timing_ms": 0.3, "k": 60},
                    "cross_encoder": {"ran": True, "reranked_count": 1, "fallback": False, "timing_ms": 2.5, "error": None},
                    "parent_expansion": {"ran": True, "expanded_count": 1, "fallback_count": 0, "timing_ms": 0.2},
                },
                "filters": {"document_id": document_id, "k": k, "candidate_k": candidate_k, "rrf_k": rrf_k},
                "timings": {"dense_ms": 1.5, "bm25_ms": 0.8, "rrf_ms": 0.3, "rerank_ms": 2.5, "parent_expansion_ms": 0.2, "total_ms": 5.3},
                "abstention_reason": None,
                "chunks": [
                    {
                        "chunk_id": "chunk_q1",
                        "parent_id": "parent_q1",
                        "document_id": "doc_query",
                        "document_name": "query_paper.pdf",
                        "source": "query_paper.pdf",
                        "page": 2,
                        "chunk_text": mock_doc.page_content,
                        "dense_rank": 1,
                        "dense_score": 0.85,
                        "bm25_rank": 1,
                        "bm25_score": 4.12,
                        "rrf_rank": 1,
                        "rrf_score": 0.033,
                        "cross_encoder_rank": 1,
                        "cross_encoder_score": 0.94,
                        "final_rank": 1,
                        "expanded": True,
                        "parent_content": "Full page parent content.",
                        "metadata": mock_doc.metadata,
                    }
                ],
            }

    sess = SessionState("test_query_sess")
    sess.retriever = MockQueryRetriever()
    sessions["test_query_sess"] = sess

    res = client.post(
        "/api/query",
        json={"query": "What is attention?", "session_id": "test_query_sess", "k": 4, "no_llm": True},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "What is attention?"
    assert len(data["citations"]) == 1
    assert data["citations"][0]["page"] == 3  # 0-indexed page 2 becomes display page 3
    assert data["diagnostics"]["timings"]["total_ms"] > 0
    assert len(data["diagnostics"]["chunks"]) == 1
    chunk = data["diagnostics"]["chunks"][0]
    assert chunk["chunk_id"] == "chunk_q1"
    assert chunk["cross_encoder_score"] == 0.94
    assert chunk["expanded"] is True


def test_partial_failure_handling(client):
    valid_pdf_bytes = b"%PDF-1.4\n%Fake PDF binary stream\n%%EOF"
    upload_res = client.post(
        "/api/upload",
        files=[
            ("files", ("good_paper.pdf", valid_pdf_bytes, "application/pdf")),
            ("files", ("bad_paper.pdf", valid_pdf_bytes, "application/pdf")),
        ],
    )
    session_id = upload_res.json()["session_id"]

    good_doc = Document(
        page_content="Good paper content.",
        metadata={"document_id": "doc_good", "document_name": "good_paper.pdf", "chunk_id": "chunk_g", "page": 0, "source": "good_paper.pdf"},
    )

    class MockPartialRetriever:
        parents = [good_doc]

        def get_indexed_papers(self):
            return [
                {"document_id": "doc_good", "document_name": "good_paper.pdf"},
                {"document_id": "doc_bad", "document_name": "bad_paper.pdf"},
            ]

        def retrieve_with_diagnostics(self, query, k=4, candidate_k=None, document_id=None, rrf_k=60):
            if document_id == "doc_bad":
                raise RuntimeError("Simulated corruption in bad_paper.pdf")
            return [good_doc], {
                "query": query,
                "stages": {
                    "dense": {"ran": True, "retrieved_count": 1, "timing_ms": 1.0, "error": None},
                    "bm25": {"ran": True, "retrieved_count": 1, "timing_ms": 0.4, "error": None},
                    "rrf": {"ran": True, "fused_count": 1, "timing_ms": 0.2, "k": 60},
                    "cross_encoder": {"ran": False, "reranked_count": 0, "fallback": False, "timing_ms": 0.0, "error": None},
                    "parent_expansion": {"ran": True, "expanded_count": 0, "fallback_count": 1, "timing_ms": 0.1},
                },
                "filters": {"document_id": document_id, "k": k, "candidate_k": candidate_k, "rrf_k": rrf_k},
                "timings": {"dense_ms": 1.0, "bm25_ms": 0.4, "rrf_ms": 0.2, "rerank_ms": 0.0, "parent_expansion_ms": 0.1, "total_ms": 1.7},
                "abstention_reason": None,
                "chunks": [],
            }

        def retrieve(self, query, k=4, candidate_k=None, document_id=None, rrf_k=60):
            docs, _ = self.retrieve_with_diagnostics(query, k=k, candidate_k=candidate_k, document_id=document_id, rrf_k=rrf_k)
            return docs

    with patch("api.app.Retriever", return_value=MockPartialRetriever()):
        res = client.post(
            "/api/analyze",
            json={"session_id": session_id, "k": 4, "no_llm": True},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["total_submitted"] == 2
        assert data["successful_count"] == 1
        assert data["failed_count"] == 1
        assert len(data["papers"]) == 2

        # Verify good paper succeeded and bad paper failed gracefully
        good = [p for p in data["papers"] if p["document_id"] == "doc_good"][0]
        bad = [p for p in data["papers"] if p["document_id"] == "doc_bad"][0]

        assert good["success"] is True
        assert bad["success"] is False
        assert "Simulated corruption" in bad["error"]

        # Only 1 successful paper, so cross-paper analysis should NOT run
        assert data["cross_analysis"] is None


def test_zero_retrieval_abstention_query(client):
    class MockEmptyQueryRetriever:
        def retrieve_with_diagnostics(self, query, k=4, candidate_k=None, document_id=None, rrf_k=60):
            return [], {
                "query": query,
                "stages": {
                    "dense": {"ran": True, "retrieved_count": 0, "timing_ms": 0.5, "error": None},
                    "bm25": {"ran": True, "retrieved_count": 0, "timing_ms": 0.2, "error": None},
                    "rrf": {"ran": True, "fused_count": 0, "timing_ms": 0.1, "k": 60},
                    "cross_encoder": {"ran": False, "reranked_count": 0, "fallback": False, "timing_ms": 0.0, "error": None},
                    "parent_expansion": {"ran": False, "expanded_count": 0, "fallback_count": 0, "timing_ms": 0.0},
                },
                "filters": {"document_id": document_id, "k": k, "candidate_k": candidate_k, "rrf_k": rrf_k},
                "timings": {"dense_ms": 0.5, "bm25_ms": 0.2, "rrf_ms": 0.1, "rerank_ms": 0.0, "parent_expansion_ms": 0.0, "total_ms": 0.8},
                "abstention_reason": "No matching chunks found by dense or lexical search.",
                "chunks": [],
            }

    sess = SessionState("test_abstain_sess")
    sess.retriever = MockEmptyQueryRetriever()
    sessions["test_abstain_sess"] = sess

    res = client.post(
        "/api/query",
        json={"query": "Quantum gravity speedups in transformers?", "session_id": "test_abstain_sess", "k": 4, "no_llm": False},
    )
    assert res.status_code == 200
    data = res.json()
    assert "sufficient evidence" in data["answer"].lower()
    assert data["citations"] == []
    assert data["diagnostics"]["abstention_reason"] is not None


def test_spa_serving(client):
    from api.app import frontend_dist
    if frontend_dist.exists() and (frontend_dist / "index.html").exists():
        res = client.get("/")
        assert res.status_code == 200
        assert "PaperTrail" in res.text


