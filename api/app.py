import logging
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableLambda

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    CitationItem,
    ClaimItem,
    ConfigInfoResponse,
    ConflictingFindingItem,
    CorroboratingFindingItem,
    CrossAnalyzeRequest,
    CrossPaperAnalysisItem,
    EvidenceItem,
    PaperAnalysisItem,
    PipelineDiagnostics,
    QueryRequest,
    QueryResponse,
    ResearchGapItem,
    StagedFilesResponse,
    UploadResponse,
)
from retrieval.config import RAGConfig
from retrieval.cross_paper_analysis import CrossPaperAnalyzer
from retrieval.evidence import ClaimVerifier, Evidence, SupportStatus
from retrieval.paper_analysis import PaperAnalysis, PaperAnalyzer
from retrieval.retriver import Retriever

logger = logging.getLogger("ragss.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="PaperTrail / RAGSS API",
    description="Backend API for research paper analysis and retrieval transparency.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KNOWN_LIMITATIONS = [
    "Parent documents are stored in an in-memory mapping. A fresh process cannot expand persisted child chunks without re-ingesting parent pages.",
    "PDF page-boundary chunking can split technical paragraphs, equations, and tables across pages.",
    "Claim verification relies on lexical token-overlap heuristics and does NOT guarantee semantic entailment or factual correctness.",
    "Document IDs for unnumbered files can depend on sorted ingestion-batch position.",
]


class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.upload_dir = PROJECT_ROOT / "uploads" / session_id
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.retriever: Optional[Retriever] = None
        self.paper_analyses: List[PaperAnalysis] = []
        self.cross_analysis: Optional[Any] = None
        self.staged_files: Dict[str, Path] = {}
        self.processing: bool = False


# In-memory session store
sessions: Dict[str, SessionState] = {}


def get_or_create_session(session_id: Optional[str] = None) -> SessionState:
    if not session_id or not session_id.strip():
        session_id = uuid.uuid4().hex[:12]
    if session_id not in sessions:
        sessions[session_id] = SessionState(session_id)
    return sessions[session_id]


def _map_evidence(ev_list: List[Evidence]) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []
    for ev in ev_list:
        items.append(
            EvidenceItem(
                text=ev.evidence_text,
                document_id=ev.document_id,
                document_name=ev.document_name,
                page=ev.page,
                chunk_id=ev.chunk_id,
                source=getattr(ev, "source", None),
            )
        )
    return items


def _map_claims(claims_list: Any, verifier: ClaimVerifier) -> List[ClaimItem]:
    items: List[ClaimItem] = []
    for c in claims_list:
        ev_items = _map_evidence(c.evidence)
        # Calculate maximum token overlap across candidate supporting evidence chunks
        max_overlap = 0.0
        for ev in c.evidence:
            overlap, _ = verifier.compute_overlap(c.text, ev.evidence_text)
            if overlap > max_overlap:
                max_overlap = overlap

        items.append(
            ClaimItem(
                text=c.text,
                support_status=c.support_status.value if hasattr(c.support_status, "value") else str(c.support_status),
                citations=c.get_citations() if hasattr(c, "get_citations") else [],
                evidence=ev_items,
                token_overlap_score=round(max_overlap, 3) if max_overlap > 0 else 0.0,
            )
        )
    return items


def _map_citations(cit_list: List[Dict[str, Any]]) -> List[CitationItem]:
    return [
        CitationItem(
            document_id=c.get("document_id"),
            document_name=c.get("document_name"),
            source=c.get("source"),
            page=c.get("page"),
        )
        for c in cit_list
    ]


def _make_no_llm_analyzer_stub():
    def _stub(pv):
        text = pv.to_string() if hasattr(pv, "to_string") else str(pv)
        match = re.search(r"Provided Evidence for ([^:\n]+):", text)
        doc_name = match.group(1).strip() if match else "Paper"
        return (
            f"TITLE: {doc_name}\n"
            f"RESEARCH PROBLEM: Extracted from provided evidence.\n"
            f"METHODOLOGY: Proposed approach from retrieved sections.\n"
            f"MAJOR CLAIMS: Claims identified in retrieved sections.\n"
            f"EVIDENCE: Empirical evidence described in paper context.\n"
            f"LIMITATIONS: Identified limitations or not available.\n"
        )
    return RunnableLambda(_stub)


def _make_no_llm_cross_stub():
    def _stub(pv):
        return (
            "SIMILARITIES:\nCommon research topics across papers.\n\n"
            "DIFFERENCES:\nDifferent methodological formulations.\n\n"
            "CORROBORATING FINDINGS:\nEmpirical findings supported across papers.\n\n"
            "CONFLICTING FINDINGS:\nNone identified from the provided evidence.\n\n"
            "RESEARCH GAPS:\nOpen limitations noted across the collection.\n\n"
            "OVERALL SYNTHESIS:\nCollective synthesis of analyzed research collection.\n"
        )
    return RunnableLambda(_stub)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": "PaperTrail / RAGSS API"}


@app.get("/api/config", response_model=ConfigInfoResponse)
def get_config():
    cfg = RAGConfig()
    return ConfigInfoResponse(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        retrieval_k=cfg.default_k,
        embedding_model=cfg.embedding_model,
        cross_encoder_model=cfg.cross_encoder_model,
        llm_model=cfg.llm_model,
        known_limitations=KNOWN_LIMITATIONS,
    )


@app.post("/api/upload", response_model=StagedFilesResponse)
async def upload_papers(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
):
    session = get_or_create_session(session_id)

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded.",
        )

    results: List[UploadResponse] = []

    for file in files:
        filename = Path(file.filename or "uploaded.pdf").name
        if not filename.lower().endswith(".pdf"):
            results.append(
                UploadResponse(
                    filename=filename,
                    size_bytes=0,
                    valid=False,
                    error="Only PDF (.pdf) files are supported.",
                )
            )
            continue

        file_bytes = await file.read()
        size = len(file_bytes)

        if size == 0:
            results.append(
                UploadResponse(
                    filename=filename,
                    size_bytes=0,
                    valid=False,
                    error="Uploaded file is empty (0 bytes).",
                )
            )
            continue

        # Check PDF magic bytes (%PDF)
        if not file_bytes.startswith(b"%PDF"):
            results.append(
                UploadResponse(
                    filename=filename,
                    size_bytes=size,
                    valid=False,
                    error="Invalid file format: missing %PDF header magic bytes.",
                )
            )
            continue

        # Save to session upload directory
        dest_path = session.upload_dir / filename
        dest_path.write_bytes(file_bytes)
        session.staged_files[filename] = dest_path

        results.append(
            UploadResponse(
                filename=filename,
                size_bytes=size,
                valid=True,
                error=None,
            )
        )

    return StagedFilesResponse(session_id=session.session_id, files=results)


@app.get("/api/uploaded", response_model=StagedFilesResponse)
def get_staged_files(session_id: str):
    session = get_or_create_session(session_id)
    file_responses: List[UploadResponse] = []
    for filename, p in session.staged_files.items():
        if p.exists():
            file_responses.append(
                UploadResponse(
                    filename=filename,
                    size_bytes=p.stat().st_size,
                    valid=True,
                )
            )
    return StagedFilesResponse(session_id=session.session_id, files=file_responses)


@app.delete("/api/upload/{filename}", response_model=StagedFilesResponse)
def remove_staged_paper(filename: str, session_id: str):
    session = get_or_create_session(session_id)
    if filename in session.staged_files:
        path = session.staged_files.pop(filename)
        if path.exists():
            try:
                path.unlink()
            except OSError as e:
                logger.warning("Could not delete file %s: %s", path, e)

    return get_staged_files(session_id=session.session_id)


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_papers_endpoint(req: AnalyzeRequest):
    session = get_or_create_session(req.session_id)

    if session.processing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An analysis is already in progress for this session. Please wait for it to complete.",
        )

    if not session.staged_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No PDF files staged for analysis. Please upload at least one PDF paper first.",
        )

    session.processing = True
    try:
        # Sort files to ensure deterministic batch ingestion
        pdf_paths = [str(p) for p in sorted(session.staged_files.values()) if p.exists()]
        if not pdf_paths:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All staged PDF files have been removed or do not exist on disk.",
            )

        # Isolated session chroma persist directory (never overwrites main collection)
        session_chroma_dir = PROJECT_ROOT / "chroma_db" / "sessions" / session.session_id
        session_collection = f"session_{session.session_id}"

        # Initialize Retriever with session isolation
        retriever = Retriever(
            file_paths=pdf_paths,
            persist_directory=str(session_chroma_dir),
            collection_name=session_collection,
            rebuild_index=True,
        )
        session.retriever = retriever

        # Setup analyzer
        verifier = ClaimVerifier()
        llm = _make_no_llm_analyzer_stub() if req.no_llm else None
        analyzer = PaperAnalyzer(retriever=retriever, llm=llm, verifier=verifier)

        indexed_papers = retriever.get_indexed_papers()
        paper_results: List[PaperAnalysisItem] = []
        successful_analyses: List[PaperAnalysis] = []

        # Analyze each paper individually, isolating failures
        for meta in indexed_papers:
            doc_id = meta["document_id"]
            doc_name = meta["document_name"]
            try:
                pa = analyzer.analyze_paper(document_id=doc_id, document_name=doc_name, k=req.k)
                successful_analyses.append(pa)

                diag_model = None
                if pa.diagnostics:
                    diag_model = PipelineDiagnostics(**pa.diagnostics)

                paper_results.append(
                    PaperAnalysisItem(
                        document_id=pa.document_id,
                        document_name=pa.document_name,
                        title=pa.title,
                        research_problem=pa.research_problem,
                        methodology=pa.methodology,
                        major_claims=pa.major_claims,
                        evidence=pa.evidence,
                        limitations=pa.limitations,
                        citations=_map_citations(pa.citations),
                        source_evidence=_map_evidence(pa.source_evidence),
                        claims=_map_claims(pa.claims, verifier),
                        diagnostics=diag_model,
                        success=True,
                    )
                )
            except Exception as e:
                logger.error("Failed to analyze paper %s (%s): %s", doc_name, doc_id, e, exc_info=True)
                paper_results.append(
                    PaperAnalysisItem(
                        document_id=doc_id,
                        document_name=doc_name,
                        title=doc_name,
                        research_problem="Analysis failed.",
                        methodology="Analysis failed.",
                        major_claims="Analysis failed.",
                        evidence="Analysis failed.",
                        limitations="Analysis failed.",
                        success=False,
                        error=f"Error analyzing paper: {str(e)}",
                    )
                )

        session.paper_analyses = successful_analyses

        # Cross-paper analysis ONLY when >= 2 papers succeed
        cross_res: Optional[CrossPaperAnalysisItem] = None
        if len(successful_analyses) >= 2:
            try:
                cross_llm = _make_no_llm_cross_stub() if req.no_llm else None
                cross_analyzer = CrossPaperAnalyzer(llm=cross_llm, verifier=verifier)
                ca = cross_analyzer.analyze(successful_analyses)
                session.cross_analysis = ca

                corroborating = [
                    CorroboratingFindingItem(
                        finding=f.finding,
                        supporting_papers=getattr(f, "supporting_papers", []),
                        evidence=_map_evidence(f.evidence),
                    )
                    for f in ca.corroborating_findings_structured
                ]
                conflicting = [
                    ConflictingFindingItem(
                        topic=f.topic,
                        papers=getattr(f, "papers", []),
                        explanation=getattr(f, "explanation", ""),
                        evidence=_map_evidence(f.evidence),
                    )
                    for f in ca.conflicting_findings_structured
                ]
                gaps = [
                    ResearchGapItem(
                        finding=f.finding,
                        relevant_papers=getattr(f, "relevant_papers", []),
                        evidence=_map_evidence(f.evidence),
                    )
                    for f in ca.research_gaps_structured
                ]

                cross_res = CrossPaperAnalysisItem(
                    similarities=ca.similarities,
                    differences=ca.differences,
                    corroborating_findings=ca.corroborating_findings,
                    conflicting_findings=ca.conflicting_findings,
                    research_gaps=ca.research_gaps,
                    overall_synthesis=ca.overall_synthesis,
                    analyzed_papers=ca.analyzed_papers,
                    source_evidence=_map_evidence(ca.source_evidence),
                    corroborating_findings_structured=corroborating,
                    conflicting_findings_structured=conflicting,
                    research_gaps_structured=gaps,
                    success=True,
                )
            except Exception as ce_err:
                logger.error("Cross-paper analysis failed: %s", ce_err, exc_info=True)
                cross_res = CrossPaperAnalysisItem(
                    similarities="",
                    differences="",
                    corroborating_findings="",
                    conflicting_findings="",
                    research_gaps="",
                    overall_synthesis="",
                    success=False,
                    error=f"Cross-paper analysis encountered an error: {str(ce_err)}",
                )

        return AnalyzeResponse(
            session_id=session.session_id,
            papers=paper_results,
            cross_analysis=cross_res,
            total_submitted=len(pdf_paths),
            successful_count=len(successful_analyses),
            failed_count=len(paper_results) - len(successful_analyses),
        )

    finally:
        session.processing = False


@app.post("/api/cross-analyze", response_model=CrossPaperAnalysisItem)
def cross_analyze_endpoint(req: CrossAnalyzeRequest):
    session = get_or_create_session(req.session_id)

    if len(session.paper_analyses) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cross-paper analysis requires at least two successfully analyzed papers. Found {len(session.paper_analyses)}.",
        )

    verifier = ClaimVerifier()
    cross_llm = _make_no_llm_cross_stub() if req.no_llm else None
    cross_analyzer = CrossPaperAnalyzer(llm=cross_llm, verifier=verifier)
    ca = cross_analyzer.analyze(session.paper_analyses)

    corroborating = [
        CorroboratingFindingItem(
            finding=f.finding,
            supporting_papers=getattr(f, "supporting_papers", []),
            evidence=_map_evidence(f.evidence),
        )
        for f in ca.corroborating_findings_structured
    ]
    conflicting = [
        ConflictingFindingItem(
            topic=f.topic,
            papers=getattr(f, "papers", []),
            explanation=getattr(f, "explanation", ""),
            evidence=_map_evidence(f.evidence),
        )
        for f in ca.conflicting_findings_structured
    ]
    gaps = [
        ResearchGapItem(
            finding=f.finding,
            relevant_papers=getattr(f, "relevant_papers", []),
            evidence=_map_evidence(f.evidence),
        )
        for f in ca.research_gaps_structured
    ]

    return CrossPaperAnalysisItem(
        similarities=ca.similarities,
        differences=ca.differences,
        corroborating_findings=ca.corroborating_findings,
        conflicting_findings=ca.conflicting_findings,
        research_gaps=ca.research_gaps,
        overall_synthesis=ca.overall_synthesis,
        analyzed_papers=ca.analyzed_papers,
        source_evidence=_map_evidence(ca.source_evidence),
        corroborating_findings_structured=corroborating,
        conflicting_findings_structured=conflicting,
        research_gaps_structured=gaps,
        success=True,
    )


@app.post("/api/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query text cannot be empty.",
        )

    session = get_or_create_session(req.session_id)
    retriever = session.retriever

    if retriever is None:
        # Fall back to root retriever if available
        cfg = RAGConfig()
        retriever = Retriever(
            persist_directory=str(cfg.chroma_dir),
            rebuild_index=False,
        )

    results, diagnostics = retriever.retrieve_with_diagnostics(
        query=query_text,
        k=req.k,
        document_id=req.document_id,
    )

    citations: List[CitationItem] = []
    for doc in results:
        raw_page = doc.metadata.get("page", 0)
        try:
            page_num = int(raw_page) + 1
        except (ValueError, TypeError):
            page_num = 1
        citations.append(
            CitationItem(
                document_id=doc.metadata.get("document_id"),
                document_name=doc.metadata.get("document_name"),
                source=doc.metadata.get("source"),
                page=page_num,
            )
        )

    if req.no_llm:
        answer = f"Retrieval-only preview. Found {len(results)} relevant chunks."
    elif not results:
        answer = "I could not find sufficient evidence in the indexed papers to answer your question."
    else:
        from retrieval.retrieve_llm_pipeline import qa_chain
        try:
            ans_payload = qa_chain.invoke({"query": query_text, "k": req.k})
            answer = ans_payload.get("answer", "")
        except Exception as e:
            logger.warning("LLM generation failed (%s), falling back to context preview.", e)
            answer = "Context retrieved successfully. LLM generation unavailable."

    return QueryResponse(
        query=query_text,
        answer=answer,
        citations=citations,
        diagnostics=PipelineDiagnostics(**diagnostics),
        session_id=session.session_id,
    )


# Serve built React frontend if frontend/dist exists
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        file_path = frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")
