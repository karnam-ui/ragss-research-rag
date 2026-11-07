from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    filename: str
    size_bytes: int
    valid: bool
    error: Optional[str] = None


class StagedFilesResponse(BaseModel):
    session_id: str
    files: List[UploadResponse]


class EvidenceItem(BaseModel):
    text: str
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    source: Optional[str] = None


class ClaimItem(BaseModel):
    text: str
    support_status: str  # "SUPPORTED", "INSUFFICIENT_EVIDENCE", "UNSUPPORTED"
    citations: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    token_overlap_score: Optional[float] = None


class CitationItem(BaseModel):
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    source: Optional[str] = None
    page: Optional[int] = None


class ChunkDiagnosticItem(BaseModel):
    chunk_id: str
    parent_id: Optional[str] = None
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    source: Optional[str] = None
    page: Optional[int] = None
    chunk_text: str
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    rrf_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    cross_encoder_rank: Optional[int] = None
    cross_encoder_score: Optional[float] = None
    final_rank: Optional[int] = None
    expanded: bool = False
    parent_content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineDiagnostics(BaseModel):
    query: Optional[str] = None
    stages: Dict[str, Any] = Field(default_factory=dict)
    filters: Dict[str, Any] = Field(default_factory=dict)
    timings: Dict[str, float] = Field(default_factory=dict)
    abstention_reason: Optional[str] = None
    chunks: List[ChunkDiagnosticItem] = Field(default_factory=list)


class PaperAnalysisItem(BaseModel):
    document_id: str
    document_name: str
    title: str
    research_problem: str
    methodology: str
    major_claims: str
    evidence: str
    limitations: str
    citations: List[CitationItem] = Field(default_factory=list)
    source_evidence: List[EvidenceItem] = Field(default_factory=list)
    claims: List[ClaimItem] = Field(default_factory=list)
    diagnostics: Optional[PipelineDiagnostics] = None
    success: bool = True
    error: Optional[str] = None


class CorroboratingFindingItem(BaseModel):
    finding: str
    supporting_papers: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ConflictingFindingItem(BaseModel):
    topic: str
    papers: List[str] = Field(default_factory=list)
    explanation: str = ""
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ResearchGapItem(BaseModel):
    finding: str
    relevant_papers: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)


class CrossPaperAnalysisItem(BaseModel):
    similarities: str
    differences: str
    corroborating_findings: str
    conflicting_findings: str
    research_gaps: str
    overall_synthesis: str
    analyzed_papers: List[Dict[str, str]] = Field(default_factory=list)
    source_evidence: List[EvidenceItem] = Field(default_factory=list)
    corroborating_findings_structured: List[CorroboratingFindingItem] = Field(default_factory=list)
    conflicting_findings_structured: List[ConflictingFindingItem] = Field(default_factory=list)
    research_gaps_structured: List[ResearchGapItem] = Field(default_factory=list)
    diagnostics: Optional[PipelineDiagnostics] = None
    success: bool = True
    error: Optional[str] = None


class AnalyzeRequest(BaseModel):
    session_id: Optional[str] = None
    k: int = 4
    no_llm: bool = False


class AnalyzeResponse(BaseModel):
    session_id: str
    papers: List[PaperAnalysisItem]
    cross_analysis: Optional[CrossPaperAnalysisItem] = None
    total_submitted: int
    successful_count: int
    failed_count: int


class CrossAnalyzeRequest(BaseModel):
    session_id: Optional[str] = None
    no_llm: bool = False


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    document_id: Optional[str] = None
    k: int = 4
    no_llm: bool = False


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationItem] = Field(default_factory=list)
    diagnostics: PipelineDiagnostics
    session_id: Optional[str] = None


class ConfigInfoResponse(BaseModel):
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    embedding_model: str
    cross_encoder_model: str
    llm_model: str
    known_limitations: List[str]
