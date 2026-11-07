export interface UploadFileResponse {
  filename: string;
  size_bytes: number;
  valid: boolean;
  error?: string | null;
}

export interface StagedFilesResponse {
  session_id: string;
  files: UploadFileResponse[];
}

export interface EvidenceItem {
  text: string;
  document_id?: string | null;
  document_name?: string | null;
  page?: number | null;
  chunk_id?: string | null;
  source?: string | null;
}

export interface ClaimItem {
  text: string;
  support_status: string; // "SUPPORTED" | "INSUFFICIENT_EVIDENCE" | "UNSUPPORTED" | "NOT_ENOUGH_EVIDENCE"
  citations: string[];
  evidence: EvidenceItem[];
  token_overlap_score?: number | null;
}

export interface CitationItem {
  document_id?: string | null;
  document_name?: string | null;
  source?: string | null;
  page?: number | null;
}

export interface ChunkDiagnosticItem {
  chunk_id: string;
  parent_id?: string | null;
  document_id?: string | null;
  document_name?: string | null;
  source?: string | null;
  page?: number | null;
  chunk_text: string;
  dense_rank?: number | null;
  dense_score?: number | null;
  bm25_rank?: number | null;
  bm25_score?: number | null;
  rrf_rank?: number | null;
  rrf_score?: number | null;
  cross_encoder_rank?: number | null;
  cross_encoder_score?: number | null;
  final_rank?: number | null;
  expanded: boolean;
  parent_content?: string | null;
  metadata: Record<string, any>;
}

export interface StageDiagnostic {
  ran: boolean;
  retrieved_count?: number;
  fused_count?: number;
  reranked_count?: number;
  expanded_count?: number;
  fallback_count?: number;
  fallback?: boolean;
  timing_ms?: number;
  error?: string | null;
  k?: number;
}

export interface PipelineDiagnostics {
  query?: string | null;
  stages: Record<string, StageDiagnostic>;
  filters: Record<string, any>;
  timings: Record<string, number>;
  abstention_reason?: string | null;
  chunks: ChunkDiagnosticItem[];
}

export interface PaperAnalysisItem {
  document_id: string;
  document_name: string;
  title: string;
  research_problem: string;
  methodology: string;
  major_claims: string;
  evidence: string;
  limitations: string;
  citations: CitationItem[];
  source_evidence: EvidenceItem[];
  claims: ClaimItem[];
  diagnostics?: PipelineDiagnostics | null;
  success: boolean;
  error?: string | null;
}

export interface CorroboratingFindingItem {
  finding: string;
  supporting_papers: string[];
  evidence: EvidenceItem[];
}

export interface ConflictingFindingItem {
  topic: string;
  papers: string[];
  explanation: string;
  evidence: EvidenceItem[];
}

export interface ResearchGapItem {
  finding: string;
  relevant_papers: string[];
  evidence: EvidenceItem[];
}

export interface CrossPaperAnalysisItem {
  similarities: string;
  differences: string;
  corroborating_findings: string;
  conflicting_findings: string;
  research_gaps: string;
  overall_synthesis: string;
  analyzed_papers: Array<{ document_id: string; document_name: string; title: string }>;
  source_evidence: EvidenceItem[];
  corroborating_findings_structured: CorroboratingFindingItem[];
  conflicting_findings_structured: ConflictingFindingItem[];
  research_gaps_structured: ResearchGapItem[];
  diagnostics?: PipelineDiagnostics | null;
  success: boolean;
  error?: string | null;
}

export interface AnalyzeResponse {
  session_id: string;
  papers: PaperAnalysisItem[];
  cross_analysis?: CrossPaperAnalysisItem | null;
  total_submitted: number;
  successful_count: number;
  failed_count: number;
}

export interface ConfigInfoResponse {
  chunk_size: number;
  chunk_overlap: number;
  retrieval_k: number;
  embedding_model: string;
  cross_encoder_model: string;
  llm_model: string;
  known_limitations: string[];
}
