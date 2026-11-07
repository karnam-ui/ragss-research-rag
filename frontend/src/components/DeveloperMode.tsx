import React, { useState } from 'react';
import type { ChunkDiagnosticItem, PaperAnalysisItem, PipelineDiagnostics } from '../types';

interface DeveloperModeProps {
  papers: PaperAnalysisItem[];
  selectedPaperId: string | null;
  onSelectPaper: (docId: string) => void;
}

export const DeveloperMode: React.FC<DeveloperModeProps> = ({
  papers,
  selectedPaperId,
  onSelectPaper,
}) => {
  const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null);
  const [copiedText, setCopiedText] = useState<string | null>(null);

  const successfulPapers = papers.filter((p) => p.success);

  // If no paper selected, default to first paper
  const activePaper =
    successfulPapers.find((p) => p.document_id === selectedPaperId) ||
    successfulPapers[0] ||
    null;

  const diagnostics: PipelineDiagnostics | null = activePaper?.diagnostics || null;
  const chunks: ChunkDiagnosticItem[] = diagnostics?.chunks || [];

  const activeChunk = chunks.find((c) => c.chunk_id === selectedChunkId) || chunks[0] || null;

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    setCopiedText(label);
    setTimeout(() => setCopiedText(null), 2000);
  };

  const exportDiagnosticJson = () => {
    if (!diagnostics && !activePaper) return;
    const report = {
      timestamp: new Date().toISOString(),
      document_id: activePaper?.document_id,
      document_name: activePaper?.document_name,
      title: activePaper?.title,
      diagnostics: activePaper?.diagnostics,
      claims_provenance: activePaper?.claims.map((c) => ({
        claim: c.text,
        support_status: c.support_status,
        token_overlap_score: c.token_overlap_score,
        citations: c.citations,
        evidence: c.evidence,
      })),
      citations: activePaper?.citations,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `diagnostics_${activePaper?.document_id || 'ragss'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (successfulPapers.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <polyline points="16 18 22 12 16 6"></polyline>
            <polyline points="8 6 2 12 8 18"></polyline>
          </svg>
        </div>
        <h3>No Diagnostic Data Available</h3>
        <p>Analyze at least one paper to inspect real retrieval ranks, scores, chunk metadata, and evidence provenance.</p>
      </div>
    );
  }

  return (
    <div className="developer-mode-container" aria-label="Developer Mode Diagnostics">
      {/* 1. Header & Controls */}
      <div className="dev-header">
        <div className="dev-header-left">
          <h2>Retrieval Pipeline Transparency</h2>
          <p className="dev-header-sub">
            Real execution telemetry: dense & lexical scores, RRF fusion ranks, cross-encoder scores, and parent-child provenance.
          </p>
        </div>

        <div className="dev-header-right">
          <button
            type="button"
            className="export-json-btn"
            onClick={exportDiagnosticJson}
            title="Download full execution diagnostics as JSON"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            Export Diagnostic JSON
          </button>
        </div>
      </div>

      {/* 2. Paper Selector Tabs */}
      <div className="paper-tabs" role="tablist">
        {successfulPapers.map((p) => (
          <button
            key={p.document_id}
            type="button"
            role="tab"
            aria-selected={activePaper?.document_id === p.document_id}
            className={`paper-tab ${activePaper?.document_id === p.document_id ? 'active' : ''}`}
            onClick={() => {
              onSelectPaper(p.document_id);
              setSelectedChunkId(null);
            }}
          >
            📄 {p.document_name}
            <span className="tab-badge">{p.diagnostics?.chunks.length ?? 0} chunks</span>
          </button>
        ))}
      </div>

      {/* 3. Pipeline Diagnostics Summary Cards */}
      {diagnostics && (
        <div className="diagnostics-summary-grid">
          <div className="diag-card">
            <div className="diag-card-header">
              <span className="diag-title">1. Dense Retrieval</span>
              <span className={`stage-status ${diagnostics.stages?.dense?.ran ? 'ran' : 'skipped'}`}>
                {diagnostics.stages?.dense?.ran ? 'Ran' : 'Skipped'}
              </span>
            </div>
            <div className="diag-metric">
              {diagnostics.stages?.dense?.retrieved_count ?? 0} <span className="unit">chunks</span>
            </div>
            <div className="diag-meta">
              Latency: {diagnostics.timings?.dense_ms ?? 0} ms | Model: bge-small-en
            </div>
          </div>

          <div className="diag-card">
            <div className="diag-card-header">
              <span className="diag-title">2. Lexical BM25</span>
              <span className={`stage-status ${diagnostics.stages?.bm25?.ran ? 'ran' : 'skipped'}`}>
                {diagnostics.stages?.bm25?.ran ? 'Ran' : 'Skipped'}
              </span>
            </div>
            <div className="diag-metric">
              {diagnostics.stages?.bm25?.retrieved_count ?? 0} <span className="unit">chunks</span>
            </div>
            <div className="diag-meta">
              Latency: {diagnostics.timings?.bm25_ms ?? 0} ms | Term overlap strictly enforced
            </div>
          </div>

          <div className="diag-card">
            <div className="diag-card-header">
              <span className="diag-title">3. Reciprocal Rank Fusion</span>
              <span className="stage-status ran">Ran</span>
            </div>
            <div className="diag-metric">
              {diagnostics.stages?.rrf?.fused_count ?? 0} <span className="unit">fused</span>
            </div>
            <div className="diag-meta">
              Latency: {diagnostics.timings?.rrf_ms ?? 0} ms | RRF constant k={diagnostics.stages?.rrf?.k ?? 60}
            </div>
          </div>

          <div className="diag-card">
            <div className="diag-card-header">
              <span className="diag-title">4. Cross-Encoder Rerank</span>
              <span className={`stage-status ${diagnostics.stages?.cross_encoder?.ran ? 'ran' : 'skipped'}`}>
                {diagnostics.stages?.cross_encoder?.ran ? 'Ran' : 'Disabled'}
              </span>
            </div>
            <div className="diag-metric">
              {diagnostics.stages?.cross_encoder?.reranked_count ?? 0} <span className="unit">scored</span>
            </div>
            <div className="diag-meta">
              Latency: {diagnostics.timings?.rerank_ms ?? 0} ms | Fallback: {diagnostics.stages?.cross_encoder?.fallback ? 'Yes' : 'No'}
            </div>
          </div>

          <div className="diag-card">
            <div className="diag-card-header">
              <span className="diag-title">5. Parent Expansion</span>
              <span className="stage-status ran">Ran</span>
            </div>
            <div className="diag-metric">
              {diagnostics.stages?.parent_expansion?.expanded_count ?? 0} <span className="unit">expanded</span>
            </div>
            <div className="diag-meta">
              Fallback orphans: {diagnostics.stages?.parent_expansion?.fallback_count ?? 0} | Latency: {diagnostics.timings?.parent_expansion_ms ?? 0} ms
            </div>
          </div>

          <div className="diag-card highlight">
            <div className="diag-card-header">
              <span className="diag-title">Total Retrieval Latency</span>
              <span className="stage-status ran">Real Time</span>
            </div>
            <div className="diag-metric highlight">
              {diagnostics.timings?.total_ms ?? 0} <span className="unit">ms</span>
            </div>
            <div className="diag-meta">
              Filter: <code>doc_id={diagnostics.filters?.document_id || 'all'}</code>
            </div>
          </div>
        </div>
      )}

      {/* 4. Retrieval Transparency Table */}
      <section className="retrieval-table-section">
        <div className="table-header-row">
          <h3>Retrieved Chunks & Ranking Scores</h3>
          <span className="caption-text">
            *Scores come from distinct scales (Dense distance, BM25 TF-IDF, RRF reciprocal rank, Cross-Encoder logit) and are labeled accordingly.
          </span>
        </div>

        {chunks.length === 0 ? (
          <div className="empty-chunks">
            No chunks were returned by retrieval. {diagnostics?.abstention_reason && `Reason: ${diagnostics.abstention_reason}`}
          </div>
        ) : (
          <div className="table-responsive">
            <table className="dev-table">
              <thead>
                <tr>
                  <th>Final Rank</th>
                  <th>Chunk ID</th>
                  <th>Source & Page</th>
                  <th>Dense (Rank / Score)</th>
                  <th>BM25 (Rank / Score)</th>
                  <th>RRF (Rank / Score)</th>
                  <th>Cross-Encoder (Rank / Score)</th>
                  <th>Parent Expansion</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {chunks.map((c) => {
                  const isSelected = activeChunk?.chunk_id === c.chunk_id;
                  return (
                    <tr
                      key={c.chunk_id}
                      className={isSelected ? 'selected-row' : ''}
                      onClick={() => setSelectedChunkId(c.chunk_id)}
                    >
                      <td>
                        <span className="rank-badge">
                          {c.final_rank ? `#${c.final_rank}` : 'Candidate'}
                        </span>
                      </td>
                      <td>
                        <code className="code-id" title={c.chunk_id}>
                          {c.chunk_id.length > 20 ? `${c.chunk_id.slice(0, 18)}...` : c.chunk_id}
                        </code>
                      </td>
                      <td>
                        <div className="cell-doc">{c.document_name || 'Paper'}</div>
                        <div className="cell-sub">Page {c.page !== null && c.page !== undefined ? Number(c.page) + 1 : 'N/A'}</div>
                      </td>
                      <td>
                        {c.dense_rank ? (
                          <span>
                            #{c.dense_rank} <span className="score-val">({c.dense_score !== null && c.dense_score !== undefined ? c.dense_score.toFixed(3) : 'dist'})</span>
                          </span>
                        ) : (
                          <span className="na-text">Not available</span>
                        )}
                      </td>
                      <td>
                        {c.bm25_rank ? (
                          <span>
                            #{c.bm25_rank} <span className="score-val">({c.bm25_score?.toFixed(2) ?? '0.00'})</span>
                          </span>
                        ) : (
                          <span className="na-text">Not available</span>
                        )}
                      </td>
                      <td>
                        {c.rrf_rank ? (
                          <span>
                            #{c.rrf_rank} <span className="score-val">({c.rrf_score?.toFixed(4) ?? '0.0000'})</span>
                          </span>
                        ) : (
                          <span className="na-text">Not available</span>
                        )}
                      </td>
                      <td>
                        {c.cross_encoder_rank ? (
                          <span>
                            #{c.cross_encoder_rank} <span className="score-val">({c.cross_encoder_score?.toFixed(3) ?? '0.000'})</span>
                          </span>
                        ) : (
                          <span className="na-text">Not available</span>
                        )}
                      </td>
                      <td>
                        {c.expanded ? (
                          <span className="expansion-badge expanded">Expanded to Page</span>
                        ) : (
                          <span className="expansion-badge child">Child Chunk</span>
                        )}
                      </td>
                      <td>
                        <button
                          type="button"
                          className="view-chunk-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedChunkId(c.chunk_id);
                          }}
                        >
                          Inspect
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* 5. Chunk Inspector Detail View */}
      {activeChunk && (
        <section className="chunk-inspector-section">
          <div className="inspector-header">
            <h3>Chunk Detail Inspector: <code>{activeChunk.chunk_id}</code></h3>
            <span className="meta-tag">{activeChunk.document_name} | Page {activeChunk.page !== null && activeChunk.page !== undefined ? Number(activeChunk.page) + 1 : 'N/A'}</span>
          </div>

          <div className="inspector-grid">
            {/* Raw Retrieved Chunk */}
            <div className="inspector-panel">
              <div className="panel-header">
                <h4>Raw Retrieved Chunk Text</h4>
                <button
                  type="button"
                  className="copy-btn"
                  onClick={() => copyToClipboard(activeChunk.chunk_text, 'chunk')}
                >
                  {copiedText === 'chunk' ? 'Copied ✓' : 'Copy Text'}
                </button>
              </div>
              <div className="panel-content scrollable">
                <pre>{activeChunk.chunk_text}</pre>
              </div>
            </div>

            {/* Parent-page context or expansion */}
            <div className="inspector-panel">
              <div className="panel-header">
                <h4>Parent-Page Context {activeChunk.expanded ? '(Expanded)' : '(Not Expanded)'}</h4>
                {activeChunk.parent_content && (
                  <button
                    type="button"
                    className="copy-btn"
                    onClick={() => copyToClipboard(activeChunk.parent_content!, 'parent')}
                  >
                    {copiedText === 'parent' ? 'Copied ✓' : 'Copy Context'}
                  </button>
                )}
              </div>
              <div className="panel-content scrollable">
                {activeChunk.parent_content ? (
                  <pre>{activeChunk.parent_content}</pre>
                ) : (
                  <div className="na-box">
                    <p>Parent document not expanded for this chunk.</p>
                    <span className="na-note">
                      Notice: As documented, parent documents are stored in memory; fallback used child chunk content directly.
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Metadata Inspector */}
          <div className="metadata-panel">
            <h4>Document & Chunk Metadata</h4>
            <div className="metadata-grid">
              <div><strong>Parent ID:</strong> <code>{activeChunk.parent_id || 'None'}</code></div>
              <div><strong>Document ID:</strong> <code>{activeChunk.document_id || 'None'}</code></div>
              <div><strong>Source Path:</strong> <code>{activeChunk.source || 'None'}</code></div>
              <div><strong>Page Index:</strong> {activeChunk.page ?? 'None'} (Display: {activeChunk.page !== null && activeChunk.page !== undefined ? Number(activeChunk.page) + 1 : 'N/A'})</div>
            </div>
          </div>
        </section>
      )}

      {/* 6. Claim Provenance & Lexical Verification Inspector */}
      {activePaper && activePaper.claims.length > 0 && (
        <section className="claims-inspector-section">
          <div className="section-header">
            <h3>Claim Provenance & Heuristic Verification</h3>
            <div className="caveat-banner">
              <strong>⚠️ Heuristic Limitation Notice:</strong> Claim verification uses lexical token-overlap heuristics (content word intersection ratio). It does <em>not</em> perform semantic entailment, cannot detect semantic negation, and uncalibrated scores do not guarantee factual correctness.
            </div>
          </div>

          <div className="provenance-cards-list">
            {activePaper.claims.map((claim, idx) => (
              <div key={idx} className="provenance-card">
                <div className="provenance-header">
                  <div className="prov-title">
                    <span className="claim-idx">Claim [{idx + 1}]:</span>
                    <span className="claim-body">{claim.text}</span>
                  </div>
                  <div className="prov-score-group">
                    <span className="overlap-score">
                      Token Overlap: <strong>{claim.token_overlap_score !== null && claim.token_overlap_score !== undefined ? (claim.token_overlap_score * 100).toFixed(1) : '0.0'}%</strong>
                    </span>
                    <span className={`status-badge ${claim.support_status.toLowerCase()}`}>
                      {claim.support_status}
                    </span>
                  </div>
                </div>

                <div className="prov-evidence-area">
                  <h5>Supporting Evidence Passages ({claim.evidence.length}):</h5>
                  {claim.evidence.length === 0 ? (
                    <p className="no-ev-text">No candidate evidence passed the lexical support threshold.</p>
                  ) : (
                    <div className="ev-passages-list">
                      {claim.evidence.map((ev, eIdx) => (
                        <div key={eIdx} className="ev-passage-card">
                          <p>"{ev.text}"</p>
                          <div className="ev-meta">
                            <span>📄 {ev.document_name}</span>
                            <span>Page {ev.page || 1}</span>
                            {ev.chunk_id && <span>Chunk: <code>{ev.chunk_id}</code></span>}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
};
