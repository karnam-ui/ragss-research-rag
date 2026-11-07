import React, { useState } from 'react';
import type { CrossPaperAnalysisItem, PaperAnalysisItem } from '../types';

interface ResearchModeProps {
  papers: PaperAnalysisItem[];
  crossAnalysis: CrossPaperAnalysisItem | null | undefined;
  onSelectPaperForDevMode?: (docId: string) => void;
}

export const ResearchMode: React.FC<ResearchModeProps> = ({
  papers,
  crossAnalysis,
  onSelectPaperForDevMode,
}) => {
  const [expandedPapers, setExpandedPapers] = useState<Record<string, boolean>>({});
  const [expandedClaims, setExpandedClaims] = useState<Record<string, boolean>>({});

  const togglePaper = (docId: string) => {
    setExpandedPapers((prev) => ({
      ...prev,
      [docId]: !prev[docId],
    }));
  };

  const toggleClaim = (claimKey: string) => {
    setExpandedClaims((prev) => ({
      ...prev,
      [claimKey]: !prev[claimKey],
    }));
  };

  const successfulPapers = papers.filter((p) => p.success);
  const failedPapers = papers.filter((p) => !p.success);

  if (papers.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
          </svg>
        </div>
        <h3>No Paper Analyses Yet</h3>
        <p>Upload one or more PDF research papers above and click <strong>Analyze Papers</strong> to generate grounded evidence analyses.</p>
      </div>
    );
  }

  const renderStatusBadge = (status: string) => {
    const s = status.toUpperCase();
    if (s === 'SUPPORTED') {
      return <span className="status-badge supported">Supported</span>;
    } else if (s === 'INSUFFICIENT_EVIDENCE' || s === 'NOT_ENOUGH_EVIDENCE' || s === 'PARTIALLY_SUPPORTED') {
      return <span className="status-badge warning">Insufficient Evidence</span>;
    } else {
      return <span className="status-badge unsupported">Unsupported</span>;
    }
  };

  return (
    <div className="research-mode-container">
      {/* 1. Partial Failures notification if any */}
      {failedPapers.length > 0 && (
        <div className="alert-box warning" role="alert">
          <h4>Analysis Warnings ({failedPapers.length} paper{failedPapers.length > 1 ? 's' : ''} failed)</h4>
          <p>The following papers encountered processing issues, but successful analyses have been preserved:</p>
          <ul className="error-list">
            {failedPapers.map((fp, i) => (
              <li key={i}>
                <strong>{fp.document_name}</strong>: {fp.error || 'Parsing or retrieval failure'}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 2. Individual Paper Analyses */}
      <section className="section-block" aria-label="Individual Paper Analyses">
        <div className="section-header">
          <div className="section-title-row">
            <h2>Individual Paper Analyses</h2>
            <span className="count-pill">{successfulPapers.length} {successfulPapers.length === 1 ? 'Paper' : 'Papers'}</span>
          </div>
          <p className="section-desc">
            Independent, structured analyses produced by <code>PaperAnalyzer</code> directly from retrieved evidence passages.
          </p>
        </div>

        <div className="papers-stack">
          {successfulPapers.map((paper, idx) => {
            const isExpanded = expandedPapers[paper.document_id] ?? true;
            return (
              <article key={paper.document_id} className="paper-card" id={`paper-${paper.document_id}`}>
                <header
                  className="paper-card-header"
                  onClick={() => togglePaper(paper.document_id)}
                  tabIndex={0}
                  role="button"
                  aria-expanded={isExpanded}
                >
                  <div className="paper-title-area">
                    <span className="paper-index-badge">[{idx + 1}]</span>
                    <div>
                      <h3 className="paper-title">{paper.title || paper.document_name}</h3>
                      <div className="paper-meta-line">
                        <span className="meta-tag doc-name">📄 {paper.document_name}</span>
                        <span className="meta-tag doc-id">ID: <code>{paper.document_id}</code></span>
                        <span className="meta-tag citations-count">{paper.citations.length} Citations</span>
                      </div>
                    </div>
                  </div>
                  <div className="card-actions">
                    {onSelectPaperForDevMode && (
                      <button
                        type="button"
                        className="inspect-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectPaperForDevMode(paper.document_id);
                        }}
                        title="Inspect retrieval chunks and scores in Developer Mode"
                      >
                        Inspect Retrieval 🔍
                      </button>
                    )}
                    <span className="toggle-chevron">{isExpanded ? '▲' : '▼'}</span>
                  </div>
                </header>

                {isExpanded && (
                  <div className="paper-card-body">
                    {/* Summary Findings Grid */}
                    <div className="findings-grid">
                      <div className="finding-box problem">
                        <h4>🎯 Research Problem</h4>
                        <p>{paper.research_problem}</p>
                      </div>

                      <div className="finding-box methodology">
                        <h4>⚙️ Methodology</h4>
                        <p>{paper.methodology}</p>
                      </div>

                      <div className="finding-box claims">
                        <h4>📌 Major Claims</h4>
                        <div className="pre-wrap-text">{paper.major_claims}</div>
                      </div>

                      <div className="finding-box evidence">
                        <h4>🔬 Empirical Evidence</h4>
                        <div className="pre-wrap-text">{paper.evidence}</div>
                      </div>

                      <div className="finding-box limitations">
                        <h4>⚠️ Stated Limitations</h4>
                        <p>{paper.limitations}</p>
                      </div>
                    </div>

                    {/* Claims Verification Section */}
                    {paper.claims.length > 0 && (
                      <div className="claims-section">
                        <h4>Verified Claims & Evidence Provenance ({paper.claims.length})</h4>
                        <p className="claims-caption">
                          Claims verified against source passages using lexical overlap heuristics.
                        </p>
                        <div className="claims-list">
                          {paper.claims.map((claim, cIdx) => {
                            const claimKey = `${paper.document_id}_claim_${cIdx}`;
                            const isClaimOpen = expandedClaims[claimKey] ?? false;
                            const hasEvidence = claim.evidence && claim.evidence.length > 0;

                            return (
                              <div key={cIdx} className="claim-card">
                                <div className="claim-header">
                                  <div className="claim-text-row">
                                    {renderStatusBadge(claim.support_status)}
                                    <span className="claim-text">{claim.text}</span>
                                  </div>
                                  {hasEvidence && (
                                    <button
                                      type="button"
                                      className="evidence-toggle-btn"
                                      onClick={() => toggleClaim(claimKey)}
                                    >
                                      {isClaimOpen ? 'Hide Evidence' : `View Evidence (${claim.evidence.length})`}
                                    </button>
                                  )}
                                </div>

                                {isClaimOpen && hasEvidence && (
                                  <div className="claim-evidence-panel">
                                    <h5>Supporting Passages:</h5>
                                    {claim.evidence.map((ev, eIdx) => (
                                      <blockquote key={eIdx} className="evidence-passage">
                                        <p>"{ev.text}"</p>
                                        <footer className="evidence-citation">
                                          — <strong>{ev.document_name}</strong> (Page {ev.page || 1})
                                        </footer>
                                      </blockquote>
                                    ))}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* Citations Section */}
                    {paper.citations.length > 0 && (
                      <div className="citations-section">
                        <h4>Evidence Citations</h4>
                        <div className="citations-pills">
                          {paper.citations.map((c, i) => (
                            <span key={i} className="citation-pill">
                              <strong>[{c.document_name || 'Paper'}]</strong> Page {c.page || 1}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </article>
            );
          })}
        </div>
      </section>

      {/* 3. Cross-Paper Comparative Analysis (Only when >= 2 papers) */}
      {successfulPapers.length >= 2 && crossAnalysis && (
        <section className="section-block cross-analysis-block" aria-label="Cross-Paper Comparative Analysis">
          <div className="section-header">
            <div className="section-title-row">
              <h2>Cross-Paper Comparative Analysis</h2>
              <span className="count-pill accent">Synthesized across {crossAnalysis.analyzed_papers.length} Papers</span>
            </div>
            <p className="section-desc">
              Multi-paper synthesis generated by <code>CrossPaperAnalyzer</code>. Distinguishes common themes, agreements, and divergences without fabricating consensus.
            </p>
          </div>

          {!crossAnalysis.success ? (
            <div className="alert-box error">
              <h4>Cross-Paper Analysis Error</h4>
              <p>{crossAnalysis.error || 'Failed to synthesize cross-paper analysis.'}</p>
            </div>
          ) : (
            <div className="cross-analysis-card">
              <div className="cross-grid">
                <div className="cross-box similarities">
                  <h4>🔗 Similarities & Shared Objectives</h4>
                  <div className="pre-wrap-text">{crossAnalysis.similarities}</div>
                </div>

                <div className="cross-box differences">
                  <h4>⚡ Methodological & Conceptual Differences</h4>
                  <div className="pre-wrap-text">{crossAnalysis.differences}</div>
                </div>

                <div className="cross-box corroborating">
                  <h4>🤝 Corroborating Findings</h4>
                  <div className="pre-wrap-text">{crossAnalysis.corroborating_findings}</div>
                  {crossAnalysis.corroborating_findings_structured.length > 0 && (
                    <div className="structured-findings">
                      {crossAnalysis.corroborating_findings_structured.map((f, i) => (
                        <div key={i} className="structured-item">
                          <strong>Finding:</strong> {f.finding}
                          <div className="paper-attribution">
                            Supported by: {f.supporting_papers.join(', ')}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="cross-box conflicting">
                  <h4>⚔️ Conflicting Findings & Divergences</h4>
                  <div className="pre-wrap-text">{crossAnalysis.conflicting_findings}</div>
                  {crossAnalysis.conflicting_findings_structured.length > 0 && (
                    <div className="structured-findings">
                      {crossAnalysis.conflicting_findings_structured.map((cf, i) => (
                        <div key={i} className="structured-item">
                          <strong>Topic:</strong> {cf.topic}
                          <p>{cf.explanation}</p>
                          <div className="paper-attribution">
                            Papers: {cf.papers.join(', ')}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="cross-box gaps">
                  <h4>🔍 Recurring Research Gaps & Limitations</h4>
                  <div className="pre-wrap-text">{crossAnalysis.research_gaps}</div>
                </div>

                <div className="cross-box synthesis full-width">
                  <h4>🌐 Overall Research Synthesis</h4>
                  <div className="pre-wrap-text synthesis-text">{crossAnalysis.overall_synthesis}</div>
                </div>
              </div>
            </div>
          )}
        </section>
      )}

      {/* Note when only 1 paper was analyzed */}
      {successfulPapers.length === 1 && (
        <div className="single-paper-info-banner">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <div>
            <strong>Single Paper Analyzed:</strong> Cross-paper comparative synthesis is disabled because only 1 paper was uploaded. Upload at least 2 papers to compare methodologies, agreements, and research gaps.
          </div>
        </div>
      )}
    </div>
  );
};
