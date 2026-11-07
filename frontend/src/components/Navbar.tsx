import React, { useState } from 'react';
import type { ConfigInfoResponse } from '../types';

interface NavbarProps {
  mode: 'research' | 'developer';
  onModeChange: (mode: 'research' | 'developer') => void;
  config: ConfigInfoResponse | null;
}

export const Navbar: React.FC<NavbarProps> = ({ mode, onModeChange, config }) => {
  const [showLimitations, setShowLimitations] = useState(false);

  return (
    <>
      <header className="navbar">
        <div className="navbar-brand">
          <div className="brand-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
              <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
              <line x1="9" y1="7" x2="15" y2="7"></line>
              <line x1="9" y1="11" x2="15" y2="11"></line>
            </svg>
          </div>
          <div className="brand-text">
            <h1 className="brand-title">PaperTrail <span className="brand-badge">RAGSS</span></h1>
            <p className="brand-subtitle">Grounded Research Paper Evidence & Retrieval Transparency</p>
          </div>
        </div>

        <div className="navbar-controls">
          <div className="mode-switcher" role="tablist" aria-label="Application Mode">
            <button
              id="mode-research-btn"
              type="button"
              role="tab"
              aria-selected={mode === 'research'}
              className={`mode-btn ${mode === 'research' ? 'active' : ''}`}
              onClick={() => onModeChange('research')}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
              </svg>
              Research Mode
            </button>
            <button
              id="mode-developer-btn"
              type="button"
              role="tab"
              aria-selected={mode === 'developer'}
              className={`mode-btn ${mode === 'developer' ? 'active' : ''}`}
              onClick={() => onModeChange('developer')}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="16 18 22 12 16 6"></polyline>
                <polyline points="8 6 2 12 8 18"></polyline>
              </svg>
              Developer Mode
            </button>
          </div>

          <button
            type="button"
            className="limitations-btn"
            onClick={() => setShowLimitations(true)}
            title="View system invariants and known limitations"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="16" x2="12" y2="12"></line>
              <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
            System Invariants
          </button>
        </div>
      </header>

      {showLimitations && (
        <div className="modal-overlay" onClick={() => setShowLimitations(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>System Architecture & Known Limitations</h2>
              <button
                className="close-btn"
                onClick={() => setShowLimitations(false)}
                aria-label="Close modal"
              >
                ✕
              </button>
            </div>
            <div className="modal-body">
              <p className="modal-intro">
                PaperTrail / RAGSS is engineered with explicit retrieval invariants. The following known limitations are preserved and surfaced transparently:
              </p>
              <ul className="limitations-list">
                {config?.known_limitations.map((lim, idx) => (
                  <li key={idx} className="limitation-item">
                    <span className="limitation-bullet">•</span>
                    <span>{lim}</span>
                  </li>
                ))}
              </ul>
              <div className="config-summary-box">
                <h4>Pipeline Configuration</h4>
                <div className="config-grid">
                  <div><strong>Chunk Size:</strong> {config?.chunk_size ?? 500} tokens</div>
                  <div><strong>Chunk Overlap:</strong> {config?.chunk_overlap ?? 100} tokens</div>
                  <div><strong>Dense Embedding:</strong> {config?.embedding_model ?? 'BAAI/bge-small-en-v1.5'}</div>
                  <div><strong>Cross-Encoder:</strong> {config?.cross_encoder_model ?? 'cross-encoder/ms-marco-MiniLM-L-6-v2'}</div>
                  <div><strong>LLM Generator:</strong> {config?.llm_model ?? 'Qwen/Qwen2.5-1.5B-Instruct'}</div>
                  <div><strong>Default K:</strong> {config?.retrieval_k ?? 4}</div>
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button className="primary-btn" onClick={() => setShowLimitations(false)}>
                Dismiss
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
