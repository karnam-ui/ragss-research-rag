import React, { useRef, useState } from 'react';
import type { UploadFileResponse } from '../types';

interface UploadAreaProps {
  stagedFiles: UploadFileResponse[];
  isAnalyzing: boolean;
  statusMessage: string;
  k: number;
  noLlm: boolean;
  onKChange: (k: number) => void;
  onNoLlmChange: (noLlm: boolean) => void;
  onFilesSelected: (files: File[]) => void;
  onRemoveFile: (filename: string) => void;
  onStartAnalysis: () => void;
}

export const UploadArea: React.FC<UploadAreaProps> = ({
  stagedFiles,
  isAnalyzing,
  statusMessage,
  k,
  noLlm,
  onKChange,
  onNoLlmChange,
  onFilesSelected,
  onRemoveFile,
  onStartAnalysis,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFilesSelected(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesSelected(Array.from(e.target.files));
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const validFiles = stagedFiles.filter((f) => f.valid);
  const invalidFiles = stagedFiles.filter((f) => !f.valid);

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const kUnits = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(kUnits));
    return parseFloat((bytes / Math.pow(kUnits, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <section className="upload-container" aria-label="Paper Upload Area">
      <div
        className={`dropzone ${isDragging ? 'dragging' : ''} ${isAnalyzing ? 'disabled' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isAnalyzing && fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload PDF papers dropzone"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          multiple
          style={{ display: 'none' }}
          onChange={handleFileInputChange}
          disabled={isAnalyzing}
        />
        <div className="dropzone-icon">
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="12" y1="18" x2="12" y2="12"></line>
            <line x1="9" y1="15" x2="15" y2="15"></line>
          </svg>
        </div>
        <div className="dropzone-text">
          <h3>Drop PDF research papers here, or <span className="browse-link">browse</span></h3>
          <p>Supports single or multiple PDF documents. Validates magic bytes and preserves persistent collections.</p>
        </div>
      </div>

      {invalidFiles.length > 0 && (
        <div className="alert-box error" role="alert">
          <h4>Validation Errors ({invalidFiles.length})</h4>
          <ul className="error-list">
            {invalidFiles.map((f, i) => (
              <li key={i}>
                <strong>{f.filename}</strong>: {f.error || 'Invalid PDF'}
              </li>
            ))}
          </ul>
        </div>
      )}

      {validFiles.length > 0 && (
        <div className="staged-list-container">
          <div className="staged-header">
            <h4>Staged Papers ({validFiles.length})</h4>
            <span className="staged-hint">Ready for ingestion and evidence analysis</span>
          </div>
          <div className="staged-grid">
            {validFiles.map((file, idx) => (
              <div key={idx} className="staged-card">
                <div className="staged-card-icon">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                  </svg>
                </div>
                <div className="staged-card-info">
                  <span className="file-name" title={file.filename}>{file.filename}</span>
                  <span className="file-size">{formatBytes(file.size_bytes)}</span>
                </div>
                {!isAnalyzing && (
                  <button
                    type="button"
                    className="remove-file-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemoveFile(file.filename);
                    }}
                    title={`Remove ${file.filename}`}
                    aria-label={`Remove ${file.filename}`}
                  >
                    ✕
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="action-bar">
        <div className="controls-group">
          <label className="param-label" htmlFor="retrieval-k-select">
            Retrieval K:
            <select
              id="retrieval-k-select"
              value={k}
              onChange={(e) => onKChange(parseInt(e.target.value, 10))}
              disabled={isAnalyzing}
              className="select-input"
            >
              <option value={2}>2 chunks</option>
              <option value={4}>4 chunks (default)</option>
              <option value={6}>6 chunks</option>
              <option value={8}>8 chunks</option>
            </select>
          </label>

          <label className="checkbox-label" title="Runs full hybrid retrieval & diagnostics without LLM token inference">
            <input
              type="checkbox"
              checked={noLlm}
              onChange={(e) => onNoLlmChange(e.target.checked)}
              disabled={isAnalyzing}
            />
            <span>Fast Diagnostic / No-LLM Mode</span>
          </label>
        </div>

        <button
          id="analyze-papers-btn"
          type="button"
          className="analyze-btn"
          disabled={isAnalyzing || validFiles.length === 0}
          onClick={onStartAnalysis}
        >
          {isAnalyzing ? (
            <>
              <span className="spinner"></span>
              Analyzing Pipeline...
            </>
          ) : (
            <>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
              </svg>
              Analyze {validFiles.length} {validFiles.length === 1 ? 'Paper' : 'Papers'}
            </>
          )}
        </button>
      </div>

      {isAnalyzing && (
        <div className="status-progress-banner" aria-live="polite">
          <div className="status-indicator">
            <span className="pulse-dot"></span>
            <span>{statusMessage || 'Processing documents through RAGSS pipeline...'}</span>
          </div>
          <p className="status-note">
            Ingestion, parent-child expansion, and reciprocal rank fusion running with isolated persistence.
          </p>
        </div>
      )}
    </section>
  );
};
