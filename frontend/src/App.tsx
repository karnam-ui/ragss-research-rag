import { useEffect, useState } from 'react';
import {
  analyzePapers,
  fetchConfig,
  fetchStagedFiles,
  removeStagedFile,
  uploadPapers,
} from './api';
import { DeveloperMode } from './components/DeveloperMode';
import { Navbar } from './components/Navbar';
import { ResearchMode } from './components/ResearchMode';
import { UploadArea } from './components/UploadArea';
import type {
  ConfigInfoResponse,
  CrossPaperAnalysisItem,
  PaperAnalysisItem,
  UploadFileResponse,
} from './types';

export function App() {
  const [mode, setMode] = useState<'research' | 'developer'>('research');
  const [sessionId, setSessionId] = useState<string>(() => {
    const saved = localStorage.getItem('ragss_session_id');
    if (saved) return saved;
    const generated = Math.random().toString(36).substring(2, 14);
    localStorage.setItem('ragss_session_id', generated);
    return generated;
  });

  const [config, setConfig] = useState<ConfigInfoResponse | null>(null);
  const [stagedFiles, setStagedFiles] = useState<UploadFileResponse[]>([]);
  const [papers, setPapers] = useState<PaperAnalysisItem[]>([]);
  const [crossAnalysis, setCrossAnalysis] = useState<CrossPaperAnalysisItem | null>(null);
  const [selectedPaperId, setSelectedPaperId] = useState<string | null>(null);

  const [k, setK] = useState<number>(4);
  const [noLlm, setNoLlm] = useState<boolean>(false);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [globalError, setGlobalError] = useState<string | null>(null);

  // Load config and staged files on mount
  useEffect(() => {
    fetchConfig()
      .then((cfg) => setConfig(cfg))
      .catch((err) => console.warn('Could not load config:', err));

    if (sessionId) {
      fetchStagedFiles(sessionId)
        .then((res) => setStagedFiles(res.files))
        .catch(() => {});
    }
  }, [sessionId]);

  const handleFilesSelected = async (files: File[]) => {
    setGlobalError(null);
    try {
      const res = await uploadPapers(files, sessionId);
      setSessionId(res.session_id);
      localStorage.setItem('ragss_session_id', res.session_id);
      setStagedFiles((prev) => {
        // Merge uniquely by filename
        const existingNames = new Set(prev.map((f) => f.filename));
        const newValidOrInvalid = res.files.filter((f) => !existingNames.has(f.filename));
        return [...prev, ...newValidOrInvalid];
      });
    } catch (err: any) {
      setGlobalError(err.message || 'File upload failed');
    }
  };

  const handleRemoveFile = async (filename: string) => {
    try {
      const res = await removeStagedFile(filename, sessionId);
      setStagedFiles(res.files);
    } catch (err: any) {
      setGlobalError(err.message || 'Failed to remove file');
    }
  };

  const handleStartAnalysis = async () => {
    setIsAnalyzing(true);
    setGlobalError(null);
    setStatusMessage('Indexing PDF pages into isolated session collection...');

    try {
      // Step update message
      const timer = setTimeout(() => {
        setStatusMessage('Running hybrid dense + BM25 retrieval and parent expansion...');
      }, 1800);

      const res = await analyzePapers(sessionId, k, noLlm);
      clearTimeout(timer);

      setPapers(res.papers);
      setCrossAnalysis(res.cross_analysis || null);
      if (res.papers.length > 0 && !selectedPaperId) {
        setSelectedPaperId(res.papers[0].document_id);
      }
    } catch (err: any) {
      setGlobalError(err.message || 'Analysis failed. Check backend logs.');
    } finally {
      setIsAnalyzing(false);
      setStatusMessage('');
    }
  };

  const handleSelectPaperForDevMode = (docId: string) => {
    setSelectedPaperId(docId);
    setMode('developer');
  };

  return (
    <div className="app-shell">
      <Navbar mode={mode} onModeChange={setMode} config={config} />

      <main className="main-content">
        {globalError && (
          <div className="global-error-banner" role="alert">
            <div className="error-title">
              <strong>Error:</strong> {globalError}
            </div>
            <button className="dismiss-btn" onClick={() => setGlobalError(null)}>
              ✕
            </button>
          </div>
        )}

        {/* 1. PDF Upload Area (always accessible at top) */}
        <UploadArea
          stagedFiles={stagedFiles}
          isAnalyzing={isAnalyzing}
          statusMessage={statusMessage}
          k={k}
          noLlm={noLlm}
          onKChange={setK}
          onNoLlmChange={setNoLlm}
          onFilesSelected={handleFilesSelected}
          onRemoveFile={handleRemoveFile}
          onStartAnalysis={handleStartAnalysis}
        />

        {/* 2. Primary Content by Mode */}
        {mode === 'research' ? (
          <ResearchMode
            papers={papers}
            crossAnalysis={crossAnalysis}
            onSelectPaperForDevMode={handleSelectPaperForDevMode}
          />
        ) : (
          <DeveloperMode
            papers={papers}
            selectedPaperId={selectedPaperId}
            onSelectPaper={setSelectedPaperId}
          />
        )}
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <span>PaperTrail / RAGSS — Verified Grounded Retrieval for Scientific Literature</span>
          <span className="footer-sep">•</span>
          <span>BGE-Small Dense + BM25 Lexical + RRF Fusion + Cross-Encoder Reranking</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
