import type {
  AnalyzeResponse,
  ConfigInfoResponse,
  CrossPaperAnalysisItem,
  StagedFilesResponse,
} from './types';

const API_BASE = '/api';

export async function fetchConfig(): Promise<ConfigInfoResponse> {
  const res = await fetch(`${API_BASE}/config`);
  if (!res.ok) {
    throw new Error(`Failed to fetch configuration: ${res.statusText}`);
  }
  return res.json();
}

export async function uploadPapers(
  files: File[],
  sessionId?: string
): Promise<StagedFilesResponse> {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }
  if (sessionId) {
    formData.append('session_id', sessionId);
  }

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || 'Upload failed');
  }

  return res.json();
}

export async function fetchStagedFiles(sessionId: string): Promise<StagedFilesResponse> {
  const res = await fetch(`${API_BASE}/uploaded?session_id=${encodeURIComponent(sessionId)}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch staged papers: ${res.statusText}`);
  }
  return res.json();
}

export async function removeStagedFile(
  filename: string,
  sessionId: string
): Promise<StagedFilesResponse> {
  const res = await fetch(
    `${API_BASE}/upload/${encodeURIComponent(filename)}?session_id=${encodeURIComponent(sessionId)}`,
    {
      method: 'DELETE',
    }
  );

  if (!res.ok) {
    throw new Error(`Failed to remove file: ${res.statusText}`);
  }
  return res.json();
}

export async function analyzePapers(
  sessionId: string,
  k = 4,
  noLlm = false
): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      k,
      no_llm: noLlm,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || `Analysis failed (${res.status})`);
  }

  return res.json();
}

export async function runCrossAnalysis(
  sessionId: string,
  noLlm = false
): Promise<CrossPaperAnalysisItem> {
  const res = await fetch(`${API_BASE}/cross-analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      no_llm: noLlm,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || `Cross-analysis failed (${res.status})`);
  }

  return res.json();
}
