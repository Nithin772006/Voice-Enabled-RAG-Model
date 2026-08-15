import { RAGResponse, HealthResponse, ReadinessResponse } from '../types/api';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Checks liveness of the FastAPI backend.
 */
export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
    cache: 'no-store'
  });
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.statusText}`);
  }
  return res.json();
}

/**
 * Checks readiness of the FastAPI backend dependencies.
 */
export async function getReadiness(): Promise<ReadinessResponse> {
  const res = await fetch(`${API_BASE}/health/ready`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
    cache: 'no-store'
  });
  if (!res.ok) {
    throw new Error(`Readiness check failed: ${res.status}`);
  }
  return res.json();
}

/**
 * Queries the RAG + Voice pipeline using the compiled LangGraph workflow.
 * Supports request cancellation via AbortSignal.
 */
export async function queryRAG(
  query: string,
  language: string,
  signal?: AbortSignal
): Promise<RAGResponse> {
  const res = await fetch(`${API_BASE}/api/v1/rag/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, language }),
    signal,
    cache: 'no-store'
  });

  if (!res.ok) {
    let errDetail = 'An unexpected error occurred on the server.';
    try {
      const errJSON = await res.json();
      errDetail = errJSON.detail || errDetail;
    } catch {
      // Ignored if response is not JSON
    }
    throw new Error(errDetail);
  }

  return res.json();
}

/**
 * Safely resolves the backend-returned relative URL into an absolute URL for streaming.
 */
export function getAudioSourceUrl(relativeUrl: string): string {
  if (!relativeUrl) return '';
  // Ensure we don't duplicate slash separators
  const base = API_BASE.endsWith('/') ? API_BASE.slice(0, -1) : API_BASE;
  const path = relativeUrl.startsWith('/') ? relativeUrl : `/${relativeUrl}`;
  return `${base}${path}`;
}
