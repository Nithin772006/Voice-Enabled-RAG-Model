export interface SourceMetadata {
  chunk_id: string;
  score?: number;
  language?: string;
  text?: string;
}

export interface RAGResponse {
  status: 'success' | 'fallback' | 'error';
  query: string;
  language: string;
  answer: string;
  sources: SourceMetadata[];
  audio_url: string | null;
  error: string | null;
}

export interface HealthResponse {
  status: string;
}

export interface ReadinessResponse {
  status: string;
  qdrant: string;
  graph: string;
}
