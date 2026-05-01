export interface DocumentChunk {
  id: string;
  content: string;
  metadata: {
    page?: number;
    source: string;
  };
}

export type ChatRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  timestamp: Date;
  sources?: DocumentChunk[];
}

export interface ApiMessage {
  role: ChatRole;
  content: string;
}

export interface RAGResponse {
  answer: string;
  sources: DocumentChunk[];
  confidence: number;
}

export interface UploadResponse {
  document: {
    id: string;
    name: string;
    chunksCount: number;
    uploadedAt: Date;
  };
}

export interface ErrorResponse {
  error: string;
  code?: string;
  details?: string;
}

export type StreamEvent =
  | { type: 'sources'; sources: DocumentChunk[]; confidence: number }
  | { type: 'token'; content: string }
  | { type: 'done' }
  | { type: 'error'; error: string; code?: string };