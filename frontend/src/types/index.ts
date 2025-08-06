export interface DocumentChunk {
  id: string;
  content: string;
  metadata: {
    page?: number;
    source: string;
  };
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: DocumentChunk[];
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