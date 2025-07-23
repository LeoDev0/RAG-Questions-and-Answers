export interface Document {
  id: string;
  name: string;
  content: string;
  chunks: DocumentChunk[];
  uploadedAt: Date;
}

export interface DocumentChunk {
  id: string;
  content: string;
  embedding?: number[];
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
  success: boolean;
  document?: {
    id: string;
    name: string;
    chunksCount: number;
    uploadedAt: Date;
  };
  error?: string;
}

export interface QueryResponse {
  success: boolean;
  answer?: string;
  sources?: DocumentChunk[];
  confidence?: number;
  error?: string;
}