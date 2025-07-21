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