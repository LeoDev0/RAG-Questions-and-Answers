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
  success: boolean;
  answer?: string;
  sources?: DocumentChunk[];
  confidence?: number;
  error?: string;
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