package types

import "time"

type Document struct {
	ID         string          `json:"id"`
	Name       string          `json:"name"`
	Content    string          `json:"content"`
	Chunks     []DocumentChunk `json:"chunks"`
	UploadedAt time.Time       `json:"uploadedAt"`
}

type DocumentChunk struct {
	ID        string            `json:"id"`
	Content   string            `json:"content"`
	Embedding []float64         `json:"embedding,omitempty"`
	Metadata  map[string]string `json:"metadata"`
}

type RAGResponse struct {
	Answer     string          `json:"answer"`
	Sources    []DocumentChunk `json:"sources"`
	Confidence float64         `json:"confidence"`
}

type UploadResponse struct {
	Success  bool                   `json:"success"`
	Document *UploadDocumentSummary `json:"document,omitempty"`
	Error    string                 `json:"error,omitempty"`
}

type UploadDocumentSummary struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	ChunksCount int       `json:"chunksCount"`
	UploadedAt  time.Time `json:"uploadedAt"`
}

type QueryResponse struct {
	Success    bool            `json:"success"`
	Answer     string          `json:"answer,omitempty"`
	Sources    []DocumentChunk `json:"sources,omitempty"`
	Confidence float64         `json:"confidence,omitempty"`
	Error      string          `json:"error,omitempty"`
}

type QueryRequest struct {
	Question string `json:"question" binding:"required"`
}

type HealthResponse struct {
	Status    string    `json:"status"`
	Timestamp time.Time `json:"timestamp"`
}
