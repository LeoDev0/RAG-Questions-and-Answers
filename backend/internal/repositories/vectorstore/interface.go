package vectorstore

import "rag-backend/pkg/types"

// VectorStore defines the interface for vector storage operations
type VectorStore interface {
	Store(chunks []types.DocumentChunk) error
	Search(embedding []float64, limit int) ([]types.ScoredChunk, error)
}
