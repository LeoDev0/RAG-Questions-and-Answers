package vectorstore

import "rag-backend/pkg/types"

// VectorStore defines the interface for vector storage operations
type VectorStore interface {
	// Store adds a document chunk with its embedding to the vector store
	Store(chunks []types.DocumentChunk) error

	// SimilaritySearch finds the most similar document chunks to the given embedding
	SimilaritySearch(embedding []float64, limit int) ([]types.ScoredChunk, error)
}
