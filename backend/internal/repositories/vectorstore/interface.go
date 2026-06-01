package vectorstore

import "rag-backend/pkg/types"

// VectorStore defines the interface for vector storage operations
type VectorStore interface {
	Store(chunks []types.DocumentChunk) error
	Search(embedding []float64, limit int) ([]types.ScoredChunk, error)
	// Neighbors returns the chunks of the given source whose ChunkIndex falls
	// within [index-radius, index+radius] (inclusive of the center). Filtering
	// on source keeps context expansion from crossing document boundaries.
	Neighbors(source string, index, radius int) ([]types.DocumentChunk, error)
}
