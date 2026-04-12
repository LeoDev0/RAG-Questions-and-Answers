package vectorstore

import "rag-backend/pkg/types"

type MockVectorStore struct {
	StoreFunc  func(chunks []types.DocumentChunk) error
	SearchFunc func(embedding []float64, limit int) ([]types.ScoredChunk, error)
}

func (m *MockVectorStore) Store(chunks []types.DocumentChunk) error {
	return m.StoreFunc(chunks)
}

func (m *MockVectorStore) Search(embedding []float64, limit int) ([]types.ScoredChunk, error) {
	return m.SearchFunc(embedding, limit)
}
