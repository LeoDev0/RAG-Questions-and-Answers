package memory

import (
	"rag-backend/internal/repositories/vectorstore"
	"rag-backend/pkg/similarity"
	"sync"

	"rag-backend/pkg/types"
)

type MemoryVectorStore struct {
	documents []types.DocumentChunk
	mutex     sync.RWMutex
}

func NewMemoryVectorStore() vectorstore.VectorStore {
	return &MemoryVectorStore{
		documents: make([]types.DocumentChunk, 0),
	}
}

func (mvs *MemoryVectorStore) Store(chunks []types.DocumentChunk) error {
	mvs.mutex.Lock()
	defer mvs.mutex.Unlock()
	mvs.documents = append(mvs.documents, chunks...)
	return nil
}

func (mvs *MemoryVectorStore) Search(embedding []float64, limit int) ([]types.ScoredChunk, error) {
	mvs.mutex.RLock()
	defer mvs.mutex.RUnlock()
	return similarity.Search(embedding, mvs.documents, limit)
}
