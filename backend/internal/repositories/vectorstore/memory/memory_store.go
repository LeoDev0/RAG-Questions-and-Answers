package memory

import (
	"rag-backend/internal/repositories/vectorstore"
	"rag-backend/pkg/similarity"
	"sync"

	"rag-backend/pkg/types"
)

type MemoryVectorStore struct {
	documents []types.DocumentChunk
	bySource  map[string][]types.DocumentChunk
	mutex     sync.RWMutex
}

func NewMemoryVectorStore() vectorstore.VectorStore {
	return &MemoryVectorStore{
		documents: make([]types.DocumentChunk, 0),
		bySource:  make(map[string][]types.DocumentChunk),
	}
}

func (mvs *MemoryVectorStore) Store(chunks []types.DocumentChunk) error {
	mvs.mutex.Lock()
	defer mvs.mutex.Unlock()
	mvs.documents = append(mvs.documents, chunks...)
	for _, chunk := range chunks {
		mvs.bySource[chunk.Source] = append(mvs.bySource[chunk.Source], chunk)
	}
	return nil
}

func (mvs *MemoryVectorStore) Search(embedding []float64, limit int) ([]types.ScoredChunk, error) {
	mvs.mutex.RLock()
	defer mvs.mutex.RUnlock()
	return similarity.Search(embedding, mvs.documents, limit)
}

func (mvs *MemoryVectorStore) Neighbors(source string, index, radius int) ([]types.DocumentChunk, error) {
	mvs.mutex.RLock()
	defer mvs.mutex.RUnlock()

	candidates := mvs.bySource[source]
	lo, hi := index-radius, index+radius
	neighbors := make([]types.DocumentChunk, 0, len(candidates))
	for _, chunk := range candidates {
		if chunk.ChunkIndex >= lo && chunk.ChunkIndex <= hi {
			neighbors = append(neighbors, chunk)
		}
	}
	return neighbors, nil
}
