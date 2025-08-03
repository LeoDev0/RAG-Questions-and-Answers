package memory

import (
	"math"
	"rag-backend/internal/repositories/vectorstore"
	"sort"
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

func (mvs *MemoryVectorStore) SimilaritySearch(embedding []float64, limit int) ([]types.ScoredChunk, error) {
	mvs.mutex.RLock()
	defer mvs.mutex.RUnlock()

	if len(mvs.documents) == 0 {
		return []types.ScoredChunk{}, nil
	}

	scored := make([]types.ScoredChunk, 0, len(mvs.documents))
	for _, chunk := range mvs.documents {
		if len(chunk.Embedding) == 0 {
			continue
		}
		score := cosineSimilarity(embedding, chunk.Embedding)
		scored = append(scored, types.ScoredChunk{Chunk: chunk, Score: score})
	}

	// Sort by score (descending)
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score > scored[j].Score
	})

	// Return top k results
	k := min(limit, len(scored))
	return scored[:k], nil
}

// cosineSimilarity calculates cosine similarity between two vectors
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0.0 || normB == 0.0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
