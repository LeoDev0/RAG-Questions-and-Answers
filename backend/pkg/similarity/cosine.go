package similarity

import (
	"math"
	"rag-backend/pkg/types"
	"sort"
)

// Search finds the most similar document chunks to the given embedding
func Search(embedding []float64, chunks []types.DocumentChunk, limit int) ([]types.ScoredChunk, error) {
	if len(chunks) == 0 {
		return []types.ScoredChunk{}, nil
	}

	scored := make([]types.ScoredChunk, 0, len(chunks))
	for _, chunk := range chunks {
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
