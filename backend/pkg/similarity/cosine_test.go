package similarity

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"

	"rag-backend/pkg/types"
)

func TestCosineSimilarity(t *testing.T) {
	type expected struct {
		score float64
	}

	tests := []struct {
		name     string
		a        []float64
		b        []float64
		expected expected
	}{
		{
			name:     "identical vectors return 1",
			a:        []float64{1, 2, 3},
			b:        []float64{1, 2, 3},
			expected: expected{score: 1.0},
		},
		{
			name:     "parallel vectors with different magnitudes return 1",
			a:        []float64{1, 2, 3},
			b:        []float64{2, 4, 6},
			expected: expected{score: 1.0},
		},
		{
			name:     "orthogonal vectors return 0",
			a:        []float64{1, 0},
			b:        []float64{0, 1},
			expected: expected{score: 0.0},
		},
		{
			name:     "opposite vectors return -1",
			a:        []float64{1, 2, 3},
			b:        []float64{-1, -2, -3},
			expected: expected{score: -1.0},
		},
		{
			name:     "computes intermediate similarity correctly",
			a:        []float64{1, 0, 0},
			b:        []float64{1, 1, 0},
			expected: expected{score: 1.0 / math.Sqrt(2)},
		},
		{
			name:     "mismatched vector lengths return 0",
			a:        []float64{1, 2, 3},
			b:        []float64{1, 2},
			expected: expected{score: 0.0},
		},
		{
			name:     "first vector is zero returns 0",
			a:        []float64{0, 0, 0},
			b:        []float64{1, 2, 3},
			expected: expected{score: 0.0},
		},
		{
			name:     "second vector is zero returns 0",
			a:        []float64{1, 2, 3},
			b:        []float64{0, 0, 0},
			expected: expected{score: 0.0},
		},
		{
			name:     "both vectors are zero returns 0",
			a:        []float64{0, 0, 0},
			b:        []float64{0, 0, 0},
			expected: expected{score: 0.0},
		},
		{
			name:     "empty vectors return 0 due to zero norm",
			a:        []float64{},
			b:        []float64{},
			expected: expected{score: 0.0},
		},
		{
			name:     "handles negative components correctly",
			a:        []float64{1, -1},
			b:        []float64{1, -1},
			expected: expected{score: 1.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cosineSimilarity(tt.a, tt.b)
			assert.InDelta(t, tt.expected.score, got, 1e-9)
		})
	}
}

func TestSearch(t *testing.T) {
	chunkA := types.DocumentChunk{ID: "a", Content: "alpha", Embedding: []float64{1, 0, 0}}
	chunkB := types.DocumentChunk{ID: "b", Content: "beta", Embedding: []float64{0, 1, 0}}
	chunkC := types.DocumentChunk{ID: "c", Content: "gamma", Embedding: []float64{1, 1, 0}}
	chunkEmptyEmbedding := types.DocumentChunk{ID: "empty", Content: "no embedding", Embedding: []float64{}}
	chunkNilEmbedding := types.DocumentChunk{ID: "nil", Content: "nil embedding"}

	type expected struct {
		ids []string
	}

	tests := []struct {
		name      string
		embedding []float64
		chunks    []types.DocumentChunk
		limit     int
		expected  expected
	}{
		{
			name:      "empty chunks slice returns empty result",
			embedding: []float64{1, 0, 0},
			chunks:    []types.DocumentChunk{},
			limit:     5,
			expected:  expected{ids: []string{}},
		},
		{
			name:      "nil chunks slice returns empty result",
			embedding: []float64{1, 0, 0},
			chunks:    nil,
			limit:     5,
			expected:  expected{ids: []string{}},
		},
		{
			name:      "single chunk is returned",
			embedding: []float64{1, 0, 0},
			chunks:    []types.DocumentChunk{chunkA},
			limit:     5,
			expected:  expected{ids: []string{"a"}},
		},
		{
			name:      "results are sorted by similarity descending",
			embedding: []float64{1, 0, 0},
			chunks:    []types.DocumentChunk{chunkB, chunkC, chunkA},
			limit:     5,
			expected:  expected{ids: []string{"a", "c", "b"}},
		},
		{
			name:      "limit caps the number of returned chunks to top matches",
			embedding: []float64{1, 0, 0},
			chunks:    []types.DocumentChunk{chunkA, chunkB, chunkC},
			limit:     2,
			expected:  expected{ids: []string{"a", "c"}},
		},
		{
			name:      "limit equal to chunk count returns all chunks",
			embedding: []float64{1, 0, 0},
			chunks:    []types.DocumentChunk{chunkA, chunkB},
			limit:     2,
			expected:  expected{ids: []string{"a", "b"}},
		},
		{
			name:      "limit larger than chunk count returns all available chunks",
			embedding: []float64{1, 0, 0},
			chunks:    []types.DocumentChunk{chunkA, chunkB},
			limit:     10,
			expected:  expected{ids: []string{"a", "b"}},
		},
		{
			name:      "limit of zero returns empty result",
			embedding: []float64{1, 0, 0},
			chunks:    []types.DocumentChunk{chunkA, chunkB},
			limit:     0,
			expected:  expected{ids: []string{}},
		},
		{
			name:      "skips chunks with empty embedding slice",
			embedding: []float64{1, 0, 0},
			chunks:    []types.DocumentChunk{chunkA, chunkEmptyEmbedding, chunkB},
			limit:     5,
			expected:  expected{ids: []string{"a", "b"}},
		},
		{
			name:      "skips chunks with nil embedding",
			embedding: []float64{1, 0, 0},
			chunks:    []types.DocumentChunk{chunkNilEmbedding, chunkA},
			limit:     5,
			expected:  expected{ids: []string{"a"}},
		},
		{
			name:      "all chunks missing embeddings produces empty result",
			embedding: []float64{1, 0, 0},
			chunks:    []types.DocumentChunk{chunkEmptyEmbedding, chunkNilEmbedding},
			limit:     5,
			expected:  expected{ids: []string{}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Search(tt.embedding, tt.chunks, tt.limit)

			assert.NoError(t, err)
			ids := make([]string, len(result))
			for i, sc := range result {
				ids[i] = sc.Chunk.ID
			}
			assert.Equal(t, tt.expected.ids, ids)
		})
	}
}

func TestSearch_PopulatesScoresAndPreservesChunkData(t *testing.T) {
	chunks := []types.DocumentChunk{
		{ID: "a", Content: "alpha", Embedding: []float64{1, 0, 0}, Metadata: map[string]string{"source": "doc1"}},
		{ID: "b", Content: "beta", Embedding: []float64{0, 1, 0}, Metadata: map[string]string{"source": "doc2"}},
		{ID: "c", Content: "gamma", Embedding: []float64{1, 1, 0}, Metadata: map[string]string{"source": "doc3"}},
	}

	result, err := Search([]float64{1, 0, 0}, chunks, 3)

	assert.NoError(t, err)
	assert.Len(t, result, 3)

	assert.Equal(t, "a", result[0].Chunk.ID)
	assert.Equal(t, "alpha", result[0].Chunk.Content)
	assert.Equal(t, map[string]string{"source": "doc1"}, result[0].Chunk.Metadata)
	assert.InDelta(t, 1.0, result[0].Score, 1e-9)

	assert.Equal(t, "c", result[1].Chunk.ID)
	assert.InDelta(t, 1.0/math.Sqrt(2), result[1].Score, 1e-9)

	assert.Equal(t, "b", result[2].Chunk.ID)
	assert.InDelta(t, 0.0, result[2].Score, 1e-9)
}
