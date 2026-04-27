package similarity

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"

	"rag-backend/pkg/types"
)

func makeChunk(id string, embedding []float64) types.DocumentChunk {
	return types.DocumentChunk{
		ID:        id,
		Content:   "content-" + id,
		Embedding: embedding,
	}
}

func TestSearch(t *testing.T) {
	type expected struct {
		ids    []string
		scores []float64
		err    string
	}

	tests := []struct {
		name      string
		embedding []float64
		chunks    []types.DocumentChunk
		limit     int
		expected  expected
	}{
		{
			name:      "returns empty slice when no chunks are provided",
			embedding: []float64{1, 0, 0},
			chunks:    []types.DocumentChunk{},
			limit:     5,
			expected:  expected{ids: []string{}},
		},
		{
			name:      "returns empty slice when chunks slice is nil",
			embedding: []float64{1, 0, 0},
			chunks:    nil,
			limit:     5,
			expected:  expected{ids: []string{}},
		},
		{
			name:      "skips chunks with empty embeddings",
			embedding: []float64{1, 0, 0},
			chunks: []types.DocumentChunk{
				makeChunk("a", []float64{1, 0, 0}),
				makeChunk("b", []float64{}),
				makeChunk("c", nil),
				makeChunk("d", []float64{0, 1, 0}),
			},
			limit: 10,
			expected: expected{
				ids:    []string{"a", "d"},
				scores: []float64{1.0, 0.0},
			},
		},
		{
			name:      "returns empty slice when all chunks have empty embeddings",
			embedding: []float64{1, 0, 0},
			chunks: []types.DocumentChunk{
				makeChunk("a", []float64{}),
				makeChunk("b", nil),
			},
			limit:    5,
			expected: expected{ids: []string{}},
		},
		{
			name:      "sorts results by score in descending order",
			embedding: []float64{1, 0, 0},
			chunks: []types.DocumentChunk{
				makeChunk("low", []float64{0, 1, 0}),
				makeChunk("high", []float64{1, 0, 0}),
				makeChunk("mid", []float64{1, 1, 0}),
			},
			limit: 3,
			expected: expected{
				ids:    []string{"high", "mid", "low"},
				scores: []float64{1.0, 1.0 / math.Sqrt(2), 0.0},
			},
		},
		{
			name:      "truncates results to limit when limit is smaller than chunk count",
			embedding: []float64{1, 0, 0},
			chunks: []types.DocumentChunk{
				makeChunk("a", []float64{1, 0, 0}),
				makeChunk("b", []float64{1, 1, 0}),
				makeChunk("c", []float64{0, 1, 0}),
				makeChunk("d", []float64{-1, 0, 0}),
			},
			limit: 2,
			expected: expected{
				ids:    []string{"a", "b"},
				scores: []float64{1.0, 1.0 / math.Sqrt(2)},
			},
		},
		{
			name:      "returns all results when limit exceeds chunk count",
			embedding: []float64{1, 0, 0},
			chunks: []types.DocumentChunk{
				makeChunk("a", []float64{1, 0, 0}),
				makeChunk("b", []float64{0, 1, 0}),
			},
			limit: 100,
			expected: expected{
				ids:    []string{"a", "b"},
				scores: []float64{1.0, 0.0},
			},
		},
		{
			name:      "returns empty slice when limit is zero",
			embedding: []float64{1, 0, 0},
			chunks: []types.DocumentChunk{
				makeChunk("a", []float64{1, 0, 0}),
			},
			limit:    0,
			expected: expected{ids: []string{}},
		},
		{
			name:      "skips dimension-mismatched chunks by scoring them as zero",
			embedding: []float64{1, 0, 0},
			chunks: []types.DocumentChunk{
				makeChunk("match", []float64{1, 0, 0}),
				makeChunk("mismatch", []float64{1, 0}),
			},
			limit: 2,
			expected: expected{
				ids:    []string{"match", "mismatch"},
				scores: []float64{1.0, 0.0},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Search(tt.embedding, tt.chunks, tt.limit)

			if tt.expected.err != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expected.err)
				return
			}

			assert.NoError(t, err)
			assert.Len(t, result, len(tt.expected.ids))

			for i, sc := range result {
				assert.Equal(t, tt.expected.ids[i], sc.Chunk.ID, "chunk at index %d should have correct ID", i)
				assert.InDelta(t, tt.expected.scores[i], sc.Score, 1e-9, "chunk at index %d should have correct score", i)
			}
		})
	}
}

func TestSearch_PreservesChunkData(t *testing.T) {
	chunk := types.DocumentChunk{
		ID:        "doc-1",
		Content:   "the quick brown fox",
		Embedding: []float64{1, 0, 0},
		Metadata:  map[string]string{"source": "test.txt"},
	}

	result, err := Search([]float64{1, 0, 0}, []types.DocumentChunk{chunk}, 1)

	assert.NoError(t, err)
	assert.Len(t, result, 1)
	assert.Equal(t, chunk.ID, result[0].Chunk.ID)
	assert.Equal(t, chunk.Content, result[0].Chunk.Content)
	assert.Equal(t, chunk.Embedding, result[0].Chunk.Embedding)
	assert.Equal(t, chunk.Metadata, result[0].Chunk.Metadata)
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a        []float64
		b        []float64
		expected float64
	}{
		{
			name:     "identical unit vectors return 1",
			a:        []float64{1, 0, 0},
			b:        []float64{1, 0, 0},
			expected: 1.0,
		},
		{
			name:     "identical non-unit vectors return 1",
			a:        []float64{3, 4, 0},
			b:        []float64{3, 4, 0},
			expected: 1.0,
		},
		{
			name:     "parallel vectors of different magnitudes return 1",
			a:        []float64{1, 2, 3},
			b:        []float64{2, 4, 6},
			expected: 1.0,
		},
		{
			name:     "orthogonal vectors return 0",
			a:        []float64{1, 0, 0},
			b:        []float64{0, 1, 0},
			expected: 0.0,
		},
		{
			name:     "opposite vectors return -1",
			a:        []float64{1, 0, 0},
			b:        []float64{-1, 0, 0},
			expected: -1.0,
		},
		{
			name:     "vectors at 45 degrees return cos(45)",
			a:        []float64{1, 0},
			b:        []float64{1, 1},
			expected: 1.0 / math.Sqrt(2),
		},
		{
			name:     "negative components are handled correctly",
			a:        []float64{1, -1, 1},
			b:        []float64{-1, 1, -1},
			expected: -1.0,
		},
		{
			name:     "different lengths return 0",
			a:        []float64{1, 2, 3},
			b:        []float64{1, 2},
			expected: 0.0,
		},
		{
			name:     "first vector longer than second returns 0",
			a:        []float64{1, 2, 3, 4},
			b:        []float64{1, 2, 3},
			expected: 0.0,
		},
		{
			name:     "first vector empty and second non-empty return 0",
			a:        []float64{},
			b:        []float64{1, 2, 3},
			expected: 0.0,
		},
		{
			name:     "both vectors empty return 0",
			a:        []float64{},
			b:        []float64{},
			expected: 0.0,
		},
		{
			name:     "first vector all zeros returns 0",
			a:        []float64{0, 0, 0},
			b:        []float64{1, 2, 3},
			expected: 0.0,
		},
		{
			name:     "second vector all zeros returns 0",
			a:        []float64{1, 2, 3},
			b:        []float64{0, 0, 0},
			expected: 0.0,
		},
		{
			name:     "both vectors all zeros return 0",
			a:        []float64{0, 0, 0},
			b:        []float64{0, 0, 0},
			expected: 0.0,
		},
		{
			name:     "single dimension vectors compute correctly",
			a:        []float64{5},
			b:        []float64{2},
			expected: 1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cosineSimilarity(tt.a, tt.b)
			assert.InDelta(t, tt.expected, result, 1e-9)
		})
	}
}

func TestCosineSimilarity_IsSymmetric(t *testing.T) {
	a := []float64{0.1, 0.5, -0.3, 0.8}
	b := []float64{-0.2, 0.4, 0.7, 0.1}

	assert.InDelta(t, cosineSimilarity(a, b), cosineSimilarity(b, a), 1e-12)
}
