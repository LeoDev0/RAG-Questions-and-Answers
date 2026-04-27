package similarity

import (
	"rag-backend/pkg/types"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSearch(t *testing.T) {
	type input struct {
		embedding []float64
		chunks    []types.DocumentChunk
		limit     int
	}

	type expected struct {
		ids []string
	}

	tests := []struct {
		name     string
		input    input
		expected expected
	}{
		{
			name: "returns empty slice when chunks list is empty",
			input: input{
				embedding: []float64{1, 0, 0},
				chunks:    []types.DocumentChunk{},
				limit:     5,
			},
			expected: expected{ids: []string{}},
		},
		{
			name: "skips chunks with empty embedding",
			input: input{
				embedding: []float64{1, 0, 0},
				chunks: []types.DocumentChunk{
					{ID: "with-embedding", Embedding: []float64{1, 0, 0}},
					{ID: "no-embedding"},
				},
				limit: 5,
			},
			expected: expected{ids: []string{"with-embedding"}},
		},
		{
			name: "orders results by descending similarity score",
			input: input{
				embedding: []float64{1, 0, 0},
				chunks: []types.DocumentChunk{
					{ID: "orthogonal", Embedding: []float64{0, 1, 0}},
					{ID: "identical", Embedding: []float64{1, 0, 0}},
					{ID: "similar", Embedding: []float64{1, 1, 0}},
				},
				limit: 5,
			},
			expected: expected{ids: []string{"identical", "similar", "orthogonal"}},
		},
		{
			name: "caps results when limit is smaller than scored count",
			input: input{
				embedding: []float64{1, 0, 0},
				chunks: []types.DocumentChunk{
					{ID: "identical", Embedding: []float64{1, 0, 0}},
					{ID: "similar", Embedding: []float64{1, 1, 0}},
					{ID: "orthogonal", Embedding: []float64{0, 1, 0}},
				},
				limit: 2,
			},
			expected: expected{ids: []string{"identical", "similar"}},
		},
		{
			name: "returns all scored chunks when limit exceeds scored count",
			input: input{
				embedding: []float64{1, 0, 0},
				chunks: []types.DocumentChunk{
					{ID: "a", Embedding: []float64{1, 0, 0}},
					{ID: "b", Embedding: []float64{0, 1, 0}},
				},
				limit: 10,
			},
			expected: expected{ids: []string{"a", "b"}},
		},
		{
			name: "returns empty slice when all chunks have empty embeddings",
			input: input{
				embedding: []float64{1, 0, 0},
				chunks: []types.DocumentChunk{
					{ID: "no-embedding-1"},
					{ID: "no-embedding-2"},
				},
				limit: 5,
			},
			expected: expected{ids: []string{}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Search(tt.input.embedding, tt.input.chunks, tt.input.limit)

			assert.NoError(t, err)
			ids := make([]string, len(result))
			for i, scored := range result {
				ids[i] = scored.Chunk.ID
			}
			assert.Equal(t, tt.expected.ids, ids)
		})
	}
}

func TestCosineSimilarity(t *testing.T) {
	type input struct {
		a []float64
		b []float64
	}

	tests := []struct {
		name     string
		input    input
		expected float64
	}{
		{
			name:     "returns 1.0 for identical vectors",
			input:    input{a: []float64{1, 2, 3}, b: []float64{1, 2, 3}},
			expected: 1.0,
		},
		{
			name:     "returns -1.0 for opposite vectors",
			input:    input{a: []float64{1, 2, 3}, b: []float64{-1, -2, -3}},
			expected: -1.0,
		},
		{
			name:     "returns 0.0 for orthogonal vectors",
			input:    input{a: []float64{1, 0, 0}, b: []float64{0, 1, 0}},
			expected: 0.0,
		},
		{
			name:     "returns 0.0 when vector lengths differ",
			input:    input{a: []float64{1, 2, 3}, b: []float64{1, 2}},
			expected: 0.0,
		},
		{
			name:     "returns 0.0 when first vector is the zero vector",
			input:    input{a: []float64{0, 0, 0}, b: []float64{1, 2, 3}},
			expected: 0.0,
		},
		{
			name:     "returns 0.0 when second vector is the zero vector",
			input:    input{a: []float64{1, 2, 3}, b: []float64{0, 0, 0}},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cosineSimilarity(tt.input.a, tt.input.b)

			assert.InDelta(t, tt.expected, result, 1e-9)
		})
	}
}
