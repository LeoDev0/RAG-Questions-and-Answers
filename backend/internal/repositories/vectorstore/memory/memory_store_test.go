package memory

import (
	"sync"
	"testing"

	"rag-backend/internal/repositories/vectorstore"
	"rag-backend/pkg/types"

	"github.com/stretchr/testify/assert"
)

func chunk(id string, embedding []float64) types.DocumentChunk {
	return types.DocumentChunk{ID: id, Embedding: embedding}
}

func sourcedChunk(source string, index int) types.DocumentChunk {
	return types.DocumentChunk{
		ID:         source + "-chunk-" + string(rune('0'+index)),
		Source:     source,
		ChunkIndex: index,
		Embedding:  []float64{1, 0, 0},
	}
}

func neighborIDs(chunks []types.DocumentChunk) []string {
	ids := make([]string, len(chunks))
	for i, c := range chunks {
		ids[i] = c.ID
	}
	return ids
}

func searchedIDs(scored []types.ScoredChunk) []string {
	ids := make([]string, len(scored))
	for i, s := range scored {
		ids[i] = s.Chunk.ID
	}
	return ids
}

func TestNewMemoryVectorStore(t *testing.T) {
	store := NewMemoryVectorStore()

	assert.NotNil(t, store)
	assert.Implements(t, (*vectorstore.VectorStore)(nil), store)

	result, err := store.Search([]float64{1, 0, 0}, 5)

	assert.NoError(t, err)
	assert.Empty(t, result)
}

func TestMemoryVectorStoreStore(t *testing.T) {
	type input struct {
		batches [][]types.DocumentChunk
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
			name: "stores a single non-empty batch",
			input: input{batches: [][]types.DocumentChunk{
				{chunk("a", []float64{1, 0, 0}), chunk("b", []float64{0, 1, 0})},
			}},
			expected: expected{ids: []string{"a", "b"}},
		},
		{
			name: "stores an empty batch as a no-op",
			input: input{batches: [][]types.DocumentChunk{
				{},
			}},
			expected: expected{ids: []string{}},
		},
		{
			name: "stores a nil batch as a no-op",
			input: input{batches: [][]types.DocumentChunk{
				nil,
			}},
			expected: expected{ids: []string{}},
		},
		{
			name: "accumulates across multiple sequential stores",
			input: input{batches: [][]types.DocumentChunk{
				{chunk("a", []float64{1, 0, 0})},
				{chunk("b", []float64{1, 0, 0})},
				{chunk("c", []float64{1, 0, 0})},
			}},
			expected: expected{ids: []string{"a", "b", "c"}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store := NewMemoryVectorStore()

			for _, batch := range tt.input.batches {
				assert.NoError(t, store.Store(batch))
			}

			result, err := store.Search([]float64{1, 0, 0}, 100)

			assert.NoError(t, err)
			assert.ElementsMatch(t, tt.expected.ids, searchedIDs(result))
		})
	}
}

func TestMemoryVectorStoreSearch(t *testing.T) {
	t.Run("returns empty result when the store is empty", func(t *testing.T) {
		store := NewMemoryVectorStore()

		result, err := store.Search([]float64{1, 0, 0}, 5)

		assert.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("delegates to similarity.Search over the stored chunks", func(t *testing.T) {
		store := NewMemoryVectorStore()
		assert.NoError(t, store.Store([]types.DocumentChunk{
			chunk("orthogonal", []float64{0, 1, 0}),
			chunk("identical", []float64{1, 0, 0}),
			chunk("similar", []float64{1, 1, 0}),
		}))

		result, err := store.Search([]float64{1, 0, 0}, 5)

		assert.NoError(t, err)
		assert.Equal(t, []string{"identical", "similar", "orthogonal"}, searchedIDs(result))
	})
}

func TestMemoryVectorStoreNeighbors(t *testing.T) {
	type input struct {
		source string
		index  int
		radius int
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
			name:     "returns the window around a middle chunk",
			input:    input{source: "doc", index: 2, radius: 1},
			expected: expected{ids: []string{"doc-chunk-1", "doc-chunk-2", "doc-chunk-3"}},
		},
		{
			name:     "clamps at the start without negative indices",
			input:    input{source: "doc", index: 0, radius: 1},
			expected: expected{ids: []string{"doc-chunk-0", "doc-chunk-1"}},
		},
		{
			name:     "clamps at the last chunk",
			input:    input{source: "doc", index: 4, radius: 1},
			expected: expected{ids: []string{"doc-chunk-3", "doc-chunk-4"}},
		},
		{
			name:     "radius zero returns only the center",
			input:    input{source: "doc", index: 2, radius: 0},
			expected: expected{ids: []string{"doc-chunk-2"}},
		},
		{
			name:     "radius beyond the document returns all of its chunks",
			input:    input{source: "doc", index: 2, radius: 99},
			expected: expected{ids: []string{"doc-chunk-0", "doc-chunk-1", "doc-chunk-2", "doc-chunk-3", "doc-chunk-4"}},
		},
		{
			name:     "unknown source returns nothing",
			input:    input{source: "missing", index: 0, radius: 2},
			expected: expected{ids: []string{}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store := NewMemoryVectorStore()
			doc := make([]types.DocumentChunk, 0, 5)
			for i := 0; i < 5; i++ {
				doc = append(doc, sourcedChunk("doc", i))
			}
			other := []types.DocumentChunk{sourcedChunk("other", 1), sourcedChunk("other", 2)}
			assert.NoError(t, store.Store(doc))
			assert.NoError(t, store.Store(other))

			result, err := store.Neighbors(tt.input.source, tt.input.index, tt.input.radius)

			assert.NoError(t, err)
			assert.ElementsMatch(t, tt.expected.ids, neighborIDs(result))
			for _, c := range result {
				assert.Equal(t, tt.input.source, c.Source, "neighbors must never cross document boundaries")
			}
		})
	}
}

func TestMemoryVectorStoreConcurrentAccess(t *testing.T) {
	const writers = 25

	store := NewMemoryVectorStore()
	embedding := []float64{1, 0, 0}

	var wg sync.WaitGroup
	for i := 0; i < writers; i++ {
		wg.Add(2)
		go func(id string) {
			defer wg.Done()
			assert.NoError(t, store.Store([]types.DocumentChunk{chunk(id, embedding)}))
		}(string(rune('a' + i)))
		go func() {
			defer wg.Done()
			_, err := store.Search(embedding, 10)
			assert.NoError(t, err)
		}()
	}
	wg.Wait()

	result, err := store.Search(embedding, writers+10)

	assert.NoError(t, err)
	ids := make([]string, writers)
	for i := 0; i < writers; i++ {
		ids[i] = string(rune('a' + i))
	}
	assert.ElementsMatch(t, ids, searchedIDs(result))
}
