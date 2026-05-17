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
