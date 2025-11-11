package services

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"rag-backend/internal/config"
	"rag-backend/pkg/types"
	"rag-backend/pkg/utils"
)

// MockVectorStore implements the VectorStore interface for testing
type MockVectorStore struct {
	mu           sync.Mutex
	storeFunc    func(chunks []types.DocumentChunk) error
	searchFunc   func(embedding []float64, limit int) ([]types.ScoredChunk, error)
	storedChunks []types.DocumentChunk
}

func NewMockVectorStore() *MockVectorStore {
	return &MockVectorStore{
		storedChunks: make([]types.DocumentChunk, 0),
	}
}

func (m *MockVectorStore) Store(chunks []types.DocumentChunk) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.storeFunc != nil {
		return m.storeFunc(chunks)
	}

	m.storedChunks = append(m.storedChunks, chunks...)
	return nil
}

func (m *MockVectorStore) Search(embedding []float64, limit int) ([]types.ScoredChunk, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.searchFunc != nil {
		return m.searchFunc(embedding, limit)
	}

	// Default behavior: return stored chunks with mock scores
	results := make([]types.ScoredChunk, 0)
	for i, chunk := range m.storedChunks {
		if i >= limit {
			break
		}
		results = append(results, types.ScoredChunk{
			Chunk: chunk,
			Score: 0.9 - float64(i)*0.1, // Decreasing scores
		})
	}
	return results, nil
}

func (m *MockVectorStore) GetStoredChunks() []types.DocumentChunk {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]types.DocumentChunk{}, m.storedChunks...)
}

func (m *MockVectorStore) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.storedChunks = make([]types.DocumentChunk, 0)
	m.storeFunc = nil
	m.searchFunc = nil
}

// Test helper to create a test config
func createTestConfig() *config.Config {
	return &config.Config{
		OpenAIAPIKey:   "test-openai-key",
		DeepSeekAPIKey: "test-deepseek-key",
	}
}

func TestNewRAGPipeline(t *testing.T) {
	cfg := createTestConfig()
	mockStore := NewMockVectorStore()

	pipeline := NewRAGPipeline(cfg, mockStore)

	if pipeline == nil {
		t.Fatal("NewRAGPipeline() returned nil")
	}
	if pipeline.config != cfg {
		t.Error("Config not properly set")
	}
	if pipeline.vectorStore != mockStore {
		t.Error("VectorStore not properly set")
	}
	if pipeline.textSplitter == nil {
		t.Error("TextSplitter should be initialized")
	}
}

func TestProcessDocument(t *testing.T) {
	// Note: This test is limited without actual API calls
	// In a real-world scenario, you'd mock the OpenAI client
	t.Run("empty content", func(t *testing.T) {
		cfg := createTestConfig()
		mockStore := NewMockVectorStore()
		pipeline := NewRAGPipeline(cfg, mockStore)

		// Override text splitter for controlled testing
		pipeline.textSplitter = utils.NewTextSplitter(100, 20)

		metadata := map[string]string{
			"source": "test-doc",
		}

		// This will fail because we can't make real API calls in tests
		// But we can verify the structure
		_, err := pipeline.ProcessDocument("", metadata)

		// We expect an error because the OpenAI API key is fake
		// and we can't make real API calls
		if err == nil {
			t.Error("Expected error when processing document without valid API")
		}
	})

	t.Run("nil metadata", func(t *testing.T) {
		cfg := createTestConfig()
		mockStore := NewMockVectorStore()
		pipeline := NewRAGPipeline(cfg, mockStore)

		_, err := pipeline.ProcessDocument("test content", nil)

		// Should handle nil metadata gracefully (even though it will fail on API call)
		// The error should be about API, not about nil metadata
		if err != nil && strings.Contains(err.Error(), "nil") {
			t.Error("Should handle nil metadata gracefully")
		}
	})
}

func TestAddDocumentToVectorStore(t *testing.T) {
	tests := []struct {
		name      string
		chunks    []types.DocumentChunk
		storeFunc func(chunks []types.DocumentChunk) error
		wantErr   bool
	}{
		{
			name: "successful storage",
			chunks: []types.DocumentChunk{
				{
					ID:        "chunk-1",
					Content:   "Test content 1",
					Embedding: []float64{0.1, 0.2, 0.3},
					Metadata:  map[string]string{"source": "test"},
				},
			},
			storeFunc: nil, // Use default behavior
			wantErr:   false,
		},
		{
			name: "storage error",
			chunks: []types.DocumentChunk{
				{
					ID:      "chunk-1",
					Content: "Test content",
				},
			},
			storeFunc: func(chunks []types.DocumentChunk) error {
				return errors.New("storage error")
			},
			wantErr: true,
		},
		{
			name:      "empty chunks",
			chunks:    []types.DocumentChunk{},
			storeFunc: nil,
			wantErr:   false,
		},
		{
			name: "multiple chunks",
			chunks: []types.DocumentChunk{
				{ID: "chunk-1", Content: "Content 1", Embedding: []float64{0.1}},
				{ID: "chunk-2", Content: "Content 2", Embedding: []float64{0.2}},
				{ID: "chunk-3", Content: "Content 3", Embedding: []float64{0.3}},
			},
			storeFunc: nil,
			wantErr:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := createTestConfig()
			mockStore := NewMockVectorStore()
			mockStore.storeFunc = tt.storeFunc
			pipeline := NewRAGPipeline(cfg, mockStore)

			err := pipeline.AddDocumentToVectorStore(tt.chunks)

			if (err != nil) != tt.wantErr {
				t.Errorf("AddDocumentToVectorStore() error = %v, wantErr %v", err, tt.wantErr)
			}

			// Verify chunks were stored if no error expected
			if !tt.wantErr && tt.storeFunc == nil {
				stored := mockStore.GetStoredChunks()
				if len(stored) != len(tt.chunks) {
					t.Errorf("Expected %d chunks stored, got %d", len(tt.chunks), len(stored))
				}
			}
		})
	}
}

func TestQuery(t *testing.T) {
	// Note: Full testing requires mocking the OpenAI client
	// These tests verify error handling and structure

	t.Run("empty question", func(t *testing.T) {
		cfg := createTestConfig()
		mockStore := NewMockVectorStore()
		pipeline := NewRAGPipeline(cfg, mockStore)

		_, err := pipeline.Query("")

		// Should get an error (likely from API call with empty string)
		if err == nil {
			t.Error("Expected error when querying with empty question")
		}
	})

	t.Run("vector store search error", func(t *testing.T) {
		cfg := createTestConfig()
		mockStore := NewMockVectorStore()
		mockStore.searchFunc = func(embedding []float64, limit int) ([]types.ScoredChunk, error) {
			return nil, errors.New("search failed")
		}
		pipeline := NewRAGPipeline(cfg, mockStore)

		// This will fail on embedding generation first, but tests the flow
		_, err := pipeline.Query("test question")

		if err == nil {
			t.Error("Expected error when vector store search fails")
		}
	})

	t.Run("vector store returns results", func(t *testing.T) {
		cfg := createTestConfig()
		mockStore := NewMockVectorStore()

		// Pre-populate store with some test chunks
		testChunks := []types.DocumentChunk{
			{
				ID:        "chunk-1",
				Content:   "This is test content about machine learning.",
				Embedding: []float64{0.1, 0.2, 0.3},
				Metadata:  map[string]string{"source": "test-doc"},
			},
			{
				ID:        "chunk-2",
				Content:   "More information about AI and deep learning.",
				Embedding: []float64{0.2, 0.3, 0.4},
				Metadata:  map[string]string{"source": "test-doc"},
			},
		}
		mockStore.Store(testChunks)

		pipeline := NewRAGPipeline(cfg, mockStore)

		// This will fail on embedding generation, but we can verify structure
		_, err := pipeline.Query("What is machine learning?")

		// We expect an API error since we're using fake keys
		if err == nil {
			t.Error("Expected error due to invalid API keys")
		}

		// Verify the error is related to embedding generation or API call
		if err != nil && !strings.Contains(err.Error(), "embedding") &&
		   !strings.Contains(err.Error(), "failed") {
			t.Logf("Error message: %v", err)
		}
	})
}

func TestTextSplitterIntegration(t *testing.T) {
	cfg := createTestConfig()
	mockStore := NewMockVectorStore()
	pipeline := NewRAGPipeline(cfg, mockStore)

	// Verify text splitter configuration
	if pipeline.textSplitter == nil {
		t.Fatal("TextSplitter should be initialized")
	}

	if pipeline.textSplitter.ChunkSize != chunkSize {
		t.Errorf("Expected chunk size %d, got %d", chunkSize, pipeline.textSplitter.ChunkSize)
	}

	if pipeline.textSplitter.ChunkOverlap != chunkOverlap {
		t.Errorf("Expected chunk overlap %d, got %d", chunkOverlap, pipeline.textSplitter.ChunkOverlap)
	}
}

func TestConstants(t *testing.T) {
	// Verify constants have reasonable values
	tests := []struct {
		name     string
		value    int
		min      int
		max      int
		checkMin bool
		checkMax bool
	}{
		{"defaultConfidence", int(defaultConfidence * 100), 0, 100, true, true},
		{"chunkSize", chunkSize, 100, 10000, true, true},
		{"chunkOverlap", chunkOverlap, 0, chunkSize, true, true},
		{"maxContentChunks", maxContentChunks, 1, 100, true, true},
		{"maxBatchSize", maxBatchSize, 1, 1000, true, true},
		{"maxConcurrency", maxConcurrency, 1, 100, true, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.checkMin && tt.value < tt.min {
				t.Errorf("%s = %d, should be >= %d", tt.name, tt.value, tt.min)
			}
			if tt.checkMax && tt.value > tt.max {
				t.Errorf("%s = %d, should be <= %d", tt.name, tt.value, tt.max)
			}
		})
	}

	// Verify overlap is less than chunk size
	if chunkOverlap >= chunkSize {
		t.Error("chunkOverlap should be less than chunkSize")
	}
}

func TestAddDocumentToVectorStore_Concurrent(t *testing.T) {
	cfg := createTestConfig()
	mockStore := NewMockVectorStore()
	pipeline := NewRAGPipeline(cfg, mockStore)

	const numGoroutines = 50
	done := make(chan bool, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(index int) {
			chunks := []types.DocumentChunk{
				{
					ID:        fmt.Sprintf("chunk-%d", index),
					Content:   fmt.Sprintf("Content %d", index),
					Embedding: []float64{float64(index)},
					Metadata:  map[string]string{"source": "test"},
				},
			}
			err := pipeline.AddDocumentToVectorStore(chunks)
			if err != nil {
				t.Errorf("Concurrent AddDocumentToVectorStore failed: %v", err)
			}
			done <- true
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < numGoroutines; i++ {
		<-done
	}

	// Verify all chunks were stored
	stored := mockStore.GetStoredChunks()
	if len(stored) != numGoroutines {
		t.Errorf("Expected %d chunks stored, got %d", numGoroutines, len(stored))
	}
}

func TestMockVectorStore(t *testing.T) {
	t.Run("Store and Search", func(t *testing.T) {
		mock := NewMockVectorStore()

		chunks := []types.DocumentChunk{
			{ID: "1", Content: "Content 1", Embedding: []float64{0.1, 0.2}},
			{ID: "2", Content: "Content 2", Embedding: []float64{0.3, 0.4}},
			{ID: "3", Content: "Content 3", Embedding: []float64{0.5, 0.6}},
		}

		err := mock.Store(chunks)
		if err != nil {
			t.Fatalf("Store failed: %v", err)
		}

		results, err := mock.Search([]float64{0.1, 0.2}, 2)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		if len(results) != 2 {
			t.Errorf("Expected 2 results, got %d", len(results))
		}

		// Verify scores are in descending order
		for i := 0; i < len(results)-1; i++ {
			if results[i].Score < results[i+1].Score {
				t.Error("Results should be sorted by descending score")
			}
		}
	})

	t.Run("Custom store function", func(t *testing.T) {
		mock := NewMockVectorStore()
		expectedErr := errors.New("custom error")

		mock.storeFunc = func(chunks []types.DocumentChunk) error {
			return expectedErr
		}

		err := mock.Store([]types.DocumentChunk{{ID: "1"}})
		if err != expectedErr {
			t.Errorf("Expected custom error, got: %v", err)
		}
	})

	t.Run("Custom search function", func(t *testing.T) {
		mock := NewMockVectorStore()
		expectedResults := []types.ScoredChunk{
			{Chunk: types.DocumentChunk{ID: "custom"}, Score: 1.0},
		}

		mock.searchFunc = func(embedding []float64, limit int) ([]types.ScoredChunk, error) {
			return expectedResults, nil
		}

		results, err := mock.Search([]float64{0.1}, 5)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		if len(results) != 1 || results[0].Chunk.ID != "custom" {
			t.Error("Custom search function not used correctly")
		}
	})

	t.Run("Reset", func(t *testing.T) {
		mock := NewMockVectorStore()

		// Store some data
		mock.Store([]types.DocumentChunk{{ID: "1"}})

		// Set custom function
		mock.storeFunc = func(chunks []types.DocumentChunk) error {
			return errors.New("error")
		}

		// Reset
		mock.Reset()

		// Verify state is reset
		stored := mock.GetStoredChunks()
		if len(stored) != 0 {
			t.Error("Stored chunks should be empty after reset")
		}

		// Verify custom function is cleared
		err := mock.Store([]types.DocumentChunk{{ID: "2"}})
		if err != nil {
			t.Error("Custom store function should be cleared after reset")
		}
	})
}

func TestRAGPipeline_NilConfig(t *testing.T) {
	// Test that pipeline can handle nil config gracefully
	// (in practice, this should be validated, but we test defensive programming)
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic with nil config")
		}
	}()

	mockStore := NewMockVectorStore()
	_ = NewRAGPipeline(nil, mockStore)
}

func TestRAGPipeline_ConfigValidation(t *testing.T) {
	tests := []struct {
		name   string
		config *config.Config
	}{
		{
			name: "empty API keys",
			config: &config.Config{
				OpenAIAPIKey:   "",
				DeepSeekAPIKey: "",
			},
		},
		{
			name: "only OpenAI key",
			config: &config.Config{
				OpenAIAPIKey:   "test-key",
				DeepSeekAPIKey: "",
			},
		},
		{
			name: "only DeepSeek key",
			config: &config.Config{
				OpenAIAPIKey:   "",
				DeepSeekAPIKey: "test-key",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockStore := NewMockVectorStore()
			pipeline := NewRAGPipeline(tt.config, mockStore)

			if pipeline == nil {
				t.Error("Pipeline should be created even with invalid config")
			}
		})
	}
}

// Benchmark tests
func BenchmarkAddDocumentToVectorStore(b *testing.B) {
	cfg := createTestConfig()
	mockStore := NewMockVectorStore()
	pipeline := NewRAGPipeline(cfg, mockStore)

	chunks := []types.DocumentChunk{
		{
			ID:        "bench-chunk",
			Content:   "Benchmark content",
			Embedding: make([]float64, 1536), // Typical embedding size
			Metadata:  map[string]string{"source": "bench"},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pipeline.AddDocumentToVectorStore(chunks)
	}
}

func BenchmarkMockVectorStore_Store(b *testing.B) {
	mock := NewMockVectorStore()
	chunk := types.DocumentChunk{
		ID:        "bench-chunk",
		Content:   "Benchmark content",
		Embedding: make([]float64, 1536),
		Metadata:  map[string]string{"source": "bench"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mock.Store([]types.DocumentChunk{chunk})
	}
}

func BenchmarkMockVectorStore_Search(b *testing.B) {
	mock := NewMockVectorStore()

	// Populate with test data
	for i := 0; i < 1000; i++ {
		mock.Store([]types.DocumentChunk{
			{
				ID:        fmt.Sprintf("chunk-%d", i),
				Content:   fmt.Sprintf("Content %d", i),
				Embedding: make([]float64, 1536),
			},
		})
	}

	embedding := make([]float64, 1536)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mock.Search(embedding, 10)
	}
}

// Test that demonstrates the expected batch size logic
func TestBatchSizeLogic(t *testing.T) {
	tests := []struct {
		name           string
		numTexts       int
		expectParallel bool
	}{
		{
			name:           "small document - use batch",
			numTexts:       10,
			expectParallel: false,
		},
		{
			name:           "exactly at threshold - use batch",
			numTexts:       maxBatchSize,
			expectParallel: false,
		},
		{
			name:           "large document - use parallel",
			numTexts:       maxBatchSize + 1,
			expectParallel: true,
		},
		{
			name:           "very large document - use parallel",
			numTexts:       200,
			expectParallel: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shouldUseParallel := tt.numTexts > maxBatchSize
			if shouldUseParallel != tt.expectParallel {
				t.Errorf("Expected parallel=%v for %d texts, got %v",
					tt.expectParallel, tt.numTexts, shouldUseParallel)
			}
		})
	}
}

// Test thread safety of RAGPipeline operations
func TestRAGPipeline_ThreadSafety(t *testing.T) {
	cfg := createTestConfig()
	mockStore := NewMockVectorStore()
	pipeline := NewRAGPipeline(cfg, mockStore)

	const numOperations = 100
	var wg sync.WaitGroup

	// Concurrent AddDocumentToVectorStore operations
	for i := 0; i < numOperations; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			chunks := []types.DocumentChunk{
				{
					ID:      fmt.Sprintf("chunk-%d", index),
					Content: fmt.Sprintf("Content %d", index),
				},
			}
			pipeline.AddDocumentToVectorStore(chunks)
		}(i)
	}

	wg.Wait()

	// Verify all operations completed
	stored := mockStore.GetStoredChunks()
	if len(stored) != numOperations {
		t.Errorf("Expected %d chunks, got %d", numOperations, len(stored))
	}
}
