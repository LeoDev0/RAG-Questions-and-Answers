package services

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/stretchr/testify/assert"

	"rag-backend/internal/config"
	"rag-backend/pkg/types"
	"rag-backend/pkg/utils"
)

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

type mockEmbeddingCreator struct {
	newFunc func(ctx context.Context, body openai.EmbeddingNewParams, opts ...option.RequestOption) (*openai.CreateEmbeddingResponse, error)
}

func (m *mockEmbeddingCreator) New(ctx context.Context, body openai.EmbeddingNewParams, opts ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
	return m.newFunc(ctx, body, opts...)
}

type mockChatCompleter struct {
	newFunc func(ctx context.Context, body openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error)
}

func (m *mockChatCompleter) New(ctx context.Context, body openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error) {
	return m.newFunc(ctx, body, opts...)
}

type mockVectorStore struct {
	storeFunc  func(chunks []types.DocumentChunk) error
	searchFunc func(embedding []float64, limit int) ([]types.ScoredChunk, error)
}

func (m *mockVectorStore) Store(chunks []types.DocumentChunk) error {
	return m.storeFunc(chunks)
}

func (m *mockVectorStore) Search(embedding []float64, limit int) ([]types.ScoredChunk, error) {
	return m.searchFunc(embedding, limit)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func newTestPipeline(ec EmbeddingCreator, cc ChatCompletionCreator, vs *mockVectorStore) *RAGPipeline {
	return &RAGPipeline{
		config:           &config.Config{Port: "3001", OpenAIAPIKey: "test-key", DeepSeekAPIKey: "test-key"},
		embeddingCreator: ec,
		chatCompleter:    cc,
		vectorStore:      vs,
		textSplitter:     utils.NewTextSplitter(chunkSize, chunkOverlap),
	}
}

func makeEmbeddingResponse(embeddings [][]float64) *openai.CreateEmbeddingResponse {
	data := make([]openai.Embedding, len(embeddings))
	for i, emb := range embeddings {
		data[i] = openai.Embedding{Embedding: emb}
	}
	return &openai.CreateEmbeddingResponse{Data: data}
}

func makeChatCompletion(content string) *openai.ChatCompletion {
	return &openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{Message: openai.ChatCompletionMessage{Content: content}},
		},
	}
}

// ---------------------------------------------------------------------------
// TestNewRAGPipeline
// ---------------------------------------------------------------------------

func TestNewRAGPipeline(t *testing.T) {
	cfg := &config.Config{
		Port:          "3001",
		OpenAIAPIKey:  "test-openai-key",
		DeepSeekAPIKey: "test-deepseek-key",
	}
	vs := &mockVectorStore{}

	pipeline := NewRAGPipeline(cfg, vs)

	assert.NotNil(t, pipeline)
	assert.Equal(t, cfg, pipeline.config)
	assert.Equal(t, vs, pipeline.vectorStore)
	assert.NotNil(t, pipeline.embeddingCreator)
	assert.NotNil(t, pipeline.chatCompleter)
	assert.NotNil(t, pipeline.textSplitter)
	assert.Equal(t, chunkSize, pipeline.textSplitter.ChunkSize)
	assert.Equal(t, chunkOverlap, pipeline.textSplitter.ChunkOverlap)
}

// ---------------------------------------------------------------------------
// TestGenerateEmbedding
// ---------------------------------------------------------------------------

func TestGenerateEmbedding(t *testing.T) {
	tests := []struct {
		name        string
		mockReturn  *openai.CreateEmbeddingResponse
		mockErr     error
		expected    []float64
		expectedErr string
	}{
		{
			name:       "returns embedding successfully",
			mockReturn: makeEmbeddingResponse([][]float64{{0.1, 0.2, 0.3}}),
			expected:   []float64{0.1, 0.2, 0.3},
		},
		{
			name:        "propagates API error",
			mockErr:     errors.New("openai api error"),
			expectedErr: "openai api error",
		},
		{
			name:        "returns error when response data is empty",
			mockReturn:  makeEmbeddingResponse([][]float64{}),
			expectedErr: "no embedding returned",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ec := &mockEmbeddingCreator{
				newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
					return tt.mockReturn, tt.mockErr
				},
			}
			pipeline := newTestPipeline(ec, nil, &mockVectorStore{})

			result, err := pipeline.generateEmbedding("test text")

			if tt.expectedErr != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedErr)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, result)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestGenerateEmbeddingBatch
// ---------------------------------------------------------------------------

func TestGenerateEmbeddingBatch(t *testing.T) {
	tests := []struct {
		name        string
		texts       []string
		mockReturn  *openai.CreateEmbeddingResponse
		mockErr     error
		expected    [][]float64
		expectedErr string
	}{
		{
			name:       "returns embeddings for multiple texts",
			texts:      []string{"hello", "world", "foo"},
			mockReturn: makeEmbeddingResponse([][]float64{{0.1}, {0.2}, {0.3}}),
			expected:   [][]float64{{0.1}, {0.2}, {0.3}},
		},
		{
			name:       "returns embedding for single text",
			texts:      []string{"single"},
			mockReturn: makeEmbeddingResponse([][]float64{{0.5, 0.6}}),
			expected:   [][]float64{{0.5, 0.6}},
		},
		{
			name:        "returns error for empty texts slice",
			texts:       []string{},
			expectedErr: "no texts provided for batch embedding",
		},
		{
			name:        "propagates API error",
			texts:       []string{"hello"},
			mockErr:     errors.New("batch api error"),
			expectedErr: "batch api error",
		},
		{
			name:        "returns error on embedding count mismatch",
			texts:       []string{"a", "b"},
			mockReturn:  makeEmbeddingResponse([][]float64{{0.1}}),
			expectedErr: "expected 2 embeddings, got 1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ec := &mockEmbeddingCreator{
				newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
					return tt.mockReturn, tt.mockErr
				},
			}
			pipeline := newTestPipeline(ec, nil, &mockVectorStore{})

			result, err := pipeline.generateEmbeddingBatch(tt.texts)

			if tt.expectedErr != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedErr)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, result)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestGenerateEmbeddingParallel
// ---------------------------------------------------------------------------

func TestGenerateEmbeddingParallel(t *testing.T) {
	// Helper: creates N texts and a mock that returns a distinct embedding per text
	makeTextsAndMock := func(n int, failBatch int) ([]string, *mockEmbeddingCreator) {
		texts := make([]string, n)
		for i := range texts {
			texts[i] = fmt.Sprintf("text-%d", i)
		}
		ec := &mockEmbeddingCreator{
			newFunc: func(_ context.Context, body openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
				batchTexts := body.Input.OfArrayOfStrings
				if failBatch >= 0 {
					// Determine which batch this is by checking the first text index
					for i := 0; i < len(texts); i += maxBatchSize {
						batchIdx := i / maxBatchSize
						end := min(i+maxBatchSize, len(texts))
						if len(batchTexts) == end-i && batchTexts[0] == texts[i] {
							if batchIdx == failBatch {
								return nil, fmt.Errorf("batch %d failed", batchIdx)
							}
							break
						}
					}
				}
				embeddings := make([][]float64, len(batchTexts))
				for i, txt := range batchTexts {
					// Parse the index from "text-N" to create a unique, verifiable embedding
					var idx int
					fmt.Sscanf(txt, "text-%d", &idx)
					embeddings[i] = []float64{float64(idx)}
				}
				return makeEmbeddingResponse(embeddings), nil
			},
		}
		return texts, ec
	}

	tests := []struct {
		name        string
		numTexts    int
		failBatch   int
		expectedErr string
	}{
		{
			name:      "exact batch size produces single batch with correct order",
			numTexts:  40,
			failBatch: -1,
		},
		{
			name:      "multiple batches preserves order (85 texts = 3 batches of 40+40+5)",
			numTexts:  85,
			failBatch: -1,
		},
		{
			name:      "boundary case of 41 texts splits into 2 batches (40+1)",
			numTexts:  41,
			failBatch: -1,
		},
		{
			name:        "returns error when a batch fails",
			numTexts:    80,
			failBatch:   1,
			expectedErr: "failed to generate embeddings for batch",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			texts, ec := makeTextsAndMock(tt.numTexts, tt.failBatch)
			pipeline := newTestPipeline(ec, nil, &mockVectorStore{})

			result, err := pipeline.generateEmbeddingParallel(texts)

			if tt.expectedErr != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedErr)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.Len(t, result, tt.numTexts)
				// Verify ordering: embedding[i] should be []float64{float64(i)}
				for i, emb := range result {
					assert.Equal(t, []float64{float64(i)}, emb, "embedding at index %d should match", i)
				}
			}
		})
	}
}

func TestGenerateEmbeddingParallel_ConcurrencyLimit(t *testing.T) {
	var currentConcurrency atomic.Int32
	var peakConcurrency atomic.Int32

	numTexts := 200 // 5 batches of 40
	texts := make([]string, numTexts)
	for i := range texts {
		texts[i] = fmt.Sprintf("text-%d", i)
	}

	ec := &mockEmbeddingCreator{
		newFunc: func(_ context.Context, body openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
			cur := currentConcurrency.Add(1)
			// Track the peak concurrency observed
			for {
				peak := peakConcurrency.Load()
				if cur <= peak || peakConcurrency.CompareAndSwap(peak, cur) {
					break
				}
			}

			// Small sleep to create overlap between goroutines
			time.Sleep(5 * time.Millisecond)
			currentConcurrency.Add(-1)

			batchTexts := body.Input.OfArrayOfStrings
			embeddings := make([][]float64, len(batchTexts))
			for i := range batchTexts {
				embeddings[i] = []float64{0.1}
			}
			return makeEmbeddingResponse(embeddings), nil
		},
	}

	pipeline := newTestPipeline(ec, nil, &mockVectorStore{})
	result, err := pipeline.generateEmbeddingParallel(texts)

	assert.NoError(t, err)
	assert.Len(t, result, numTexts)
	assert.LessOrEqual(t, int(peakConcurrency.Load()), maxConcurrency,
		"peak concurrency %d should not exceed maxConcurrency %d", peakConcurrency.Load(), maxConcurrency)
}

// ---------------------------------------------------------------------------
// TestGenerateResponse
// ---------------------------------------------------------------------------

func TestGenerateResponse(t *testing.T) {
	tests := []struct {
		name           string
		contextInfo    string
		question       string
		mockReturn     *openai.ChatCompletion
		mockErr        error
		expectedAnswer string
		expectedErr    string
	}{
		{
			name:           "returns response content from first choice",
			contextInfo:    "Go is a compiled language.",
			question:       "What is Go?",
			mockReturn:     makeChatCompletion("Go is a compiled programming language."),
			expectedAnswer: "Go is a compiled programming language.",
		},
		{
			name:        "wraps API error",
			contextInfo: "ctx",
			question:    "q",
			mockErr:     errors.New("deepseek timeout"),
			expectedErr: "failed to generate response",
		},
		{
			name:        "returns error when choices are empty",
			contextInfo: "ctx",
			question:    "q",
			mockReturn:  &openai.ChatCompletion{Choices: []openai.ChatCompletionChoice{}},
			expectedErr: "no response from DeepSeek API",
		},
		{
			name:           "prompt contains context and question",
			contextInfo:    "Rust is memory safe.",
			question:       "Is Rust safe?",
			mockReturn:     makeChatCompletion("Yes"),
			expectedAnswer: "Yes",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var capturedPrompt string
			cc := &mockChatCompleter{
				newFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
					if len(body.Messages) > 0 {
						// Extract the user message content for prompt verification
						capturedPrompt = body.Messages[0].OfUser.Content.OfString.Value
					}
					return tt.mockReturn, tt.mockErr
				},
			}
			pipeline := newTestPipeline(nil, cc, &mockVectorStore{})

			result, err := pipeline.generateResponse(tt.contextInfo, tt.question)

			if tt.expectedErr != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedErr)
				assert.Empty(t, result)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expectedAnswer, result)
			}

			// For the prompt format test, verify the prompt structure
			if tt.name == "prompt contains context and question" {
				assert.Contains(t, capturedPrompt, "Context information:")
				assert.Contains(t, capturedPrompt, tt.contextInfo)
				assert.Contains(t, capturedPrompt, "Question: "+tt.question)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestProcessDocument
// ---------------------------------------------------------------------------

func TestProcessDocument(t *testing.T) {
	tests := []struct {
		name           string
		content        string
		metadata       map[string]string
		mockReturn     *openai.CreateEmbeddingResponse
		mockErr        error
		expectedChunks int
		expectedErr    string
	}{
		{
			name:           "short content produces single chunk via batch path",
			content:        "hello world",
			metadata:       map[string]string{"source": "doc1"},
			mockReturn:     makeEmbeddingResponse([][]float64{{0.1, 0.2}}),
			expectedChunks: 1,
		},
		{
			name:    "multiple chunks produced from medium content",
			content: strings.Repeat("a", 2500),
			metadata: map[string]string{"source": "doc2"},
			// TextSplitter with chunkSize=1000, overlap=200 on 2500 chars:
			// chunk0: [0,1000), chunk1: [800,1800), chunk2: [1600,2500)
			mockReturn:     makeEmbeddingResponse([][]float64{{0.1}, {0.2}, {0.3}}),
			expectedChunks: 3,
		},
		{
			name:        "embedding error is wrapped",
			content:     "some text",
			metadata:    map[string]string{"source": "err"},
			mockErr:     errors.New("api failure"),
			expectedErr: "failed to generate embeddings",
		},
		{
			name:           "empty content produces single chunk",
			content:        "",
			metadata:       map[string]string{"source": "empty"},
			mockReturn:     makeEmbeddingResponse([][]float64{{0.0}}),
			expectedChunks: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ec := &mockEmbeddingCreator{
				newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
					return tt.mockReturn, tt.mockErr
				},
			}
			pipeline := newTestPipeline(ec, nil, &mockVectorStore{})

			chunks, err := pipeline.ProcessDocument(tt.content, tt.metadata)

			if tt.expectedErr != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedErr)
				assert.Nil(t, chunks)
			} else {
				assert.NoError(t, err)
				assert.Len(t, chunks, tt.expectedChunks)

				for i, chunk := range chunks {
					expectedID := fmt.Sprintf("%s-chunk-%d", tt.metadata["source"], i)
					assert.Equal(t, expectedID, chunk.ID, "chunk %d should have correct ID", i)
					assert.Equal(t, tt.metadata, chunk.Metadata)
					assert.NotNil(t, chunk.Embedding)
				}
			}
		})
	}
}

func TestProcessDocument_LargeDocumentUsesParallelPath(t *testing.T) {
	// Generate content large enough to produce > 40 chunks
	// With chunkSize=1000 and chunkOverlap=200, each chunk after the first advances by 800 chars.
	// For 41 chunks: 1000 + 40*800 = 33000 characters
	content := strings.Repeat("x", 33_000)
	metadata := map[string]string{"source": "large-doc"}

	var callCount atomic.Int32
	ec := &mockEmbeddingCreator{
		newFunc: func(_ context.Context, body openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
			callCount.Add(1)
			batchTexts := body.Input.OfArrayOfStrings
			embeddings := make([][]float64, len(batchTexts))
			for i := range batchTexts {
				embeddings[i] = []float64{0.1}
			}
			return makeEmbeddingResponse(embeddings), nil
		},
	}
	pipeline := newTestPipeline(ec, nil, &mockVectorStore{})

	chunks, err := pipeline.ProcessDocument(content, metadata)

	assert.NoError(t, err)
	assert.Greater(t, len(chunks), maxBatchSize, "should have more than maxBatchSize chunks to trigger parallel path")
	// Parallel path splits into multiple batches, so the mock should be called more than once
	assert.Greater(t, int(callCount.Load()), 1, "parallel path should call embedding API multiple times")
}

// ---------------------------------------------------------------------------
// TestAddDocumentToVectorStore
// ---------------------------------------------------------------------------

func TestAddDocumentToVectorStore(t *testing.T) {
	sampleChunks := []types.DocumentChunk{
		{ID: "c1", Content: "hello", Embedding: []float64{0.1}},
		{ID: "c2", Content: "world", Embedding: []float64{0.2}},
	}

	tests := []struct {
		name        string
		chunks      []types.DocumentChunk
		storeErr    error
		expectedErr string
	}{
		{
			name:   "stores chunks successfully",
			chunks: sampleChunks,
		},
		{
			name:        "wraps store error",
			chunks:      sampleChunks,
			storeErr:    errors.New("disk full"),
			expectedErr: "failed to store chunks",
		},
		{
			name:   "handles empty chunks slice",
			chunks: []types.DocumentChunk{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var storedChunks []types.DocumentChunk
			vs := &mockVectorStore{
				storeFunc: func(chunks []types.DocumentChunk) error {
					storedChunks = chunks
					return tt.storeErr
				},
			}
			pipeline := newTestPipeline(nil, nil, vs)

			err := pipeline.AddDocumentToVectorStore(tt.chunks)

			if tt.expectedErr != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedErr)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.chunks, storedChunks)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestQuery
// ---------------------------------------------------------------------------

func TestQuery(t *testing.T) {
	tests := []struct {
		name             string
		question         string
		embeddingReturn  *openai.CreateEmbeddingResponse
		embeddingErr     error
		searchReturn     []types.ScoredChunk
		searchErr        error
		chatReturn       *openai.ChatCompletion
		chatErr          error
		expectedAnswer   string
		expectedSources  int
		expectedErr      string
	}{
		{
			name:            "full pipeline success with single source",
			question:        "What is Go?",
			embeddingReturn: makeEmbeddingResponse([][]float64{{0.5, 0.6}}),
			searchReturn: []types.ScoredChunk{
				{Chunk: types.DocumentChunk{ID: "c1", Content: "Go is a language"}, Score: 0.9},
			},
			chatReturn:      makeChatCompletion("Go is a compiled language."),
			expectedAnswer:  "Go is a compiled language.",
			expectedSources: 1,
		},
		{
			name:            "multiple sources joined with double newline separator",
			question:        "Tell me about Go",
			embeddingReturn: makeEmbeddingResponse([][]float64{{0.5}}),
			searchReturn: []types.ScoredChunk{
				{Chunk: types.DocumentChunk{ID: "c1", Content: "Go is compiled"}, Score: 0.9},
				{Chunk: types.DocumentChunk{ID: "c2", Content: "Go has goroutines"}, Score: 0.8},
				{Chunk: types.DocumentChunk{ID: "c3", Content: "Go is statically typed"}, Score: 0.7},
			},
			chatReturn:      makeChatCompletion("Go is great."),
			expectedAnswer:  "Go is great.",
			expectedSources: 3,
		},
		{
			name:         "returns error when embedding generation fails",
			question:     "fail",
			embeddingErr: errors.New("openai down"),
			expectedErr:  "failed to generate embedding for query",
		},
		{
			name:            "returns error when vector store search fails",
			question:        "search fail",
			embeddingReturn: makeEmbeddingResponse([][]float64{{0.1}}),
			searchErr:       errors.New("search broken"),
			expectedErr:     "failed to search vector store",
		},
		{
			name:            "returns error when response generation fails",
			question:        "resp fail",
			embeddingReturn: makeEmbeddingResponse([][]float64{{0.1}}),
			searchReturn:    []types.ScoredChunk{{Chunk: types.DocumentChunk{Content: "ctx"}, Score: 0.5}},
			chatErr:         errors.New("deepseek timeout"),
			expectedErr:     "failed to generate response",
		},
		{
			name:            "handles no search results with empty context",
			question:        "obscure topic",
			embeddingReturn: makeEmbeddingResponse([][]float64{{0.1}}),
			searchReturn:    []types.ScoredChunk{},
			chatReturn:      makeChatCompletion("I don't have enough information."),
			expectedAnswer:  "I don't have enough information.",
			expectedSources: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ec := &mockEmbeddingCreator{
				newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
					return tt.embeddingReturn, tt.embeddingErr
				},
			}

			var capturedContext string
			cc := &mockChatCompleter{
				newFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
					if len(body.Messages) > 0 {
						capturedContext = body.Messages[0].OfUser.Content.OfString.Value
					}
					return tt.chatReturn, tt.chatErr
				},
			}

			vs := &mockVectorStore{
				searchFunc: func(embedding []float64, limit int) ([]types.ScoredChunk, error) {
					return tt.searchReturn, tt.searchErr
				},
			}

			pipeline := newTestPipeline(ec, cc, vs)
			result, err := pipeline.Query(tt.question)

			if tt.expectedErr != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedErr)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				assert.Equal(t, tt.expectedAnswer, result.Answer)
				assert.Len(t, result.Sources, tt.expectedSources)
				assert.Equal(t, defaultConfidence, result.Confidence)

				// Verify context contains all source contents joined by "\n\n"
				for _, sc := range tt.searchReturn {
					assert.Contains(t, capturedContext, sc.Chunk.Content)
				}
			}
		})
	}
}

func TestQuery_ContextBuiltFromMultipleSources(t *testing.T) {
	ec := &mockEmbeddingCreator{
		newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
			return makeEmbeddingResponse([][]float64{{0.1}}), nil
		},
	}

	var capturedPrompt string
	cc := &mockChatCompleter{
		newFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
			if len(body.Messages) > 0 {
				capturedPrompt = body.Messages[0].OfUser.Content.OfString.Value
			}
			return makeChatCompletion("answer"), nil
		},
	}

	vs := &mockVectorStore{
		searchFunc: func(_ []float64, _ int) ([]types.ScoredChunk, error) {
			return []types.ScoredChunk{
				{Chunk: types.DocumentChunk{Content: "First chunk"}, Score: 0.9},
				{Chunk: types.DocumentChunk{Content: "Second chunk"}, Score: 0.8},
			}, nil
		},
	}

	pipeline := newTestPipeline(ec, cc, vs)
	_, err := pipeline.Query("test question")

	assert.NoError(t, err)
	// Verify chunks are separated by double newline in the context
	assert.Contains(t, capturedPrompt, "First chunk\n\nSecond chunk")
}

func TestQuery_PassesCorrectSearchLimit(t *testing.T) {
	ec := &mockEmbeddingCreator{
		newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
			return makeEmbeddingResponse([][]float64{{0.1}}), nil
		},
	}
	cc := &mockChatCompleter{
		newFunc: func(_ context.Context, _ openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
			return makeChatCompletion("answer"), nil
		},
	}

	var capturedLimit int
	vs := &mockVectorStore{
		searchFunc: func(_ []float64, limit int) ([]types.ScoredChunk, error) {
			capturedLimit = limit
			return []types.ScoredChunk{}, nil
		},
	}

	pipeline := newTestPipeline(ec, cc, vs)
	_, err := pipeline.Query("test")

	assert.NoError(t, err)
	assert.Equal(t, maxContentChunks, capturedLimit)
}
