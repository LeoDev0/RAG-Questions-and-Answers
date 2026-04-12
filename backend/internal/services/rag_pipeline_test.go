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
	"rag-backend/internal/repositories/vectorstore"
	"rag-backend/pkg/types"
	"rag-backend/pkg/utils"
)

func newTestPipeline(ec EmbeddingCreator, cc ChatCompletionCreator, vs *vectorstore.MockVectorStore) *RAGPipeline {
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

func TestNewRAGPipeline(t *testing.T) {
	cfg := &config.Config{
		Port:           "3001",
		OpenAIAPIKey:   "test-openai-key",
		DeepSeekAPIKey: "test-deepseek-key",
	}
	vs := &vectorstore.MockVectorStore{}

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

func TestGenerateEmbedding(t *testing.T) {
	type expected struct {
		result []float64
		err    string
	}
	type mock struct {
		response *openai.CreateEmbeddingResponse
		err      error
	}

	tests := []struct {
		name     string
		mock     mock
		expected expected
	}{
		{
			name: "returns embedding successfully",
			mock: mock{
				response: makeEmbeddingResponse([][]float64{{0.1, 0.2, 0.3}}),
			},
			expected: expected{
				result: []float64{0.1, 0.2, 0.3},
			},
		},
		{
			name: "propagates API error",
			mock: mock{
				err: errors.New("openai api error"),
			},
			expected: expected{
				err: "openai api error",
			},
		},
		{
			name: "returns error when response data is empty",
			mock: mock{
				response: makeEmbeddingResponse([][]float64{}),
			},
			expected: expected{
				err: "no embedding returned",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ec := &mockEmbeddingCreator{
				newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
					return tt.mock.response, tt.mock.err
				},
			}
			pipeline := newTestPipeline(ec, nil, &vectorstore.MockVectorStore{})

			result, err := pipeline.generateEmbedding("test text")

			if tt.expected.err != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expected.err)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected.result, result)
			}
		})
	}
}

func TestGenerateEmbeddingBatch(t *testing.T) {
	type expected struct {
		result [][]float64
		err    string
	}
	type mock struct {
		response *openai.CreateEmbeddingResponse
		err      error
	}

	tests := []struct {
		name     string
		texts    []string
		mock     mock
		expected expected
	}{
		{
			name:  "returns embeddings for multiple texts",
			texts: []string{"hello", "world", "foo"},
			mock: mock{
				response: makeEmbeddingResponse([][]float64{{0.1}, {0.2}, {0.3}}),
			},
			expected: expected{
				result: [][]float64{{0.1}, {0.2}, {0.3}},
			},
		},
		{
			name:  "returns embedding for single text",
			texts: []string{"single"},
			mock: mock{
				response: makeEmbeddingResponse([][]float64{{0.5, 0.6}}),
			},
			expected: expected{
				result: [][]float64{{0.5, 0.6}},
			},
		},
		{
			name:  "returns error for empty texts slice",
			texts: []string{},
			expected: expected{
				err: "no texts provided for batch embedding",
			},
		},
		{
			name:  "propagates API error",
			texts: []string{"hello"},
			mock: mock{
				err: errors.New("batch api error"),
			},
			expected: expected{
				err: "batch api error",
			},
		},
		{
			name:  "returns error on embedding count mismatch",
			texts: []string{"a", "b"},
			mock: mock{
				response: makeEmbeddingResponse([][]float64{{0.1}}),
			},
			expected: expected{
				err: "expected 2 embeddings, got 1",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ec := &mockEmbeddingCreator{
				newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
					return tt.mock.response, tt.mock.err
				},
			}
			pipeline := newTestPipeline(ec, nil, &vectorstore.MockVectorStore{})

			result, err := pipeline.generateEmbeddingBatch(tt.texts)

			if tt.expected.err != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expected.err)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected.result, result)
			}
		})
	}
}

func TestGenerateEmbeddingParallel(t *testing.T) {
	makeTextsAndMock := func(n int, shouldFail bool) ([]string, *mockEmbeddingCreator) {
		texts := make([]string, n)
		for i := range texts {
			texts[i] = fmt.Sprintf("text-%d", i)
		}
		ec := &mockEmbeddingCreator{
			newFunc: func(_ context.Context, body openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
				if shouldFail {
					return nil, fmt.Errorf("batch failed")
				}
				batchTexts := body.Input.OfArrayOfStrings
				embeddings := make([][]float64, len(batchTexts))
				for i, txt := range batchTexts {
					var idx int
					fmt.Sscanf(txt, "text-%d", &idx)
					embeddings[i] = []float64{float64(idx)}
				}
				return makeEmbeddingResponse(embeddings), nil
			},
		}
		return texts, ec
	}

	type expected struct {
		err string
	}

	tests := []struct {
		name       string
		numTexts   int
		shouldFail bool
		expected   expected
	}{
		{
			name:     "exact batch size produces single batch with correct order",
			numTexts: 40,
		},
		{
			name:     "multiple batches preserves order (85 texts = 3 batches of 40+40+5)",
			numTexts: 85,
		},
		{
			name:     "boundary case of 41 texts splits into 2 batches (40+1)",
			numTexts: 41,
		},
		{
			name:       "returns error when a batch fails",
			numTexts:   80,
			shouldFail: true,
			expected: expected{
				err: "failed to generate embeddings for batch",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			texts, ec := makeTextsAndMock(tt.numTexts, tt.shouldFail)
			pipeline := newTestPipeline(ec, nil, &vectorstore.MockVectorStore{})

			result, err := pipeline.generateEmbeddingParallel(texts)

			if tt.expected.err != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expected.err)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.Len(t, result, tt.numTexts)
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

	numTexts := 200
	texts := make([]string, numTexts)
	for i := range texts {
		texts[i] = fmt.Sprintf("text-%d", i)
	}

	ec := &mockEmbeddingCreator{
		newFunc: func(_ context.Context, body openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
			cur := currentConcurrency.Add(1)
			for {
				peak := peakConcurrency.Load()
				if cur <= peak || peakConcurrency.CompareAndSwap(peak, cur) {
					break
				}
			}

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

	pipeline := newTestPipeline(ec, nil, &vectorstore.MockVectorStore{})
	result, err := pipeline.generateEmbeddingParallel(texts)

	assert.NoError(t, err)
	assert.Len(t, result, numTexts)
	assert.LessOrEqual(t, int(peakConcurrency.Load()), maxConcurrency,
		"peak concurrency %d should not exceed maxConcurrency %d", peakConcurrency.Load(), maxConcurrency)
}

func TestGenerateResponse(t *testing.T) {
	type expected struct {
		answer string
		err    string
	}
	type mock struct {
		response *openai.ChatCompletion
		err      error
	}

	tests := []struct {
		name        string
		contextInfo string
		question    string
		mock        mock
		expected    expected
	}{
		{
			name:        "returns response content from first choice",
			contextInfo: "Go is a compiled language.",
			question:    "What is Go?",
			mock: mock{
				response: makeChatCompletion("Go is a compiled programming language."),
			},
			expected: expected{
				answer: "Go is a compiled programming language.",
			},
		},
		{
			name:        "wraps API error",
			contextInfo: "ctx",
			question:    "q",
			mock: mock{
				err: errors.New("deepseek timeout"),
			},
			expected: expected{
				err: "failed to generate response",
			},
		},
		{
			name:        "returns error when choices are empty",
			contextInfo: "ctx",
			question:    "q",
			mock: mock{
				response: &openai.ChatCompletion{Choices: []openai.ChatCompletionChoice{}},
			},
			expected: expected{
				err: "no response from DeepSeek API",
			},
		},
		{
			name:        "prompt contains context and question",
			contextInfo: "Rust is memory safe.",
			question:    "Is Rust safe?",
			mock: mock{
				response: makeChatCompletion("Yes"),
			},
			expected: expected{
				answer: "Yes",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var capturedPrompt string
			cc := &mockChatCompleter{
				newFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
					if len(body.Messages) > 0 {
						capturedPrompt = body.Messages[0].OfUser.Content.OfString.Value
					}
					return tt.mock.response, tt.mock.err
				},
			}
			pipeline := newTestPipeline(nil, cc, &vectorstore.MockVectorStore{})

			result, err := pipeline.generateResponse(tt.contextInfo, tt.question)

			if tt.expected.err != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expected.err)
				assert.Empty(t, result)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected.answer, result)
			}

			if tt.name == "prompt contains context and question" {
				assert.Contains(t, capturedPrompt, "Context information:")
				assert.Contains(t, capturedPrompt, tt.contextInfo)
				assert.Contains(t, capturedPrompt, "Question: "+tt.question)
			}
		})
	}
}

func TestProcessDocument(t *testing.T) {
	type expected struct {
		chunks int
		err    string
	}
	type mock struct {
		response *openai.CreateEmbeddingResponse
		err      error
	}

	tests := []struct {
		name     string
		content  string
		metadata map[string]string
		mock     mock
		expected expected
	}{
		{
			name:     "short content produces single chunk via batch path",
			content:  "hello world",
			metadata: map[string]string{"source": "doc1"},
			mock: mock{
				response: makeEmbeddingResponse([][]float64{{0.1, 0.2}}),
			},
			expected: expected{
				chunks: 1,
			},
		},
		{
			name:     "multiple chunks produced from medium content",
			content:  strings.Repeat("a", 2500),
			metadata: map[string]string{"source": "doc2"},
			mock: mock{
				response: makeEmbeddingResponse([][]float64{{0.1}, {0.2}, {0.3}}),
			},
			expected: expected{
				chunks: 3,
			},
		},
		{
			name:     "embedding error is wrapped",
			content:  "some text",
			metadata: map[string]string{"source": "err"},
			mock: mock{
				err: errors.New("api failure"),
			},
			expected: expected{
				err: "failed to generate embeddings",
			},
		},
		{
			name:     "empty content produces single chunk",
			content:  "",
			metadata: map[string]string{"source": "empty"},
			mock: mock{
				response: makeEmbeddingResponse([][]float64{{0.0}}),
			},
			expected: expected{
				chunks: 1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ec := &mockEmbeddingCreator{
				newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
					return tt.mock.response, tt.mock.err
				},
			}
			pipeline := newTestPipeline(ec, nil, &vectorstore.MockVectorStore{})

			chunks, err := pipeline.ProcessDocument(tt.content, tt.metadata)

			if tt.expected.err != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expected.err)
				assert.Nil(t, chunks)
			} else {
				assert.NoError(t, err)
				assert.Len(t, chunks, tt.expected.chunks)

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
	pipeline := newTestPipeline(ec, nil, &vectorstore.MockVectorStore{})

	chunks, err := pipeline.ProcessDocument(content, metadata)

	assert.NoError(t, err)
	assert.Greater(t, len(chunks), maxBatchSize, "should have more than maxBatchSize chunks to trigger parallel path")
	assert.Greater(t, int(callCount.Load()), 1, "parallel path should call embedding API multiple times")
}

func TestAddDocumentToVectorStore(t *testing.T) {
	sampleChunks := []types.DocumentChunk{
		{ID: "c1", Content: "hello", Embedding: []float64{0.1}},
		{ID: "c2", Content: "world", Embedding: []float64{0.2}},
	}

	type expected struct {
		err string
	}
	type mock struct {
		storeErr error
	}

	tests := []struct {
		name     string
		chunks   []types.DocumentChunk
		mock     mock
		expected expected
	}{
		{
			name:   "stores chunks successfully",
			chunks: sampleChunks,
		},
		{
			name:   "wraps store error",
			chunks: sampleChunks,
			mock: mock{
				storeErr: errors.New("disk full"),
			},
			expected: expected{
				err: "failed to store chunks",
			},
		},
		{
			name:   "handles empty chunks slice",
			chunks: []types.DocumentChunk{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var storedChunks []types.DocumentChunk
			vs := &vectorstore.MockVectorStore{
				StoreFunc: func(chunks []types.DocumentChunk) error {
					storedChunks = chunks
					return tt.mock.storeErr
				},
			}
			pipeline := newTestPipeline(nil, nil, vs)

			err := pipeline.AddDocumentToVectorStore(tt.chunks)

			if tt.expected.err != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expected.err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.chunks, storedChunks)
			}
		})
	}
}

func TestQuery(t *testing.T) {
	type embeddingMock struct {
		response *openai.CreateEmbeddingResponse
		err      error
	}
	type searchMock struct {
		result []types.ScoredChunk
		err    error
	}
	type chatMock struct {
		response *openai.ChatCompletion
		err      error
	}
	type mock struct {
		embedding embeddingMock
		search    searchMock
		chat      chatMock
	}
	type expected struct {
		answer  string
		sources int
		err     string
	}

	tests := []struct {
		name     string
		question string
		mock     mock
		expected expected
	}{
		{
			name:     "full pipeline success with single source",
			question: "What is Go?",
			mock: mock{
				embedding: embeddingMock{response: makeEmbeddingResponse([][]float64{{0.5, 0.6}})},
				search: searchMock{result: []types.ScoredChunk{
					{Chunk: types.DocumentChunk{ID: "c1", Content: "Go is a language"}, Score: 0.9},
				}},
				chat: chatMock{response: makeChatCompletion("Go is a compiled language.")},
			},
			expected: expected{
				answer:  "Go is a compiled language.",
				sources: 1,
			},
		},
		{
			name:     "multiple sources joined with double newline separator",
			question: "Tell me about Go",
			mock: mock{
				embedding: embeddingMock{response: makeEmbeddingResponse([][]float64{{0.5}})},
				search: searchMock{result: []types.ScoredChunk{
					{Chunk: types.DocumentChunk{ID: "c1", Content: "Go is compiled"}, Score: 0.9},
					{Chunk: types.DocumentChunk{ID: "c2", Content: "Go has goroutines"}, Score: 0.8},
					{Chunk: types.DocumentChunk{ID: "c3", Content: "Go is statically typed"}, Score: 0.7},
				}},
				chat: chatMock{response: makeChatCompletion("Go is great.")},
			},
			expected: expected{
				answer:  "Go is great.",
				sources: 3,
			},
		},
		{
			name:     "returns error when embedding generation fails",
			question: "fail",
			mock: mock{
				embedding: embeddingMock{err: errors.New("openai down")},
			},
			expected: expected{
				err: "failed to generate embedding for query",
			},
		},
		{
			name:     "returns error when vector store search fails",
			question: "search fail",
			mock: mock{
				embedding: embeddingMock{response: makeEmbeddingResponse([][]float64{{0.1}})},
				search:    searchMock{err: errors.New("search broken")},
			},
			expected: expected{
				err: "failed to search vector store",
			},
		},
		{
			name:     "returns error when response generation fails",
			question: "resp fail",
			mock: mock{
				embedding: embeddingMock{response: makeEmbeddingResponse([][]float64{{0.1}})},
				search:    searchMock{result: []types.ScoredChunk{{Chunk: types.DocumentChunk{Content: "ctx"}, Score: 0.5}}},
				chat:      chatMock{err: errors.New("deepseek timeout")},
			},
			expected: expected{
				err: "failed to generate response",
			},
		},
		{
			name:     "handles no search results with empty context",
			question: "obscure topic",
			mock: mock{
				embedding: embeddingMock{response: makeEmbeddingResponse([][]float64{{0.1}})},
				search:    searchMock{result: []types.ScoredChunk{}},
				chat:      chatMock{response: makeChatCompletion("I don't have enough information.")},
			},
			expected: expected{
				answer:  "I don't have enough information.",
				sources: 0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ec := &mockEmbeddingCreator{
				newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
					return tt.mock.embedding.response, tt.mock.embedding.err
				},
			}

			var capturedContext string
			cc := &mockChatCompleter{
				newFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
					if len(body.Messages) > 0 {
						capturedContext = body.Messages[0].OfUser.Content.OfString.Value
					}
					return tt.mock.chat.response, tt.mock.chat.err
				},
			}

			vs := &vectorstore.MockVectorStore{
				SearchFunc: func(embedding []float64, limit int) ([]types.ScoredChunk, error) {
					return tt.mock.search.result, tt.mock.search.err
				},
			}

			pipeline := newTestPipeline(ec, cc, vs)
			result, err := pipeline.Query(tt.question)

			if tt.expected.err != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expected.err)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				assert.Equal(t, tt.expected.answer, result.Answer)
				assert.Len(t, result.Sources, tt.expected.sources)
				assert.Equal(t, defaultConfidence, result.Confidence)

				for _, sc := range tt.mock.search.result {
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

	vs := &vectorstore.MockVectorStore{
		SearchFunc: func(_ []float64, _ int) ([]types.ScoredChunk, error) {
			return []types.ScoredChunk{
				{Chunk: types.DocumentChunk{Content: "First chunk"}, Score: 0.9},
				{Chunk: types.DocumentChunk{Content: "Second chunk"}, Score: 0.8},
			}, nil
		},
	}

	pipeline := newTestPipeline(ec, cc, vs)
	_, err := pipeline.Query("test question")

	assert.NoError(t, err)
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
	vs := &vectorstore.MockVectorStore{
		SearchFunc: func(_ []float64, limit int) ([]types.ScoredChunk, error) {
			capturedLimit = limit
			return []types.ScoredChunk{}, nil
		},
	}

	pipeline := newTestPipeline(ec, cc, vs)
	_, err := pipeline.Query("test")

	assert.NoError(t, err)
	assert.Equal(t, maxContentChunks, capturedLimit)
}
