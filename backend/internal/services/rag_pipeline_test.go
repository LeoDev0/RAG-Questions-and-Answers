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
		data[i] = openai.Embedding{Index: int64(i), Embedding: emb}
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
		{
			name:  "reorders out-of-order response by Index",
			texts: []string{"a", "b", "c"},
			mock: mock{
				response: &openai.CreateEmbeddingResponse{Data: []openai.Embedding{
					{Index: 2, Embedding: []float64{0.3}},
					{Index: 0, Embedding: []float64{0.1}},
					{Index: 1, Embedding: []float64{0.2}},
				}},
			},
			expected: expected{
				result: [][]float64{{0.1}, {0.2}, {0.3}},
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
					_, _ = fmt.Sscanf(txt, "text-%d", &idx)
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

	numTexts := 400
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
			var capturedSystem string
			var capturedQuestion string
			cc := &mockChatCompleter{
				newFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
					if len(body.Messages) >= 2 {
						capturedSystem = body.Messages[0].OfSystem.Content.OfString.Value
						capturedQuestion = body.Messages[len(body.Messages)-1].OfUser.Content.OfString.Value
					}
					return tt.mock.response, tt.mock.err
				},
			}
			pipeline := newTestPipeline(nil, cc, &vectorstore.MockVectorStore{})

			result, err := pipeline.generateResponse(tt.contextInfo, nil, tt.question)

			if tt.expected.err != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expected.err)
				assert.Empty(t, result)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected.answer, result)
			}

			if tt.name == "prompt contains context and question" {
				assert.Contains(t, capturedSystem, "Context information:")
				assert.Contains(t, capturedSystem, tt.contextInfo)
				assert.Equal(t, tt.question, capturedQuestion)
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
					assert.Equal(t, tt.metadata["source"], chunk.Source, "chunk %d should carry source", i)
					assert.Equal(t, i, chunk.ChunkIndex, "chunk %d should be indexed in order", i)
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

func TestProcessDocument_CharOffsetsLocateChunksInNormalizedText(t *testing.T) {
	type expected struct {
		multiChunk bool
	}

	tests := []struct {
		name     string
		content  string
		expected expected
	}{
		{
			name:     "single chunk spans its trimmed content",
			content:  "hello world",
			expected: expected{multiChunk: false},
		},
		{
			name:     "multi chunk offsets stay monotonic and locate content",
			content:  strings.Repeat("alpha beta gamma delta. ", 200),
			expected: expected{multiChunk: true},
		},
		{
			name:     "unicode content keeps valid byte offsets",
			content:  strings.Repeat("これはテストです。日本語の文章を分割します。", 80),
			expected: expected{multiChunk: true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ec := &mockEmbeddingCreator{
				newFunc: func(_ context.Context, body openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
					n := len(body.Input.OfArrayOfStrings)
					embeddings := make([][]float64, n)
					for i := range embeddings {
						embeddings[i] = []float64{0.1}
					}
					return makeEmbeddingResponse(embeddings), nil
				},
			}
			pipeline := newTestPipeline(ec, nil, &vectorstore.MockVectorStore{})

			normalized := utils.Normalize(tt.content)
			chunks, err := pipeline.ProcessDocument(tt.content, map[string]string{"source": "doc"})
			assert.NoError(t, err)

			if tt.expected.multiChunk {
				assert.Greater(t, len(chunks), 1)
			}

			prevStart := -1
			for i, chunk := range chunks {
				assert.GreaterOrEqual(t, chunk.StartOffset, 0)
				assert.LessOrEqual(t, chunk.EndOffset, len(normalized))
				assert.Equal(t, len(chunk.Content), chunk.EndOffset-chunk.StartOffset)
				assert.Equal(t, chunk.Content, normalized[chunk.StartOffset:chunk.EndOffset], "chunk %d offsets must locate its content", i)
				assert.Greater(t, chunk.StartOffset, prevStart, "chunk %d start must be strictly monotonic", i)
				prevStart = chunk.StartOffset
			}
		})
	}
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
		answer     string
		sources    int
		confidence float64
		err        string
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
				answer:     "Go is a compiled language.",
				sources:    1,
				confidence: 0.9,
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
				answer:     "Go is great.",
				sources:    3,
				confidence: 0.9,
			},
		},
		{
			name:     "drops chunks below the similarity threshold",
			question: "What is Go?",
			mock: mock{
				embedding: embeddingMock{response: makeEmbeddingResponse([][]float64{{0.5}})},
				search: searchMock{result: []types.ScoredChunk{
					{Chunk: types.DocumentChunk{ID: "c1", Content: "Go is compiled"}, Score: 0.6},
					{Chunk: types.DocumentChunk{ID: "c2", Content: "weak match"}, Score: 0.2},
				}},
				chat: chatMock{response: makeChatCompletion("Go is great.")},
			},
			expected: expected{
				answer:     "Go is great.",
				sources:    1,
				confidence: 0.6,
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
				answer:     "I don't have enough information.",
				sources:    0,
				confidence: 0.0,
			},
		},
		{
			name:     "all chunks below threshold yields empty context and zero confidence",
			question: "off topic",
			mock: mock{
				embedding: embeddingMock{response: makeEmbeddingResponse([][]float64{{0.1}})},
				search: searchMock{result: []types.ScoredChunk{
					{Chunk: types.DocumentChunk{ID: "c1", Content: "irrelevant one"}, Score: 0.25},
					{Chunk: types.DocumentChunk{ID: "c2", Content: "irrelevant two"}, Score: 0.1},
				}},
				chat: chatMock{response: makeChatCompletion("I don't have enough information.")},
			},
			expected: expected{
				answer:     "I don't have enough information.",
				sources:    0,
				confidence: 0.0,
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
						capturedContext = body.Messages[0].OfSystem.Content.OfString.Value
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
			result, err := pipeline.Query(tt.question, nil)

			if tt.expected.err != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expected.err)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				assert.Equal(t, tt.expected.answer, result.Answer)
				assert.Len(t, result.Sources, tt.expected.sources)
				assert.Equal(t, tt.expected.confidence, result.Confidence)

				// Search results are sorted by descending score, so the first
				// expected.sources are retained and the rest are filtered out.
				for i, sc := range tt.mock.search.result {
					if i < tt.expected.sources {
						assert.Contains(t, capturedContext, sc.Chunk.Content)
					} else {
						assert.NotContains(t, capturedContext, sc.Chunk.Content)
					}
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
				capturedPrompt = body.Messages[0].OfSystem.Content.OfString.Value
			}
			return makeChatCompletion("answer"), nil
		},
	}

	vs := &vectorstore.MockVectorStore{
		SearchFunc: func(_ []float64, _ int) ([]types.ScoredChunk, error) {
			return []types.ScoredChunk{
				{Chunk: types.DocumentChunk{ID: "doc-chunk-0", Source: "doc", ChunkIndex: 0, Content: "First chunk"}, Score: 0.9},
				{Chunk: types.DocumentChunk{ID: "doc-chunk-1", Source: "doc", ChunkIndex: 1, Content: "Second chunk"}, Score: 0.8},
			}, nil
		},
	}

	pipeline := newTestPipeline(ec, cc, vs)
	_, err := pipeline.Query("test question", nil)

	assert.NoError(t, err)
	assert.Contains(t, capturedPrompt, "First chunk\n\nSecond chunk")
}

func TestRetainAboveThreshold(t *testing.T) {
	chunk := func(id string, score float64) types.ScoredChunk {
		return types.ScoredChunk{Chunk: types.DocumentChunk{ID: id}, Score: score}
	}

	type expected struct {
		ids []string
	}

	tests := []struct {
		name     string
		input    []types.ScoredChunk
		expected expected
	}{
		{
			name:     "empty input",
			input:    []types.ScoredChunk{},
			expected: expected{ids: []string{}},
		},
		{
			name:     "all above threshold are kept",
			input:    []types.ScoredChunk{chunk("a", 0.9), chunk("b", 0.5), chunk("c", 0.3)},
			expected: expected{ids: []string{"a", "b", "c"}},
		},
		{
			name:     "boundary score equal to threshold is kept",
			input:    []types.ScoredChunk{chunk("a", 0.9), chunk("b", similarityThreshold)},
			expected: expected{ids: []string{"a", "b"}},
		},
		{
			name:     "keeps qualifying chunks regardless of order",
			input:    []types.ScoredChunk{chunk("a", 0.8), chunk("b", 0.29), chunk("c", 0.5)},
			expected: expected{ids: []string{"a", "c"}},
		},
		{
			name:     "all below threshold yields nothing",
			input:    []types.ScoredChunk{chunk("a", 0.2), chunk("b", 0.1)},
			expected: expected{ids: []string{}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			retained := retainAboveThreshold(tt.input)

			ids := make([]string, 0, len(retained))
			for _, sc := range retained {
				ids = append(ids, sc.Chunk.ID)
			}
			assert.Equal(t, tt.expected.ids, ids)
		})
	}
}

func TestConfidenceFromTopScore(t *testing.T) {
	score := func(s float64) types.ScoredChunk {
		return types.ScoredChunk{Score: s}
	}

	tests := []struct {
		name     string
		input    []types.ScoredChunk
		expected float64
	}{
		{
			name:     "empty yields zero",
			input:    []types.ScoredChunk{},
			expected: 0.0,
		},
		{
			name:     "single score is returned",
			input:    []types.ScoredChunk{score(0.42)},
			expected: 0.42,
		},
		{
			name:     "returns the max of a descending slice",
			input:    []types.ScoredChunk{score(0.9), score(0.5), score(0.3)},
			expected: 0.9,
		},
		{
			name:     "returns the max regardless of order",
			input:    []types.ScoredChunk{score(0.3), score(0.9), score(0.5)},
			expected: 0.9,
		},
		{
			name:     "score above one clamps to one",
			input:    []types.ScoredChunk{score(1.4)},
			expected: 1.0,
		},
		{
			name:     "negative score clamps to zero",
			input:    []types.ScoredChunk{score(-0.2)},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.InDelta(t, tt.expected, confidenceFromTopScore(tt.input), 1e-9)
		})
	}
}

func TestQuery_NeighborsExpandedIntoContextInDocumentOrder(t *testing.T) {
	ec := &mockEmbeddingCreator{
		newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
			return makeEmbeddingResponse([][]float64{{0.1}}), nil
		},
	}

	var capturedPrompt string
	cc := &mockChatCompleter{
		newFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
			if len(body.Messages) > 0 {
				capturedPrompt = body.Messages[0].OfSystem.Content.OfString.Value
			}
			return makeChatCompletion("answer"), nil
		},
	}

	doc := []types.DocumentChunk{
		{ID: "doc-chunk-0", Source: "doc", ChunkIndex: 0, Content: "chunk zero"},
		{ID: "doc-chunk-1", Source: "doc", ChunkIndex: 1, Content: "chunk one"},
		{ID: "doc-chunk-2", Source: "doc", ChunkIndex: 2, Content: "chunk two"},
		{ID: "doc-chunk-3", Source: "doc", ChunkIndex: 3, Content: "chunk three"},
	}
	neighborsOf := func(index, radius int) []types.DocumentChunk {
		lo, hi := index-radius, index+radius
		out := make([]types.DocumentChunk, 0, len(doc))
		for _, c := range doc {
			if c.ChunkIndex >= lo && c.ChunkIndex <= hi {
				out = append(out, c)
			}
		}
		return out
	}

	vs := &vectorstore.MockVectorStore{
		SearchFunc: func(_ []float64, _ int) ([]types.ScoredChunk, error) {
			return []types.ScoredChunk{{Chunk: doc[2], Score: 0.9}}, nil
		},
		NeighborsFunc: func(source string, index, radius int) ([]types.DocumentChunk, error) {
			assert.Equal(t, "doc", source)
			return neighborsOf(index, radius), nil
		},
	}

	pipeline := newTestPipeline(ec, cc, vs)
	result, err := pipeline.Query("test question", nil)

	assert.NoError(t, err)
	assert.Contains(t, capturedPrompt, "chunk one\n\nchunk two\n\nchunk three")
	assert.NotContains(t, capturedPrompt, "chunk zero")
	assert.Len(t, result.Sources, 1, "sources stay the original hits, not the expanded neighbors")
	assert.Equal(t, "doc-chunk-2", result.Sources[0].ID)
}

func TestQuery_OverlappingNeighborWindowsDedupedAndDocumentsNotInterleaved(t *testing.T) {
	ec := &mockEmbeddingCreator{
		newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
			return makeEmbeddingResponse([][]float64{{0.1}}), nil
		},
	}

	var capturedPrompt string
	cc := &mockChatCompleter{
		newFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
			if len(body.Messages) > 0 {
				capturedPrompt = body.Messages[0].OfSystem.Content.OfString.Value
			}
			return makeChatCompletion("answer"), nil
		},
	}

	chunks := map[string][]types.DocumentChunk{
		"a": {
			{ID: "a-chunk-0", Source: "a", ChunkIndex: 0, Content: "a0"},
			{ID: "a-chunk-1", Source: "a", ChunkIndex: 1, Content: "a1"},
			{ID: "a-chunk-2", Source: "a", ChunkIndex: 2, Content: "a2"},
		},
		"b": {
			{ID: "b-chunk-0", Source: "b", ChunkIndex: 0, Content: "b0"},
			{ID: "b-chunk-1", Source: "b", ChunkIndex: 1, Content: "b1"},
		},
	}

	vs := &vectorstore.MockVectorStore{
		SearchFunc: func(_ []float64, _ int) ([]types.ScoredChunk, error) {
			return []types.ScoredChunk{
				{Chunk: chunks["a"][0], Score: 0.9},
				{Chunk: chunks["a"][2], Score: 0.8},
				{Chunk: chunks["b"][1], Score: 0.7},
			}, nil
		},
		NeighborsFunc: func(source string, index, radius int) ([]types.DocumentChunk, error) {
			lo, hi := index-radius, index+radius
			out := make([]types.DocumentChunk, 0, len(chunks[source]))
			for _, c := range chunks[source] {
				if c.ChunkIndex >= lo && c.ChunkIndex <= hi {
					out = append(out, c)
				}
			}
			return out, nil
		},
	}

	pipeline := newTestPipeline(ec, cc, vs)
	_, err := pipeline.Query("test question", nil)

	assert.NoError(t, err)
	// Hits a0 and a2 both pull a1 as a neighbor; it must appear exactly once,
	// and each document's chunks stay contiguous and in order.
	assert.Contains(t, capturedPrompt, "a0\n\na1\n\na2\n\nb0\n\nb1")
	assert.Equal(t, 1, strings.Count(capturedPrompt, "a1"))
}

func TestAssembleContext(t *testing.T) {
	type expected struct {
		context string
	}

	tests := []struct {
		name     string
		chunks   []types.DocumentChunk
		expected expected
	}{
		{
			name: "splices out duplicated overlap between adjacent chunks",
			chunks: []types.DocumentChunk{
				{Content: "Hello world foo", Source: "d", StartOffset: 0, EndOffset: 15},
				{Content: "world foo bar baz", Source: "d", StartOffset: 6, EndOffset: 23},
			},
			expected: expected{context: "Hello world foo bar baz"},
		},
		{
			name: "keeps a separator across a non-adjacent gap",
			chunks: []types.DocumentChunk{
				{Content: "Alpha block", Source: "d", StartOffset: 0, EndOffset: 11},
				{Content: "Gamma block", Source: "d", StartOffset: 40, EndOffset: 51},
			},
			expected: expected{context: "Alpha block\n\nGamma block"},
		},
		{
			name: "skips a fully contained chunk",
			chunks: []types.DocumentChunk{
				{Content: "abcdefghij", Source: "d", StartOffset: 0, EndOffset: 10},
				{Content: "cdef", Source: "d", StartOffset: 2, EndOffset: 6},
			},
			expected: expected{context: "abcdefghij"},
		},
		{
			name: "falls back to a separator when offsets are missing",
			chunks: []types.DocumentChunk{
				{Content: "one", Source: "d"},
				{Content: "two", Source: "d"},
			},
			expected: expected{context: "one\n\ntwo"},
		},
		{
			name: "never merges across a source boundary",
			chunks: []types.DocumentChunk{
				{Content: "doc a part", Source: "a", StartOffset: 0, EndOffset: 10},
				{Content: "doc b part", Source: "b", StartOffset: 0, EndOffset: 10},
			},
			expected: expected{context: "doc a part\n\ndoc b part"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected.context, assembleContext(tt.chunks))
		})
	}
}

func TestSelectWithinBudget(t *testing.T) {
	hit := func(id string, size int) types.DocumentChunk {
		return types.DocumentChunk{ID: id, Content: strings.Repeat("x", size)}
	}

	type input struct {
		ordered []types.DocumentChunk
		hits    []string
		budget  int
	}
	type expected struct {
		ids []string
	}

	ordered := []types.DocumentChunk{
		hit("h1", 10), hit("n1", 10), hit("h2", 10), hit("n2", 10),
	}

	tests := []struct {
		name     string
		input    input
		expected expected
	}{
		{
			name:     "generous budget keeps every chunk",
			input:    input{ordered: ordered, hits: []string{"h1", "h2"}, budget: 1000},
			expected: expected{ids: []string{"h1", "n1", "h2", "n2"}},
		},
		{
			name:     "tight budget drops neighbors but keeps all hits",
			input:    input{ordered: ordered, hits: []string{"h1", "h2"}, budget: 25},
			expected: expected{ids: []string{"h1", "h2"}},
		},
		{
			name:     "hits exceeding the budget are still kept",
			input:    input{ordered: ordered, hits: []string{"h1", "h2"}, budget: 0},
			expected: expected{ids: []string{"h1", "h2"}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hitIDs := make(map[string]bool, len(tt.input.hits))
			for _, id := range tt.input.hits {
				hitIDs[id] = true
			}

			selected := selectWithinBudget(tt.input.ordered, hitIDs, tt.input.budget)

			ids := make([]string, len(selected))
			for i, c := range selected {
				ids[i] = c.ID
			}
			assert.Equal(t, tt.expected.ids, ids)
		})
	}
}

func TestTrimHistory(t *testing.T) {
	makeHistory := func(n int) []types.Message {
		h := make([]types.Message, n)
		for i := range h {
			h[i] = types.Message{Role: types.RoleUser, Content: fmt.Sprintf("m%d", i)}
		}
		return h
	}

	tests := []struct {
		name     string
		input    []types.Message
		expected int
	}{
		{
			name:     "below cap is unchanged",
			input:    makeHistory(3),
			expected: 3,
		},
		{
			name:     "at cap is unchanged",
			input:    makeHistory(maxHistoryTurns),
			expected: maxHistoryTurns,
		},
		{
			name:     "above cap is trimmed to last N",
			input:    makeHistory(maxHistoryTurns + 5),
			expected: maxHistoryTurns,
		},
		{
			name:     "empty stays empty",
			input:    nil,
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := trimHistory(tt.input)
			assert.Len(t, result, tt.expected)
			if len(result) > 0 && len(tt.input) > maxHistoryTurns {
				assert.Equal(t, tt.input[len(tt.input)-maxHistoryTurns].Content, result[0].Content)
				assert.Equal(t, tt.input[len(tt.input)-1].Content, result[len(result)-1].Content)
			}
		})
	}
}

func TestRewriteQueryForRetrieval(t *testing.T) {
	tests := []struct {
		name     string
		history  []types.Message
		question string
		expected string
	}{
		{
			name:     "empty history returns just the question",
			history:  nil,
			question: "what now",
			expected: "what now",
		},
		{
			name: "one prior user turn is concatenated",
			history: []types.Message{
				{Role: types.RoleUser, Content: "first"},
			},
			question: "second",
			expected: "first second",
		},
		{
			name: "assistant turns are skipped",
			history: []types.Message{
				{Role: types.RoleUser, Content: "u1"},
				{Role: types.RoleAssistant, Content: "a1"},
			},
			question: "u2",
			expected: "u1 u2",
		},
		{
			name: "only the last retrievalRewriteWindow user turns are picked",
			history: []types.Message{
				{Role: types.RoleUser, Content: "old"},
				{Role: types.RoleAssistant, Content: "a"},
				{Role: types.RoleUser, Content: "u-2"},
				{Role: types.RoleAssistant, Content: "a"},
				{Role: types.RoleUser, Content: "u-1"},
				{Role: types.RoleAssistant, Content: "a"},
			},
			question: "now",
			expected: "u-2 u-1 now",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := rewriteQueryForRetrieval(tt.history, tt.question)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestChatCompletionParams_MessageOrder(t *testing.T) {
	type expected struct {
		roles []string
		texts []string
	}
	tests := []struct {
		name            string
		history         []types.Message
		question        string
		expectedMessage expected
	}{
		{
			name:     "empty history yields system then user",
			history:  nil,
			question: "hello",
			expectedMessage: expected{
				roles: []string{"system", "user"},
				texts: []string{"system-prompt", "hello"},
			},
		},
		{
			name: "interleaved history preserved between system and final user",
			history: []types.Message{
				{Role: types.RoleUser, Content: "q1"},
				{Role: types.RoleAssistant, Content: "a1"},
				{Role: types.RoleUser, Content: "q2"},
				{Role: types.RoleAssistant, Content: "a2"},
			},
			question: "q3",
			expectedMessage: expected{
				roles: []string{"system", "user", "assistant", "user", "assistant", "user"},
				texts: []string{"system-prompt", "q1", "a1", "q2", "a2", "q3"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params := chatCompletionParams("system-prompt", tt.history, tt.question)
			assert.Len(t, params.Messages, len(tt.expectedMessage.roles))
			for i, want := range tt.expectedMessage.roles {
				m := params.Messages[i]
				switch want {
				case "system":
					assert.NotNil(t, m.OfSystem, "expected system message at %d", i)
					assert.Equal(t, tt.expectedMessage.texts[i], m.OfSystem.Content.OfString.Value)
				case "user":
					assert.NotNil(t, m.OfUser, "expected user message at %d", i)
					assert.Equal(t, tt.expectedMessage.texts[i], m.OfUser.Content.OfString.Value)
				case "assistant":
					assert.NotNil(t, m.OfAssistant, "expected assistant message at %d", i)
					assert.Equal(t, tt.expectedMessage.texts[i], m.OfAssistant.Content.OfString.Value)
				}
			}
		})
	}
}

func TestQuery_HistoryThreadedToLLMAndRetrieval(t *testing.T) {
	type expected struct {
		messageRoles []string
		messageTexts []string
		embedInput   string
	}

	tests := []struct {
		name     string
		history  []types.Message
		question string
		expected expected
	}{
		{
			name:     "no history sends single user message after system, embeds question only",
			history:  nil,
			question: "what is Go",
			expected: expected{
				messageRoles: []string{"system", "user"},
				messageTexts: []string{"ctx", "what is Go"},
				embedInput:   "what is Go",
			},
		},
		{
			name: "short history forwarded in order, embedding includes prior user turn",
			history: []types.Message{
				{Role: types.RoleUser, Content: "what sections"},
				{Role: types.RoleAssistant, Content: "Setup, Usage, Troubleshooting"},
			},
			question: "tell me about the second one",
			expected: expected{
				messageRoles: []string{"system", "user", "assistant", "user"},
				messageTexts: []string{"ctx", "what sections", "Setup, Usage, Troubleshooting", "tell me about the second one"},
				embedInput:   "what sections tell me about the second one",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var capturedEmbed string
			ec := &mockEmbeddingCreator{
				newFunc: func(_ context.Context, body openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
					capturedEmbed = body.Input.OfString.Value
					return makeEmbeddingResponse([][]float64{{0.1}}), nil
				},
			}

			var capturedMsgs []openai.ChatCompletionMessageParamUnion
			cc := &mockChatCompleter{
				newFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
					capturedMsgs = body.Messages
					return makeChatCompletion("ok"), nil
				},
			}

			vs := &vectorstore.MockVectorStore{
				SearchFunc: func(_ []float64, _ int) ([]types.ScoredChunk, error) {
					return []types.ScoredChunk{
						{Chunk: types.DocumentChunk{Content: "ctx"}, Score: 0.9},
					}, nil
				},
			}

			pipeline := newTestPipeline(ec, cc, vs)
			_, err := pipeline.Query(tt.question, tt.history)
			assert.NoError(t, err)

			assert.Equal(t, tt.expected.embedInput, capturedEmbed)
			assert.Len(t, capturedMsgs, len(tt.expected.messageRoles))
			for i, role := range tt.expected.messageRoles {
				m := capturedMsgs[i]
				switch role {
				case "system":
					assert.NotNil(t, m.OfSystem)
					assert.Contains(t, m.OfSystem.Content.OfString.Value, tt.expected.messageTexts[i])
				case "user":
					assert.NotNil(t, m.OfUser)
					assert.Equal(t, tt.expected.messageTexts[i], m.OfUser.Content.OfString.Value)
				case "assistant":
					assert.NotNil(t, m.OfAssistant)
					assert.Equal(t, tt.expected.messageTexts[i], m.OfAssistant.Content.OfString.Value)
				}
			}
		})
	}
}

func TestQuery_HistoryBeyondCapIsTrimmed(t *testing.T) {
	history := make([]types.Message, maxHistoryTurns+4)
	for i := range history {
		history[i] = types.Message{Role: types.RoleUser, Content: fmt.Sprintf("m%d", i)}
	}

	ec := &mockEmbeddingCreator{
		newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
			return makeEmbeddingResponse([][]float64{{0.1}}), nil
		},
	}

	var capturedMsgs []openai.ChatCompletionMessageParamUnion
	cc := &mockChatCompleter{
		newFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) (*openai.ChatCompletion, error) {
			capturedMsgs = body.Messages
			return makeChatCompletion("ok"), nil
		},
	}

	vs := &vectorstore.MockVectorStore{
		SearchFunc: func(_ []float64, _ int) ([]types.ScoredChunk, error) {
			return []types.ScoredChunk{}, nil
		},
	}

	pipeline := newTestPipeline(ec, cc, vs)
	_, err := pipeline.Query("now", history)
	assert.NoError(t, err)

	assert.Len(t, capturedMsgs, maxHistoryTurns+2)
	assert.NotNil(t, capturedMsgs[0].OfSystem)
	assert.Equal(t, "now", capturedMsgs[len(capturedMsgs)-1].OfUser.Content.OfString.Value)
	// Oldest retained history turn is m4 (we trimmed the first 4).
	assert.Equal(t, "m4", capturedMsgs[1].OfUser.Content.OfString.Value)
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
	_, err := pipeline.Query("test", nil)

	assert.NoError(t, err)
	assert.Equal(t, maxContentChunks, capturedLimit)
}
