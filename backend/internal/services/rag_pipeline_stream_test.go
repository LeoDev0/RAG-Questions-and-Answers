package services

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/stretchr/testify/assert"

	"rag-backend/internal/repositories/vectorstore"
	"rag-backend/pkg/types"
)

func makeChatCompletionChunk(content string) openai.ChatCompletionChunk {
	return openai.ChatCompletionChunk{
		Choices: []openai.ChatCompletionChunkChoice{
			{Delta: openai.ChatCompletionChunkChoiceDelta{Content: content}},
		},
	}
}

func drainEvents(t *testing.T, ch <-chan StreamEvent) []StreamEvent {
	t.Helper()
	var events []StreamEvent
	timeout := time.After(2 * time.Second)
	for {
		select {
		case ev, ok := <-ch:
			if !ok {
				return events
			}
			events = append(events, ev)
		case <-timeout:
			t.Fatalf("timed out waiting for channel close; received %d events", len(events))
		}
	}
}

func TestQueryStream_PreStreamErrors(t *testing.T) {
	type embeddingMock struct {
		response *openai.CreateEmbeddingResponse
		err      error
	}
	type searchMock struct {
		result []types.ScoredChunk
		err    error
	}
	type mock struct {
		embedding embeddingMock
		search    searchMock
	}
	type expected struct {
		err string
	}

	tests := []struct {
		name     string
		mock     mock
		expected expected
	}{
		{
			name: "returns error when embedding generation fails",
			mock: mock{
				embedding: embeddingMock{err: errors.New("openai down")},
			},
			expected: expected{err: "failed to generate embedding for query"},
		},
		{
			name: "returns error when vector store search fails",
			mock: mock{
				embedding: embeddingMock{response: makeEmbeddingResponse([][]float64{{0.1}})},
				search:    searchMock{err: errors.New("search broken")},
			},
			expected: expected{err: "failed to search vector store"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ec := &mockEmbeddingCreator{
				newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
					return tt.mock.embedding.response, tt.mock.embedding.err
				},
			}
			vs := &vectorstore.MockVectorStore{
				SearchFunc: func(_ []float64, _ int) ([]types.ScoredChunk, error) {
					return tt.mock.search.result, tt.mock.search.err
				},
			}
			pipeline := newTestPipeline(ec, &mockChatCompleter{}, vs)

			events, err := pipeline.QueryStream(context.Background(), "q", nil)

			assert.Error(t, err)
			assert.Contains(t, err.Error(), tt.expected.err)
			assert.Nil(t, events)
		})
	}
}

func TestQueryStream_Success(t *testing.T) {
	type expected struct {
		tokens  string
		sources int
	}
	type mock struct {
		searchResults []types.ScoredChunk
		streamChunks  []openai.ChatCompletionChunk
	}

	tests := []struct {
		name     string
		mock     mock
		expected expected
	}{
		{
			name: "concatenates three chunks in order",
			mock: mock{
				searchResults: []types.ScoredChunk{
					{Chunk: types.DocumentChunk{ID: "c1", Content: "Go is compiled"}, Score: 0.9},
				},
				streamChunks: []openai.ChatCompletionChunk{
					makeChatCompletionChunk("Hello"),
					makeChatCompletionChunk(" "),
					makeChatCompletionChunk("world"),
				},
			},
			expected: expected{tokens: "Hello world", sources: 1},
		},
		{
			name: "skips empty-content chunks",
			mock: mock{
				searchResults: []types.ScoredChunk{
					{Chunk: types.DocumentChunk{ID: "c1", Content: "ctx"}, Score: 0.5},
				},
				streamChunks: []openai.ChatCompletionChunk{
					makeChatCompletionChunk(""),
					makeChatCompletionChunk("answer"),
					{Choices: []openai.ChatCompletionChunkChoice{}},
				},
			},
			expected: expected{tokens: "answer", sources: 1},
		},
		{
			name: "empty stream emits sources then done with no tokens",
			mock: mock{
				searchResults: []types.ScoredChunk{
					{Chunk: types.DocumentChunk{ID: "c1", Content: "ctx"}, Score: 0.5},
				},
				streamChunks: []openai.ChatCompletionChunk{},
			},
			expected: expected{tokens: "", sources: 1},
		},
		{
			name: "handles no search results",
			mock: mock{
				searchResults: []types.ScoredChunk{},
				streamChunks: []openai.ChatCompletionChunk{
					makeChatCompletionChunk("no info"),
				},
			},
			expected: expected{tokens: "no info", sources: 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ec := &mockEmbeddingCreator{
				newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
					return makeEmbeddingResponse([][]float64{{0.1}}), nil
				},
			}
			stream := &mockChatStream{chunks: tt.mock.streamChunks}
			cc := &mockChatCompleter{
				newStreamingFunc: func(_ context.Context, _ openai.ChatCompletionNewParams, _ ...option.RequestOption) ChatStream {
					return stream
				},
			}
			vs := &vectorstore.MockVectorStore{
				SearchFunc: func(_ []float64, _ int) ([]types.ScoredChunk, error) {
					return tt.mock.searchResults, nil
				},
			}
			pipeline := newTestPipeline(ec, cc, vs)

			events, err := pipeline.QueryStream(context.Background(), "q", nil)
			assert.NoError(t, err)
			assert.NotNil(t, events)

			received := drainEvents(t, events)

			assert.GreaterOrEqual(t, len(received), 2, "expected at least sources + done")
			assert.NotNil(t, received[0].Sources)
			assert.Len(t, received[0].Sources, tt.expected.sources)
			assert.Equal(t, defaultConfidence, received[0].Confidence)

			var tokens string
			for _, ev := range received[1 : len(received)-1] {
				tokens += ev.Token
			}
			assert.Equal(t, tt.expected.tokens, tokens)

			last := received[len(received)-1]
			assert.True(t, last.Done)
			assert.NoError(t, last.Err)
			assert.True(t, stream.closed, "stream should be closed")
		})
	}
}

func TestQueryStream_HistoryThreaded(t *testing.T) {
	history := []types.Message{
		{Role: types.RoleUser, Content: "u1"},
		{Role: types.RoleAssistant, Content: "a1"},
	}

	var capturedEmbed string
	ec := &mockEmbeddingCreator{
		newFunc: func(_ context.Context, body openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
			capturedEmbed = body.Input.OfString.Value
			return makeEmbeddingResponse([][]float64{{0.1}}), nil
		},
	}

	var capturedMsgs []openai.ChatCompletionMessageParamUnion
	stream := &mockChatStream{chunks: []openai.ChatCompletionChunk{makeChatCompletionChunk("ok")}}
	cc := &mockChatCompleter{
		newStreamingFunc: func(_ context.Context, body openai.ChatCompletionNewParams, _ ...option.RequestOption) ChatStream {
			capturedMsgs = body.Messages
			return stream
		},
	}
	vs := &vectorstore.MockVectorStore{
		SearchFunc: func(_ []float64, _ int) ([]types.ScoredChunk, error) {
			return []types.ScoredChunk{}, nil
		},
	}

	pipeline := newTestPipeline(ec, cc, vs)
	events, err := pipeline.QueryStream(context.Background(), "u2", history)
	assert.NoError(t, err)
	drainEvents(t, events)

	assert.Equal(t, "u1 u2", capturedEmbed)
	assert.Len(t, capturedMsgs, 4)
	assert.NotNil(t, capturedMsgs[0].OfSystem)
	assert.Equal(t, "u1", capturedMsgs[1].OfUser.Content.OfString.Value)
	assert.Equal(t, "a1", capturedMsgs[2].OfAssistant.Content.OfString.Value)
	assert.Equal(t, "u2", capturedMsgs[3].OfUser.Content.OfString.Value)
}

func TestQueryStream_UpstreamError(t *testing.T) {
	ec := &mockEmbeddingCreator{
		newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
			return makeEmbeddingResponse([][]float64{{0.1}}), nil
		},
	}
	stream := &mockChatStream{
		chunks: []openai.ChatCompletionChunk{makeChatCompletionChunk("partial")},
		err:    errors.New("deepseek disconnected"),
	}
	cc := &mockChatCompleter{
		newStreamingFunc: func(_ context.Context, _ openai.ChatCompletionNewParams, _ ...option.RequestOption) ChatStream {
			return stream
		},
	}
	vs := &vectorstore.MockVectorStore{
		SearchFunc: func(_ []float64, _ int) ([]types.ScoredChunk, error) {
			return []types.ScoredChunk{{Chunk: types.DocumentChunk{Content: "ctx"}}}, nil
		},
	}
	pipeline := newTestPipeline(ec, cc, vs)

	events, err := pipeline.QueryStream(context.Background(), "q", nil)
	assert.NoError(t, err)

	received := drainEvents(t, events)

	last := received[len(received)-1]
	assert.Error(t, last.Err)
	assert.Contains(t, last.Err.Error(), "deepseek disconnected")
	assert.False(t, last.Done)
	assert.True(t, stream.closed)
}

func TestQueryStream_ContextCancellation(t *testing.T) {
	ec := &mockEmbeddingCreator{
		newFunc: func(_ context.Context, _ openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
			return makeEmbeddingResponse([][]float64{{0.1}}), nil
		},
	}

	chunks := make([]openai.ChatCompletionChunk, 100)
	for i := range chunks {
		chunks[i] = makeChatCompletionChunk("x")
	}
	stream := &mockChatStream{chunks: chunks}
	cc := &mockChatCompleter{
		newStreamingFunc: func(_ context.Context, _ openai.ChatCompletionNewParams, _ ...option.RequestOption) ChatStream {
			return stream
		},
	}
	vs := &vectorstore.MockVectorStore{
		SearchFunc: func(_ []float64, _ int) ([]types.ScoredChunk, error) {
			return []types.ScoredChunk{{Chunk: types.DocumentChunk{Content: "ctx"}}}, nil
		},
	}
	pipeline := newTestPipeline(ec, cc, vs)

	ctx, cancel := context.WithCancel(context.Background())
	events, err := pipeline.QueryStream(ctx, "q", nil)
	assert.NoError(t, err)

	// Consume the initial sources event, then cancel and stop reading.
	sources, ok := <-events
	assert.True(t, ok)
	assert.NotNil(t, sources.Sources)
	cancel()

	// Drain any remaining events until the channel closes. The goroutine
	// should detect the cancellation and close promptly.
	closed := make(chan struct{})
	go func() {
		for range events {
		}
		close(closed)
	}()

	select {
	case <-closed:
	case <-time.After(1 * time.Second):
		t.Fatal("channel did not close after ctx cancellation")
	}
}
