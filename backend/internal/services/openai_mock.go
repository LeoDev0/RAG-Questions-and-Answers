package services

import (
	"context"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

type mockEmbeddingCreator struct {
	newFunc func(ctx context.Context, body openai.EmbeddingNewParams, opts ...option.RequestOption) (*openai.CreateEmbeddingResponse, error)
}

func (m *mockEmbeddingCreator) New(ctx context.Context, body openai.EmbeddingNewParams, opts ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
	return m.newFunc(ctx, body, opts...)
}

type mockChatCompleter struct {
	newFunc          func(ctx context.Context, body openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error)
	newStreamingFunc func(ctx context.Context, body openai.ChatCompletionNewParams, opts ...option.RequestOption) ChatStream
}

func (m *mockChatCompleter) New(ctx context.Context, body openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error) {
	return m.newFunc(ctx, body, opts...)
}

func (m *mockChatCompleter) NewStreamingIter(ctx context.Context, body openai.ChatCompletionNewParams, opts ...option.RequestOption) ChatStream {
	return m.newStreamingFunc(ctx, body, opts...)
}

type mockChatStream struct {
	chunks   []openai.ChatCompletionChunk
	index    int
	err      error
	closed   bool
	closeErr error
}

func (m *mockChatStream) Next() bool {
	if m.err != nil {
		return false
	}
	if m.index >= len(m.chunks) {
		return false
	}
	m.index++
	return true
}

func (m *mockChatStream) Current() openai.ChatCompletionChunk {
	if m.index == 0 || m.index > len(m.chunks) {
		return openai.ChatCompletionChunk{}
	}
	return m.chunks[m.index-1]
}

func (m *mockChatStream) Err() error {
	return m.err
}

func (m *mockChatStream) Close() error {
	m.closed = true
	return m.closeErr
}
