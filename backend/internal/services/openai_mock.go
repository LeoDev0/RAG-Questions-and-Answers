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
	newFunc func(ctx context.Context, body openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error)
}

func (m *mockChatCompleter) New(ctx context.Context, body openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error) {
	return m.newFunc(ctx, body, opts...)
}
