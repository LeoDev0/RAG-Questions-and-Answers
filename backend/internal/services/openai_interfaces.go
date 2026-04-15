package services

import (
	"context"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// EmbeddingCreator abstracts the OpenAI embedding API call for testability.
type EmbeddingCreator interface {
	New(ctx context.Context, body openai.EmbeddingNewParams, opts ...option.RequestOption) (*openai.CreateEmbeddingResponse, error)
}

// ChatCompletionCreator abstracts the OpenAI chat completion API call for testability.
type ChatCompletionCreator interface {
	New(ctx context.Context, body openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error)
}
