package services

import (
	"context"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/ssestream"
)

// EmbeddingCreator abstracts the OpenAI embedding API call for testability.
type EmbeddingCreator interface {
	New(ctx context.Context, body openai.EmbeddingNewParams, opts ...option.RequestOption) (*openai.CreateEmbeddingResponse, error)
}

// ChatStream iterates over streamed chat completion chunks. It mirrors the
// subset of *ssestream.Stream[openai.ChatCompletionChunk] that the pipeline
// consumes, allowing tests to stub streaming without the SDK's concrete type.
type ChatStream interface {
	Next() bool
	Current() openai.ChatCompletionChunk
	Err() error
	Close() error
}

// ChatCompletionCreator abstracts the OpenAI chat completion API call for testability.
type ChatCompletionCreator interface {
	New(ctx context.Context, body openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error)
	NewStreamingIter(ctx context.Context, body openai.ChatCompletionNewParams, opts ...option.RequestOption) ChatStream
}

// chatCompletionsAdapter wraps the SDK's chat completion service so it satisfies
// ChatCompletionCreator, including the streaming iterator method.
type chatCompletionsAdapter struct {
	inner *openai.ChatCompletionService
}

func (a *chatCompletionsAdapter) New(ctx context.Context, body openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error) {
	return a.inner.New(ctx, body, opts...)
}

func (a *chatCompletionsAdapter) NewStreamingIter(ctx context.Context, body openai.ChatCompletionNewParams, opts ...option.RequestOption) ChatStream {
	return &sdkChatStream{stream: a.inner.NewStreaming(ctx, body, opts...)}
}

type sdkChatStream struct {
	stream *ssestream.Stream[openai.ChatCompletionChunk]
}

func (s *sdkChatStream) Next() bool                          { return s.stream.Next() }
func (s *sdkChatStream) Current() openai.ChatCompletionChunk { return s.stream.Current() }
func (s *sdkChatStream) Err() error                          { return s.stream.Err() }
func (s *sdkChatStream) Close() error                        { return s.stream.Close() }
