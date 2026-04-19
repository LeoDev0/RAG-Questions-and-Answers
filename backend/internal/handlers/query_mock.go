package handlers

import (
	"context"

	"rag-backend/internal/services"
	"rag-backend/pkg/types"
)

type mockQueryService struct {
	queryFunc       func(question string) (*types.RAGResponse, error)
	queryStreamFunc func(ctx context.Context, question string) (<-chan services.StreamEvent, error)
}

func (m *mockQueryService) Query(question string) (*types.RAGResponse, error) {
	return m.queryFunc(question)
}

func (m *mockQueryService) QueryStream(ctx context.Context, question string) (<-chan services.StreamEvent, error) {
	return m.queryStreamFunc(ctx, question)
}
