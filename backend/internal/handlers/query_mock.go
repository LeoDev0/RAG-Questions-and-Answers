package handlers

import (
	"context"

	"rag-backend/internal/services"
	"rag-backend/pkg/types"
)

type mockQueryService struct {
	queryFunc       func(question string, history []types.Message) (*types.RAGResponse, error)
	queryStreamFunc func(ctx context.Context, question string, history []types.Message) (<-chan services.StreamEvent, error)
}

func (m *mockQueryService) Query(question string, history []types.Message) (*types.RAGResponse, error) {
	return m.queryFunc(question, history)
}

func (m *mockQueryService) QueryStream(ctx context.Context, question string, history []types.Message) (<-chan services.StreamEvent, error) {
	return m.queryStreamFunc(ctx, question, history)
}
