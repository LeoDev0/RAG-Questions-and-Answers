package handlers

import (
	"context"
	"net/http"
	"rag-backend/pkg/codes"

	"github.com/gin-gonic/gin"

	"rag-backend/internal/services"
	"rag-backend/pkg/types"
)

type QueryService interface {
	Query(question string) (*types.RAGResponse, error)
	QueryStream(ctx context.Context, question string) (<-chan services.StreamEvent, error)
}

type QueryHandler struct {
	ragPipeline QueryService
}

func NewQueryHandler(ragPipeline QueryService) *QueryHandler {
	return &QueryHandler{
		ragPipeline: ragPipeline,
	}
}

func (h *QueryHandler) HandleQuery(c *gin.Context) {
	var request types.QueryRequest

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, types.ErrorResponse{
			Error: "Question is required and must be a string",
			Code:  codes.ErrInvalidRequest,
		})
		return
	}

	if request.Question == "" {
		c.JSON(http.StatusBadRequest, types.ErrorResponse{
			Error: "Question cannot be empty",
			Code:  codes.ErrEmptyQuestion,
		})
		return
	}

	response, err := h.ragPipeline.Query(request.Question)
	if err != nil {
		c.JSON(http.StatusInternalServerError, types.ErrorResponse{
			Error:   "Failed to process query",
			Code:    codes.ErrQueryError,
			Details: err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, types.QueryResponse{
		Answer:     response.Answer,
		Sources:    response.Sources,
		Confidence: response.Confidence,
	})
}
