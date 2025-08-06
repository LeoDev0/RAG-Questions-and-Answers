package handlers

import (
	"net/http"
	"rag-backend/pkg/codes"

	"github.com/gin-gonic/gin"

	"rag-backend/internal/services"
	"rag-backend/pkg/types"
)

type QueryHandler struct {
	ragPipeline *services.RAGPipeline
}

func NewQueryHandler(ragPipeline *services.RAGPipeline) *QueryHandler {
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
