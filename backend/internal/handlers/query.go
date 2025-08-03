package handlers

import (
	"net/http"

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
		c.JSON(http.StatusBadRequest, types.QueryResponse{
			Success: false,
			Error:   "Question is required and must be a string",
		})
		return
	}

	if request.Question == "" {
		c.JSON(http.StatusBadRequest, types.QueryResponse{
			Success: false,
			Error:   "Question cannot be empty",
		})
		return
	}

	response, err := h.ragPipeline.Query(request.Question)
	if err != nil {
		c.JSON(http.StatusInternalServerError, types.QueryResponse{
			Success: false,
			Error:   "Failed to process query: " + err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, types.QueryResponse{
		Success:    true,
		Answer:     response.Answer,
		Sources:    response.Sources,
		Confidence: response.Confidence,
	})
}
