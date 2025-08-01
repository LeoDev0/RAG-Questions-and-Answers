package handlers

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"rag-backend/pkg/types"
)

type HealthHandler struct{}

func NewHealthHandler() *HealthHandler {
	return &HealthHandler{}
}

func (h *HealthHandler) HandleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, types.HealthResponse{
		Status:    "OK",
		Timestamp: time.Now(),
	})
}
