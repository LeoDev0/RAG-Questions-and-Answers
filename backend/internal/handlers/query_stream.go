package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"

	"rag-backend/internal/services"
	"rag-backend/pkg/codes"
	"rag-backend/pkg/types"
)

const (
	sseEventSources = "sources"
	sseEventToken   = "token"
	sseEventDone    = "done"
	sseEventError   = "error"
)

type sseEvent struct {
	Type       string                `json:"type"`
	Sources    []types.DocumentChunk `json:"sources,omitempty"`
	Confidence float64               `json:"confidence,omitempty"`
	Content    string                `json:"content,omitempty"`
	Error      string                `json:"error,omitempty"`
	Code       string                `json:"code,omitempty"`
}

func (h *QueryHandler) HandleQueryStream(c *gin.Context) {
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

	events, err := h.ragPipeline.QueryStream(c.Request.Context(), request.Question)
	if err != nil {
		c.JSON(http.StatusInternalServerError, types.ErrorResponse{
			Error:   "Failed to start stream",
			Code:    codes.ErrStreamError,
			Details: err.Error(),
		})
		return
	}

	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.Header().Set("X-Accel-Buffering", "no")
	c.Status(http.StatusOK)

	c.Stream(func(w io.Writer) bool {
		select {
		case <-c.Request.Context().Done():
			return false
		case ev, ok := <-events:
			if !ok {
				return false
			}
			return writeStreamEvent(w, ev)
		}
	})
}

func writeStreamEvent(w io.Writer, ev services.StreamEvent) bool {
	switch {
	case ev.Err != nil:
		writeSSEFrame(w, sseEvent{
			Type:  sseEventError,
			Error: ev.Err.Error(),
			Code:  codes.ErrStreamError,
		})
		return false
	case ev.Done:
		writeSSEFrame(w, sseEvent{Type: sseEventDone})
		return false
	case ev.Sources != nil:
		writeSSEFrame(w, sseEvent{
			Type:       sseEventSources,
			Sources:    ev.Sources,
			Confidence: ev.Confidence,
		})
		return true
	case ev.Token != "":
		writeSSEFrame(w, sseEvent{
			Type:    sseEventToken,
			Content: ev.Token,
		})
		return true
	default:
		return true
	}
}

func writeSSEFrame(w io.Writer, payload sseEvent) {
	data, err := json.Marshal(payload)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "data: %s\n\n", data)
}
