package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"

	"rag-backend/internal/services"
	"rag-backend/pkg/codes"
	"rag-backend/pkg/types"
)

type fakeQueryService struct {
	queryFunc       func(question string) (*types.RAGResponse, error)
	queryStreamFunc func(ctx context.Context, question string) (<-chan services.StreamEvent, error)
}

func (f *fakeQueryService) Query(question string) (*types.RAGResponse, error) {
	return f.queryFunc(question)
}

func (f *fakeQueryService) QueryStream(ctx context.Context, question string) (<-chan services.StreamEvent, error) {
	return f.queryStreamFunc(ctx, question)
}

func parseSSEFrames(body string) []map[string]any {
	var frames []map[string]any
	for _, block := range strings.Split(body, "\n\n") {
		block = strings.TrimSpace(block)
		if !strings.HasPrefix(block, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(block, "data:"))
		var m map[string]any
		if err := json.Unmarshal([]byte(payload), &m); err == nil {
			frames = append(frames, m)
		}
	}
	return frames
}

func newStreamRequest(body string) *http.Request {
	req := httptest.NewRequest(http.MethodPost, "/api/query/stream", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	return req
}

func TestHandleQueryStream_ValidationErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)

	type expected struct {
		status int
		code   string
	}

	tests := []struct {
		name     string
		body     string
		expected expected
	}{
		{
			name:     "rejects malformed JSON",
			body:     "{not-json",
			expected: expected{status: http.StatusBadRequest, code: codes.ErrInvalidRequest},
		},
		{
			name:     "rejects empty question",
			body:     `{"question": ""}`,
			expected: expected{status: http.StatusBadRequest, code: codes.ErrInvalidRequest},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := NewQueryHandler(&fakeQueryService{})

			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = newStreamRequest(tt.body)

			h.HandleQueryStream(c)

			assert.Equal(t, tt.expected.status, w.Code)

			var resp types.ErrorResponse
			err := json.Unmarshal(w.Body.Bytes(), &resp)
			assert.NoError(t, err)
			assert.Equal(t, tt.expected.code, resp.Code)
		})
	}
}

func TestHandleQueryStream_PreStreamError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	h := NewQueryHandler(&fakeQueryService{
		queryStreamFunc: func(_ context.Context, _ string) (<-chan services.StreamEvent, error) {
			return nil, errors.New("vector store exploded")
		},
	})

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = newStreamRequest(`{"question":"hi"}`)

	h.HandleQueryStream(c)

	assert.Equal(t, http.StatusInternalServerError, w.Code)

	var resp types.ErrorResponse
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	assert.NoError(t, err)
	assert.Equal(t, codes.ErrStreamError, resp.Code)
	assert.Contains(t, resp.Details, "vector store exploded")
}

func TestHandleQueryStream_SuccessPath(t *testing.T) {
	gin.SetMode(gin.TestMode)

	type expected struct {
		frameTypes []string
		tokens     string
	}

	tests := []struct {
		name     string
		send     []services.StreamEvent
		expected expected
	}{
		{
			name: "sources, three tokens, done",
			send: []services.StreamEvent{
				{Sources: []types.DocumentChunk{{ID: "c1", Content: "ctx"}}, Confidence: 0.8},
				{Token: "Hello"},
				{Token: " "},
				{Token: "world"},
				{Done: true},
			},
			expected: expected{
				frameTypes: []string{"sources", "token", "token", "token", "done"},
				tokens:     "Hello world",
			},
		},
		{
			name: "mid-stream error produces error frame",
			send: []services.StreamEvent{
				{Sources: []types.DocumentChunk{}, Confidence: 0.8},
				{Token: "partial"},
				{Err: fmt.Errorf("deepseek disconnected")},
			},
			expected: expected{
				frameTypes: []string{"sources", "token", "error"},
				tokens:     "partial",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ch := make(chan services.StreamEvent, len(tt.send))
			for _, ev := range tt.send {
				ch <- ev
			}
			close(ch)

			h := NewQueryHandler(&fakeQueryService{
				queryStreamFunc: func(_ context.Context, _ string) (<-chan services.StreamEvent, error) {
					return ch, nil
				},
			})

			router := gin.New()
			router.POST("/api/query/stream", h.HandleQueryStream)
			server := httptest.NewServer(router)
			defer server.Close()

			req, err := http.NewRequest(http.MethodPost, server.URL+"/api/query/stream", strings.NewReader(`{"question":"hi"}`))
			assert.NoError(t, err)
			req.Header.Set("Content-Type", "application/json")

			client := &http.Client{Timeout: 2 * time.Second}
			resp, err := client.Do(req)
			assert.NoError(t, err)
			defer resp.Body.Close()

			assert.Equal(t, http.StatusOK, resp.StatusCode)
			assert.Equal(t, "text/event-stream", resp.Header.Get("Content-Type"))
			assert.Equal(t, "no-cache", resp.Header.Get("Cache-Control"))
			assert.Equal(t, "no", resp.Header.Get("X-Accel-Buffering"))

			bodyBytes, err := io.ReadAll(resp.Body)
			assert.NoError(t, err)

			frames := parseSSEFrames(string(bodyBytes))

			frameTypes := make([]string, len(frames))
			var tokens string
			for i, f := range frames {
				frameTypes[i] = f["type"].(string)
				if f["type"] == "token" {
					tokens += f["content"].(string)
				}
			}

			assert.Equal(t, tt.expected.frameTypes, frameTypes)
			assert.Equal(t, tt.expected.tokens, tokens)
		})
	}
}
