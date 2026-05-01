package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"

	"rag-backend/pkg/codes"
	"rag-backend/pkg/types"
)

func newQueryRequest(body string) *http.Request {
	req := httptest.NewRequest(http.MethodPost, "/api/query", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	return req
}

func TestHandleQuery(t *testing.T) {
	gin.SetMode(gin.TestMode)

	type mock struct {
		response *types.RAGResponse
		err      error
	}
	type expected struct {
		status          int
		code            string
		detailSubstr    string
		answer          string
		sources         []types.DocumentChunk
		confidence      float64
		capturedHistory []types.Message
	}

	canned := &types.RAGResponse{
		Answer: "forty-two",
		Sources: []types.DocumentChunk{
			{ID: "c1", Content: "ctx"},
		},
		Confidence: 0.8,
	}

	tests := []struct {
		name     string
		body     string
		mock     mock
		expected expected
	}{
		{
			name:     "rejects malformed JSON",
			body:     "{not-json",
			expected: expected{status: http.StatusBadRequest, code: codes.ErrInvalidRequest},
		},
		{
			name:     "rejects empty question",
			body:     `{"question":""}`,
			expected: expected{status: http.StatusBadRequest, code: codes.ErrInvalidRequest},
		},
		{
			name:     "rejects history entry with invalid role",
			body:     `{"question":"hi","history":[{"role":"system","content":"x"}]}`,
			expected: expected{status: http.StatusBadRequest, code: codes.ErrInvalidRequest},
		},
		{
			name:     "rejects history entry with empty content",
			body:     `{"question":"hi","history":[{"role":"user","content":""}]}`,
			expected: expected{status: http.StatusBadRequest, code: codes.ErrInvalidRequest},
		},
		{
			name: "returns 500 when pipeline fails",
			body: `{"question":"hi"}`,
			mock: mock{err: errors.New("vector store down")},
			expected: expected{
				status:       http.StatusInternalServerError,
				code:         codes.ErrQueryError,
				detailSubstr: "vector store down",
			},
		},
		{
			name: "returns 200 with answer sources and confidence",
			body: `{"question":"hi"}`,
			mock: mock{response: canned},
			expected: expected{
				status:     http.StatusOK,
				answer:     canned.Answer,
				sources:    canned.Sources,
				confidence: canned.Confidence,
			},
		},
		{
			name: "forwards history to pipeline",
			body: `{"question":"q2","history":[{"role":"user","content":"q1"},{"role":"assistant","content":"a1"}]}`,
			mock: mock{response: canned},
			expected: expected{
				status:     http.StatusOK,
				answer:     canned.Answer,
				sources:    canned.Sources,
				confidence: canned.Confidence,
				capturedHistory: []types.Message{
					{Role: types.RoleUser, Content: "q1"},
					{Role: types.RoleAssistant, Content: "a1"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var capturedHistory []types.Message
			h := NewQueryHandler(&mockQueryService{
				queryFunc: func(_ string, history []types.Message) (*types.RAGResponse, error) {
					capturedHistory = history
					return tt.mock.response, tt.mock.err
				},
			})

			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = newQueryRequest(tt.body)

			h.HandleQuery(c)

			assert.Equal(t, tt.expected.status, w.Code)

			if tt.expected.status == http.StatusOK {
				var resp types.QueryResponse
				err := json.Unmarshal(w.Body.Bytes(), &resp)
				assert.NoError(t, err)
				assert.Equal(t, tt.expected.answer, resp.Answer)
				assert.Equal(t, tt.expected.sources, resp.Sources)
				assert.Equal(t, tt.expected.confidence, resp.Confidence)
				if tt.expected.capturedHistory != nil {
					assert.Equal(t, tt.expected.capturedHistory, capturedHistory)
				}
				return
			}

			var resp types.ErrorResponse
			err := json.Unmarshal(w.Body.Bytes(), &resp)
			assert.NoError(t, err)
			assert.Equal(t, tt.expected.code, resp.Code)
			if tt.expected.detailSubstr != "" {
				assert.Contains(t, resp.Details, tt.expected.detailSubstr)
			}
		})
	}
}
