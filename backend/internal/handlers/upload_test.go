package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"

	"rag-backend/pkg/codes"
	"rag-backend/pkg/types"
)

func newUploadRequest(t *testing.T, filename, contentType string, content []byte) *http.Request {
	t.Helper()
	body := new(bytes.Buffer)
	writer := multipart.NewWriter(body)

	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename="%s"`, filename))
	h.Set("Content-Type", contentType)

	part, err := writer.CreatePart(h)
	if err != nil {
		t.Fatalf("newUploadRequest: create part: %v", err)
	}
	if _, err := part.Write(content); err != nil {
		t.Fatalf("newUploadRequest: write content: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("newUploadRequest: close writer: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	return req
}

func newUploadRequestWithoutFile(t *testing.T) *http.Request {
	t.Helper()
	body := new(bytes.Buffer)
	writer := multipart.NewWriter(body)
	if err := writer.WriteField("other", "value"); err != nil {
		t.Fatalf("newUploadRequestWithoutFile: write field: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("newUploadRequestWithoutFile: close writer: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/upload", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	return req
}

func newUploadRequestOversized(t *testing.T, filename string, size int64) *http.Request {
	t.Helper()
	req := newUploadRequest(t, filename, "text/plain", []byte("x"))
	if err := req.ParseMultipartForm(10 << 20); err != nil {
		t.Fatalf("newUploadRequestOversized: parse form: %v", err)
	}
	req.MultipartForm.File["file"][0].Size = size
	return req
}

func TestHandleUpload(t *testing.T) {
	gin.SetMode(gin.TestMode)

	type mock struct {
		processFileContent string
		processFileErr     error
		createDocument     types.Document
		processDocChunks   []types.DocumentChunk
		processDocErr      error
		addToStoreErr      error
	}
	type calls struct {
		processFile     int
		createDocument  int
		processDocument int
		addToStore      int
	}
	type expected struct {
		status        int
		code          string
		detailSubstr  string
		errorContains string
		document      *types.UploadDocumentSummary
		calls         calls
	}

	fixedID := "doc-123"
	fixedTime := time.Date(2026, 4, 23, 10, 0, 0, 0, time.UTC)
	fixedChunks := []types.DocumentChunk{
		{ID: "sample.txt-chunk-0", Content: "c0"},
		{ID: "sample.txt-chunk-1", Content: "c1"},
		{ID: "sample.txt-chunk-2", Content: "c2"},
	}
	stubDoc := types.Document{
		ID:         fixedID,
		Name:       "sample.txt",
		UploadedAt: fixedTime,
		Chunks:     []types.DocumentChunk{},
	}

	tests := []struct {
		name         string
		buildRequest func(t *testing.T) *http.Request
		mock         mock
		expected     expected
	}{
		{
			name: "returns 400 when no file is provided",
			buildRequest: func(t *testing.T) *http.Request {
				return newUploadRequestWithoutFile(t)
			},
			expected: expected{
				status: http.StatusBadRequest,
				code:   codes.ErrNoFile,
				calls:  calls{},
			},
		},
		{
			name: "returns 400 when file exceeds max size",
			buildRequest: func(t *testing.T) *http.Request {
				return newUploadRequestOversized(t, "big.txt", int64(maxFileSize)+1)
			},
			expected: expected{
				status:        http.StatusBadRequest,
				code:          codes.ErrFileTooLarge,
				errorContains: "10MB",
				calls:         calls{},
			},
		},
		{
			name: "returns 500 when document processor fails",
			buildRequest: func(t *testing.T) *http.Request {
				return newUploadRequest(t, "hello.txt", "text/plain", []byte("x"))
			},
			mock: mock{processFileErr: errors.New("read boom")},
			expected: expected{
				status:       http.StatusInternalServerError,
				code:         codes.ErrProcessing,
				detailSubstr: "read boom",
				calls:        calls{processFile: 1},
			},
		},
		{
			name: "returns 500 when pipeline chunking fails",
			buildRequest: func(t *testing.T) *http.Request {
				return newUploadRequest(t, "sample.txt", "text/plain", []byte("content"))
			},
			mock: mock{
				processFileContent: "parsed content",
				createDocument:     stubDoc,
				processDocErr:      errors.New("embed boom"),
			},
			expected: expected{
				status:       http.StatusInternalServerError,
				code:         codes.ErrChunking,
				detailSubstr: "embed boom",
				calls:        calls{processFile: 1, createDocument: 1, processDocument: 1},
			},
		},
		{
			name: "returns 500 when vector store storage fails",
			buildRequest: func(t *testing.T) *http.Request {
				return newUploadRequest(t, "sample.txt", "text/plain", []byte("content"))
			},
			mock: mock{
				processFileContent: "parsed content",
				createDocument:     stubDoc,
				processDocChunks:   fixedChunks[:2],
				addToStoreErr:      errors.New("store boom"),
			},
			expected: expected{
				status:       http.StatusInternalServerError,
				code:         codes.ErrStorage,
				detailSubstr: "store boom",
				calls:        calls{processFile: 1, createDocument: 1, processDocument: 1, addToStore: 1},
			},
		},
		{
			name: "returns 200 with upload summary on success",
			buildRequest: func(t *testing.T) *http.Request {
				return newUploadRequest(t, "sample.txt", "text/plain", []byte("full content"))
			},
			mock: mock{
				processFileContent: "parsed content",
				createDocument:     stubDoc,
				processDocChunks:   fixedChunks,
			},
			expected: expected{
				status: http.StatusOK,
				document: &types.UploadDocumentSummary{
					ID:          fixedID,
					Name:        "sample.txt",
					ChunksCount: 3,
					UploadedAt:  fixedTime,
				},
				calls: calls{processFile: 1, createDocument: 1, processDocument: 1, addToStore: 1},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got calls
			var capturedContent string
			var capturedMetadata map[string]string
			var capturedChunks []types.DocumentChunk
			var createDocumentCalledWith struct {
				content  string
				fileName string
			}

			ingester := &mockDocumentIngester{
				processDocumentFunc: func(content string, metadata map[string]string) ([]types.DocumentChunk, error) {
					got.processDocument++
					capturedContent = content
					capturedMetadata = metadata
					return tt.mock.processDocChunks, tt.mock.processDocErr
				},
				addDocumentToVectorStoreFunc: func(chunks []types.DocumentChunk) error {
					got.addToStore++
					capturedChunks = chunks
					return tt.mock.addToStoreErr
				},
			}
			processor := &mockFileProcessor{
				processFileFunc: func(*multipart.FileHeader) (string, error) {
					got.processFile++
					return tt.mock.processFileContent, tt.mock.processFileErr
				},
				createDocumentFunc: func(content, fileName string) types.Document {
					got.createDocument++
					createDocumentCalledWith.content = content
					createDocumentCalledWith.fileName = fileName
					return tt.mock.createDocument
				},
			}
			h := NewUploadHandler(ingester, processor)

			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = tt.buildRequest(t)

			h.HandleUpload(c)

			assert.Equal(t, tt.expected.status, w.Code)
			assert.Equal(t, tt.expected.calls, got)

			if tt.expected.status == http.StatusOK {
				var resp types.UploadResponse
				err := json.Unmarshal(w.Body.Bytes(), &resp)
				assert.NoError(t, err)
				assert.NotNil(t, resp.Document)
				assert.Equal(t, tt.expected.document.ID, resp.Document.ID)
				assert.Equal(t, tt.expected.document.Name, resp.Document.Name)
				assert.Equal(t, tt.expected.document.ChunksCount, resp.Document.ChunksCount)
				assert.True(t, tt.expected.document.UploadedAt.Equal(resp.Document.UploadedAt))

				assert.Equal(t, "parsed content", capturedContent)
				assert.Equal(t, map[string]string{"source": "sample.txt"}, capturedMetadata)
				assert.Equal(t, fixedChunks, capturedChunks)
				assert.Equal(t, "parsed content", createDocumentCalledWith.content)
				assert.Equal(t, "sample.txt", createDocumentCalledWith.fileName)
				return
			}

			var resp types.ErrorResponse
			err := json.Unmarshal(w.Body.Bytes(), &resp)
			assert.NoError(t, err)
			assert.Equal(t, tt.expected.code, resp.Code)
			if tt.expected.detailSubstr != "" {
				assert.Contains(t, resp.Details, tt.expected.detailSubstr)
			}
			if tt.expected.errorContains != "" {
				assert.Contains(t, resp.Error, tt.expected.errorContains)
			}
		})
	}
}

func TestUserFriendlyFileSizeFormatter(t *testing.T) {
	tests := []struct {
		name     string
		bytes    int64
		expected string
	}{
		{name: "formats exact 10MB", bytes: 10 << 20, expected: "10MB"},
		{name: "formats exact 1MB", bytes: 1 << 20, expected: "1MB"},
		{name: "formats zero as exact MB", bytes: 0, expected: "0MB"},
		{name: "formats 1.5MB with decimal", bytes: (1 << 20) + (1 << 19), expected: "1.5MB"},
		{name: "formats non-exact as decimal with trailing zero", bytes: (10 << 20) + 1, expected: "10.0MB"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, userFriendlyFileSizeFormatter(tt.bytes))
		})
	}
}
