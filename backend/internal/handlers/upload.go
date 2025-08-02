package handlers

import (
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"

	"rag-backend/internal/services"
	"rag-backend/pkg/types"
)

const maxFileSize = 10 << 20 // 10mb

type UploadHandler struct {
	ragPipeline       *services.RAGPipeline
	documentProcessor *services.DocumentProcessor
}

func NewUploadHandler(ragPipeline *services.RAGPipeline, documentProcessor *services.DocumentProcessor) *UploadHandler {
	return &UploadHandler{
		ragPipeline:       ragPipeline,
		documentProcessor: documentProcessor,
	}
}

func (h *UploadHandler) HandleUpload(c *gin.Context) {
	fileHeader, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, types.UploadResponse{
			Success: false,
			Error:   "No file provided",
		})
		return
	}

	if fileHeader.Size > maxFileSize {
		c.JSON(http.StatusBadRequest, types.UploadResponse{
			Success: false,
			Error:   fmt.Sprintf("File too large. Maximum size is %s", userFriendlyFileSizeFormatter(maxFileSize)),
		})
		return
	}

	content, err := h.documentProcessor.ProcessFile(fileHeader)
	if err != nil {
		c.JSON(http.StatusInternalServerError, types.UploadResponse{
			Success: false,
			Error:   "Failed to process document: " + err.Error(),
		})
		return
	}

	document := h.documentProcessor.CreateDocument(content, fileHeader.Filename)

	// Process into chunks with embeddings
	metadata := map[string]string{
		"source": fileHeader.Filename,
	}
	chunks, err := h.ragPipeline.ProcessDocument(content, metadata)
	if err != nil {
		c.JSON(http.StatusInternalServerError, types.UploadResponse{
			Success: false,
			Error:   "Failed to process document chunks: " + err.Error(),
		})
		return
	}

	h.ragPipeline.AddDocumentToVectorStore(chunks)

	document.Chunks = chunks

	c.JSON(http.StatusOK, types.UploadResponse{
		Success: true,
		Document: &types.UploadDocumentSummary{
			ID:          document.ID,
			Name:        document.Name,
			ChunksCount: len(document.Chunks),
			UploadedAt:  document.UploadedAt,
		},
	})
}

func userFriendlyFileSizeFormatter(bytes int64) string {
	const mb = 1 << 20 // 1MB in bytes
	if bytes%mb == 0 {
		return fmt.Sprintf("%dMB", bytes/mb)
	}
	// For non-exact MB values, show with decimal
	return fmt.Sprintf("%.1fMB", float64(bytes)/float64(mb))
}
