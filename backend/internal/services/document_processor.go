package services

import (
	"bytes"
	"fmt"
	"io"
	"mime/multipart"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/ledongthuc/pdf"

	"rag-backend/pkg/types"
)

type DocumentProcessor struct{}

func NewDocumentProcessor() *DocumentProcessor {
	return &DocumentProcessor{}
}

func (dp *DocumentProcessor) ProcessFile(fileHeader *multipart.FileHeader) (string, error) {
	file, err := fileHeader.Open()
	if err != nil {
		return "", fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		return "", fmt.Errorf("failed to read file: %w", err)
	}

	contentType := fileHeader.Header.Get("Content-Type")

	switch contentType {
	case "application/pdf":
		return dp.processPDF(content)
	case "text/plain":
		return string(content), nil
	default:
		return "", fmt.Errorf("unsupported file type: %s", contentType)
	}
}

func (dp *DocumentProcessor) processPDF(content []byte) (string, error) {
	reader := bytes.NewReader(content)

	pdfReader, err := pdf.NewReader(reader, int64(len(content)))
	if err != nil {
		return "", fmt.Errorf("failed to create PDF reader: %w", err)
	}

	var textBuilder strings.Builder
	numPages := pdfReader.NumPage()

	for i := 1; i <= numPages; i++ {
		page := pdfReader.Page(i)
		if page.V.IsNull() {
			continue
		}

		text, err := page.GetPlainText(nil)
		if err != nil {
			continue // Skip pages that can't be processed
		}

		textBuilder.WriteString(text)
		textBuilder.WriteString("\n")
	}

	text := textBuilder.String()
	if text == "" {
		return "", fmt.Errorf("no text could be extracted from PDF")
	}

	return strings.TrimSpace(text), nil
}

func (dp *DocumentProcessor) CreateDocument(content, fileName string) types.Document {
	return types.Document{
		ID:         uuid.New().String(),
		Name:       fileName,
		Content:    content,
		Chunks:     []types.DocumentChunk{},
		UploadedAt: time.Now(),
	}
}
