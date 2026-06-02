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
	"rag-backend/pkg/utils"
)

type DocumentProcessor struct{}

func NewDocumentProcessor() *DocumentProcessor {
	return &DocumentProcessor{}
}

func (dp *DocumentProcessor) ProcessFile(fileHeader *multipart.FileHeader) (types.ProcessedDocument, error) {
	file, err := fileHeader.Open()
	if err != nil {
		return types.ProcessedDocument{}, fmt.Errorf("failed to open file: %w", err)
	}
	defer func() { _ = file.Close() }()

	content, err := io.ReadAll(file)
	if err != nil {
		return types.ProcessedDocument{}, fmt.Errorf("failed to read file: %w", err)
	}

	contentType := fileHeader.Header.Get("Content-Type")

	switch contentType {
	case "application/pdf":
		return dp.processPDF(content)
	case "text/plain":
		return types.ProcessedDocument{NormalizedText: utils.Normalize(string(content))}, nil
	default:
		return types.ProcessedDocument{}, fmt.Errorf("unsupported file type: %s", contentType)
	}
}

func (dp *DocumentProcessor) processPDF(content []byte) (types.ProcessedDocument, error) {
	reader := bytes.NewReader(content)

	pdfReader, err := pdf.NewReader(reader, int64(len(content)))
	if err != nil {
		return types.ProcessedDocument{}, fmt.Errorf("failed to create PDF reader: %w", err)
	}

	numPages := pdfReader.NumPage()
	pages := make([]types.Page, 0, numPages)

	for i := 1; i <= numPages; i++ {
		page := pdfReader.Page(i)
		if page.V.IsNull() {
			continue
		}

		text, err := page.GetPlainText(nil)
		if err != nil {
			continue // Skip pages that can't be processed
		}

		pages = append(pages, types.Page{Number: i, Text: text})
	}

	processed := buildProcessedDocument(pages)
	if processed.NormalizedText == "" {
		return types.ProcessedDocument{}, fmt.Errorf("no text could be extracted from PDF")
	}

	return processed, nil
}

// buildProcessedDocument strips repeated headers/footers across the page slice,
// normalizes each surviving page independently, and concatenates them with
// blank-line page breaks. Normalizing per page keeps the recorded PageSpans
// exact in the final normalized coordinate space: each span's [Start:End]
// slice of NormalizedText is precisely that page's normalized text.
func buildProcessedDocument(pages []types.Page) types.ProcessedDocument {
	texts := make([]string, 0, len(pages))
	for _, page := range pages {
		texts = append(texts, page.Text)
	}
	cleaned := utils.StripRepeatedHeadersFooters(texts)

	spans := make([]types.PageSpan, 0, len(cleaned))
	var b strings.Builder
	for i, page := range cleaned {
		normalized := utils.Normalize(page)
		if normalized == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteString("\n\n")
		}
		start := b.Len()
		b.WriteString(normalized)
		spans = append(spans, types.PageSpan{Page: pages[i].Number, Start: start, End: b.Len()})
	}

	return types.ProcessedDocument{NormalizedText: b.String(), PageSpans: spans}
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
