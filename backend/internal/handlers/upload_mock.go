package handlers

import (
	"mime/multipart"

	"rag-backend/pkg/types"
)

type mockDocumentIngester struct {
	processDocumentFunc          func(doc types.ProcessedDocument, metadata map[string]string) ([]types.DocumentChunk, error)
	addDocumentToVectorStoreFunc func(chunks []types.DocumentChunk) error
}

func (m *mockDocumentIngester) ProcessDocument(doc types.ProcessedDocument, metadata map[string]string) ([]types.DocumentChunk, error) {
	return m.processDocumentFunc(doc, metadata)
}

func (m *mockDocumentIngester) AddDocumentToVectorStore(chunks []types.DocumentChunk) error {
	return m.addDocumentToVectorStoreFunc(chunks)
}

type mockFileProcessor struct {
	processFileFunc    func(fileHeader *multipart.FileHeader) (types.ProcessedDocument, error)
	createDocumentFunc func(content, fileName string) types.Document
}

func (m *mockFileProcessor) ProcessFile(fileHeader *multipart.FileHeader) (types.ProcessedDocument, error) {
	return m.processFileFunc(fileHeader)
}

func (m *mockFileProcessor) CreateDocument(content, fileName string) types.Document {
	return m.createDocumentFunc(content, fileName)
}
