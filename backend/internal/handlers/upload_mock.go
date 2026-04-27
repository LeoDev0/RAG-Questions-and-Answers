package handlers

import (
	"mime/multipart"

	"rag-backend/pkg/types"
)

type mockDocumentIngester struct {
	processDocumentFunc          func(content string, metadata map[string]string) ([]types.DocumentChunk, error)
	addDocumentToVectorStoreFunc func(chunks []types.DocumentChunk) error
}

func (m *mockDocumentIngester) ProcessDocument(content string, metadata map[string]string) ([]types.DocumentChunk, error) {
	return m.processDocumentFunc(content, metadata)
}

func (m *mockDocumentIngester) AddDocumentToVectorStore(chunks []types.DocumentChunk) error {
	return m.addDocumentToVectorStoreFunc(chunks)
}

type mockFileProcessor struct {
	processFileFunc    func(fileHeader *multipart.FileHeader) (string, error)
	createDocumentFunc func(content, fileName string) types.Document
}

func (m *mockFileProcessor) ProcessFile(fileHeader *multipart.FileHeader) (string, error) {
	return m.processFileFunc(fileHeader)
}

func (m *mockFileProcessor) CreateDocument(content, fileName string) types.Document {
	return m.createDocumentFunc(content, fileName)
}
