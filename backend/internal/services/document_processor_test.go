package services

import (
	"bytes"
	"io"
	"mime/multipart"
	"net/textproto"
	"strings"
	"testing"
	"time"

	"rag-backend/pkg/types"
)

func TestNewDocumentProcessor(t *testing.T) {
	dp := NewDocumentProcessor()
	if dp == nil {
		t.Fatal("NewDocumentProcessor() returned nil")
	}
}

func TestCreateDocument(t *testing.T) {
	dp := NewDocumentProcessor()
	content := "Test content"
	fileName := "test.txt"

	doc := dp.CreateDocument(content, fileName)

	if doc.ID == "" {
		t.Error("Document ID should not be empty")
	}
	if doc.Name != fileName {
		t.Errorf("Expected name %s, got %s", fileName, doc.Name)
	}
	if doc.Content != content {
		t.Errorf("Expected content %s, got %s", content, doc.Content)
	}
	if doc.Chunks == nil {
		t.Error("Chunks should be initialized as empty slice, not nil")
	}
	if len(doc.Chunks) != 0 {
		t.Errorf("Expected empty chunks, got %d chunks", len(doc.Chunks))
	}
	if doc.UploadedAt.IsZero() {
		t.Error("UploadedAt should be set")
	}
	// Verify the upload time is recent (within last second)
	if time.Since(doc.UploadedAt) > time.Second {
		t.Error("UploadedAt should be close to current time")
	}
}

func TestCreateDocument_UniqueIDs(t *testing.T) {
	dp := NewDocumentProcessor()

	doc1 := dp.CreateDocument("content1", "file1.txt")
	doc2 := dp.CreateDocument("content2", "file2.txt")

	if doc1.ID == doc2.ID {
		t.Error("Document IDs should be unique")
	}
}

func TestProcessFile_TextPlain(t *testing.T) {
	dp := NewDocumentProcessor()

	tests := []struct {
		name        string
		content     string
		expected    string
		description string
	}{
		{
			name:        "simple text",
			content:     "Hello, World!",
			expected:    "Hello, World!",
			description: "Process simple plain text",
		},
		{
			name:        "multiline text",
			content:     "Line 1\nLine 2\nLine 3",
			expected:    "Line 1\nLine 2\nLine 3",
			description: "Process multiline text",
		},
		{
			name:        "empty text",
			content:     "",
			expected:    "",
			description: "Process empty text file",
		},
		{
			name:        "text with special characters",
			content:     "Special chars: @#$%^&*()\n\tTabbed content",
			expected:    "Special chars: @#$%^&*()\n\tTabbed content",
			description: "Process text with special characters",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fileHeader := createMultipartFileHeader(tt.content, "test.txt", "text/plain")

			result, err := dp.ProcessFile(fileHeader)
			if err != nil {
				t.Fatalf("ProcessFile() error = %v", err)
			}
			if result != tt.expected {
				t.Errorf("ProcessFile() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestProcessFile_UnsupportedType(t *testing.T) {
	dp := NewDocumentProcessor()

	unsupportedTypes := []string{
		"application/json",
		"image/png",
		"application/xml",
		"video/mp4",
		"",
	}

	for _, contentType := range unsupportedTypes {
		t.Run(contentType, func(t *testing.T) {
			fileHeader := createMultipartFileHeader("content", "test.file", contentType)

			_, err := dp.ProcessFile(fileHeader)
			if err == nil {
				t.Error("ProcessFile() should return error for unsupported type")
			}
			if !strings.Contains(err.Error(), "unsupported file type") {
				t.Errorf("Expected 'unsupported file type' error, got: %v", err)
			}
		})
	}
}

func TestProcessFile_InvalidFileHeader(t *testing.T) {
	dp := NewDocumentProcessor()

	// Create a file header with a broken reader
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	part, _ := writer.CreateFormFile("file", "test.txt")
	part.Write([]byte("test content"))
	writer.Close()

	// Parse the multipart form
	reader := multipart.NewReader(&buf, writer.Boundary())
	form, _ := reader.ReadForm(1024)

	// Get the file header
	fileHeaders := form.File["file"]
	if len(fileHeaders) == 0 {
		t.Fatal("Failed to create test file header")
	}
	fileHeader := fileHeaders[0]

	// Close the underlying temp files to simulate an error
	form.RemoveAll()

	// Now trying to open should fail
	_, err := dp.ProcessFile(fileHeader)
	if err == nil {
		t.Error("ProcessFile() should return error when file cannot be opened")
	}
	// The error could be either "failed to open file" or "unsupported file type"
	// depending on how the multipart.FileHeader behaves after RemoveAll
	if !strings.Contains(err.Error(), "failed to open file") &&
	   !strings.Contains(err.Error(), "unsupported file type") {
		t.Errorf("Expected 'failed to open file' or 'unsupported file type' error, got: %v", err)
	}
}

func TestProcessPDF_EmptyPDF(t *testing.T) {
	dp := NewDocumentProcessor()

	// Create an invalid PDF that will fail to parse
	invalidPDF := []byte("This is not a valid PDF")

	_, err := dp.processPDF(invalidPDF)
	if err == nil {
		t.Error("processPDF() should return error for invalid PDF")
	}
	if !strings.Contains(err.Error(), "failed to create PDF reader") {
		t.Errorf("Expected 'failed to create PDF reader' error, got: %v", err)
	}
}

// Helper function to create a multipart.FileHeader for testing
func createMultipartFileHeader(content, filename, contentType string) *multipart.FileHeader {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// Create the form file with the specified content type
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", `form-data; name="file"; filename="`+filename+`"`)
	h.Set("Content-Type", contentType)

	part, _ := writer.CreatePart(h)
	io.Copy(part, strings.NewReader(content))
	writer.Close()

	// Parse the multipart form to get the FileHeader
	reader := multipart.NewReader(body, writer.Boundary())
	form, _ := reader.ReadForm(1024 * 1024)
	fileHeaders := form.File["file"]

	if len(fileHeaders) == 0 {
		return nil
	}

	return fileHeaders[0]
}

// Benchmark tests
func BenchmarkCreateDocument(b *testing.B) {
	dp := NewDocumentProcessor()
	content := "Test content for benchmarking"
	fileName := "test.txt"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dp.CreateDocument(content, fileName)
	}
}

func BenchmarkProcessFile_TextPlain(b *testing.B) {
	dp := NewDocumentProcessor()
	content := strings.Repeat("Test content for benchmarking. ", 100)
	fileHeader := createMultipartFileHeader(content, "test.txt", "text/plain")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dp.ProcessFile(fileHeader)
	}
}

// Table-driven test for comprehensive edge cases
func TestProcessFile_EdgeCases(t *testing.T) {
	dp := NewDocumentProcessor()

	tests := []struct {
		name        string
		content     string
		contentType string
		filename    string
		wantErr     bool
		errContains string
	}{
		{
			name:        "large text file",
			content:     strings.Repeat("A", 1024*1024), // 1MB
			contentType: "text/plain",
			filename:    "large.txt",
			wantErr:     false,
		},
		{
			name:        "unicode text",
			content:     "Hello 世界! 🌍 مرحبا בעולם",
			contentType: "text/plain",
			filename:    "unicode.txt",
			wantErr:     false,
		},
		{
			name:        "text with null bytes",
			content:     "Text with\x00null bytes",
			contentType: "text/plain",
			filename:    "null.txt",
			wantErr:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fileHeader := createMultipartFileHeader(tt.content, tt.filename, tt.contentType)
			if fileHeader == nil {
				t.Fatal("Failed to create file header")
			}

			result, err := dp.ProcessFile(fileHeader)

			if (err != nil) != tt.wantErr {
				t.Errorf("ProcessFile() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err != nil && tt.errContains != "" {
				if !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("Expected error to contain %q, got %q", tt.errContains, err.Error())
				}
			}

			if !tt.wantErr && result != tt.content {
				t.Errorf("ProcessFile() content mismatch")
			}
		})
	}
}

// Test concurrent document creation to ensure thread safety
func TestCreateDocument_Concurrent(t *testing.T) {
	dp := NewDocumentProcessor()
	const numGoroutines = 100

	docs := make([]types.Document, numGoroutines)
	done := make(chan bool)

	for i := 0; i < numGoroutines; i++ {
		go func(index int) {
			docs[index] = dp.CreateDocument("content", "file.txt")
			done <- true
		}(i)
	}

	// Wait for all goroutines to complete
	for i := 0; i < numGoroutines; i++ {
		<-done
	}

	// Check that all IDs are unique
	idMap := make(map[string]bool)
	for _, doc := range docs {
		if idMap[doc.ID] {
			t.Errorf("Duplicate document ID found: %s", doc.ID)
		}
		idMap[doc.ID] = true
	}
}
