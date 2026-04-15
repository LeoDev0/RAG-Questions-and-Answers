package services

import (
	"bytes"
	"fmt"
	"mime/multipart"
	"net/textproto"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
)

// buildTestPDF produces a minimal but valid PDF byte slice.
// When contentStream is empty, the resulting PDF has zero pages, which causes
// processPDF to return "no text could be extracted from PDF".
// When contentStream is non-empty, it is embedded as the single page's content.
func buildTestPDF(contentStream string) []byte {
	var buf bytes.Buffer
	var offsets []int

	buf.WriteString("%PDF-1.4\n")

	writeObj := func(objNum int, body string) {
		offsets = append(offsets, buf.Len())
		fmt.Fprintf(&buf, "%d 0 obj\n%s\nendobj\n", objNum, body)
	}

	writeObj(1, "<< /Type /Catalog /Pages 2 0 R >>")

	if contentStream == "" {
		writeObj(2, "<< /Type /Pages /Kids [] /Count 0 >>")
	} else {
		writeObj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
		writeObj(3, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << >> >>")
		streamContent := contentStream + "\n"
		writeObj(4, fmt.Sprintf("<< /Length %d >>\nstream\n%sendstream", len(streamContent), streamContent))
	}

	xrefOffset := buf.Len()
	numObjs := len(offsets)
	fmt.Fprintf(&buf, "xref\n0 %d\n", numObjs+1)
	fmt.Fprintf(&buf, "%010d 65535 f \n", 0)
	for _, off := range offsets {
		fmt.Fprintf(&buf, "%010d 00000 n \n", off)
	}
	fmt.Fprintf(&buf, "trailer << /Size %d /Root 1 0 R >>\n", numObjs+1)
	fmt.Fprintf(&buf, "startxref\n%d\n%%%%EOF\n", xrefOffset)

	return buf.Bytes()
}

var (
	minimalPDFNoText   = buildTestPDF("")
	minimalPDFWithText = buildTestPDF("BT /F1 12 Tf 72 720 Td (Hello World) Tj ET")
)

// makeFileHeader builds a multipart.FileHeader by round-tripping through a real
// multipart writer and reader. This is the only way to obtain a valid
// FileHeader (with a working Open() method) outside of an HTTP request.
func makeFileHeader(t *testing.T, filename, contentType string, content []byte) *multipart.FileHeader {
	t.Helper()
	body := new(bytes.Buffer)
	writer := multipart.NewWriter(body)

	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename="%s"`, filename))
	h.Set("Content-Type", contentType)

	part, err := writer.CreatePart(h)
	if err != nil {
		t.Fatalf("makeFileHeader: create part: %v", err)
	}
	if _, err = part.Write(content); err != nil {
		t.Fatalf("makeFileHeader: write content: %v", err)
	}
	writer.Close()

	reader := multipart.NewReader(body, writer.Boundary())
	form, err := reader.ReadForm(10 << 20)
	if err != nil {
		t.Fatalf("makeFileHeader: read form: %v", err)
	}
	files, ok := form.File["file"]
	if !ok || len(files) == 0 {
		t.Fatalf("makeFileHeader: no file entry found in parsed form")
	}
	return files[0]
}

func TestNewDocumentProcessor(t *testing.T) {
	dp := NewDocumentProcessor()
	assert.NotNil(t, dp)
}

func TestProcessFile(t *testing.T) {
	type expected struct {
		result   string
		nonEmpty bool
		err      string
	}

	tests := []struct {
		name        string
		filename    string
		contentType string
		content     []byte
		expected    expected
	}{
		{
			name:        "returns content for text/plain file",
			filename:    "hello.txt",
			contentType: "text/plain",
			content:     []byte("hello world"),
			expected:    expected{result: "hello world"},
		},
		{
			name:        "preserves unicode content in text/plain",
			filename:    "unicode.txt",
			contentType: "text/plain",
			content:     []byte("日本語テスト"),
			expected:    expected{result: "日本語テスト"},
		},
		{
			name:        "returns error for unsupported content type",
			filename:    "image.png",
			contentType: "image/png",
			content:     []byte{0x89, 0x50, 0x4e, 0x47},
			expected:    expected{err: "unsupported file type: image/png"},
		},
		{
			name:        "returns error when PDF bytes are invalid",
			filename:    "bad.pdf",
			contentType: "application/pdf",
			content:     []byte("not a pdf"),
			expected:    expected{err: "failed to create PDF reader"},
		},
		{
			name:        "extracts text from valid PDF",
			filename:    "text.pdf",
			contentType: "application/pdf",
			content:     minimalPDFWithText,
			expected:    expected{nonEmpty: true},
		},
		{
			name:        "returns error when PDF has no extractable text",
			filename:    "empty.pdf",
			contentType: "application/pdf",
			content:     minimalPDFNoText,
			expected:    expected{err: "no text could be extracted from PDF"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dp := NewDocumentProcessor()
			fh := makeFileHeader(t, tt.filename, tt.contentType, tt.content)

			result, err := dp.ProcessFile(fh)

			if tt.expected.err != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expected.err)
				assert.Empty(t, result)
			} else if tt.expected.nonEmpty {
				assert.NoError(t, err)
				assert.NotEmpty(t, result)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected.result, result)
			}
		})
	}
}

func TestProcessPDF(t *testing.T) {
	type expected struct {
		nonEmpty bool
		err      string
	}

	tests := []struct {
		name     string
		content  []byte
		expected expected
	}{
		{
			name:     "extracts and trims text from valid PDF",
			content:  minimalPDFWithText,
			expected: expected{nonEmpty: true},
		},
		{
			name:     "returns error for non-PDF bytes",
			content:  []byte("garbage data xyz"),
			expected: expected{err: "failed to create PDF reader"},
		},
		{
			name:     "returns error for empty byte slice",
			content:  []byte{},
			expected: expected{err: "failed to create PDF reader"},
		},
		{
			name:     "returns error when PDF has no text content",
			content:  minimalPDFNoText,
			expected: expected{err: "no text could be extracted from PDF"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dp := NewDocumentProcessor()

			result, err := dp.processPDF(tt.content)

			if tt.expected.err != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expected.err)
				assert.Empty(t, result)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, result)
			}
		})
	}
}

func TestCreateDocument(t *testing.T) {
	type expected struct {
		name    string
		content string
	}

	tests := []struct {
		name     string
		content  string
		fileName string
		expected expected
	}{
		{
			name:     "sets name and content from parameters",
			content:  "document body",
			fileName: "test.txt",
			expected: expected{name: "test.txt", content: "document body"},
		},
		{
			name:     "handles empty content",
			content:  "",
			fileName: "empty.txt",
			expected: expected{name: "empty.txt", content: ""},
		},
		{
			name:     "handles empty file name",
			content:  "some content",
			fileName: "",
			expected: expected{name: "", content: "some content"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dp := NewDocumentProcessor()
			before := time.Now()

			doc := dp.CreateDocument(tt.content, tt.fileName)

			assert.Equal(t, tt.expected.name, doc.Name)
			assert.Equal(t, tt.expected.content, doc.Content)
			assert.NotEmpty(t, doc.ID)
			_, parseErr := uuid.Parse(doc.ID)
			assert.NoError(t, parseErr, "ID should be a valid UUID")
			assert.NotNil(t, doc.Chunks)
			assert.Len(t, doc.Chunks, 0)
			assert.WithinDuration(t, before, doc.UploadedAt, 5*time.Second)
		})
	}
}
