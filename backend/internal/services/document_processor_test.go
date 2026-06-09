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

	"rag-backend/pkg/types"
)

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
	_ = writer.Close()

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
				assert.Empty(t, result.NormalizedText)
			} else if tt.expected.nonEmpty {
				assert.NoError(t, err)
				assert.NotEmpty(t, result.NormalizedText)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected.result, result.NormalizedText)
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
				assert.Empty(t, result.NormalizedText)
			} else {
				assert.NoError(t, err)
				assert.NotEmpty(t, result.NormalizedText)
			}
		})
	}
}

func TestBuildProcessedDocument(t *testing.T) {
	type expected struct {
		normalized string
		spans      []types.PageSpan
	}

	tests := []struct {
		name     string
		pages    []types.Page
		expected expected
	}{
		{
			name: "multi-page records contiguous spans with running offsets",
			pages: []types.Page{
				{Number: 1, Text: "First page content"},
				{Number: 2, Text: "Second page content"},
				{Number: 3, Text: "Third page content"},
			},
			expected: expected{
				normalized: "First page content\n\nSecond page content\n\nThird page content",
				spans: []types.PageSpan{
					{Page: 1, Start: 0, End: 18},
					{Page: 2, Start: 20, End: 39},
					{Page: 3, Start: 41, End: 59},
				},
			},
		},
		{
			name: "skips an empty middle page but keeps true page numbers",
			pages: []types.Page{
				{Number: 1, Text: "Alpha content here"},
				{Number: 2, Text: "   "},
				{Number: 3, Text: "Gamma content here"},
			},
			expected: expected{
				normalized: "Alpha content here\n\nGamma content here",
				spans: []types.PageSpan{
					{Page: 1, Start: 0, End: 18},
					{Page: 3, Start: 20, End: 38},
				},
			},
		},
		{
			name:  "single page produces a single span",
			pages: []types.Page{{Number: 1, Text: "Only page text"}},
			expected: expected{
				normalized: "Only page text",
				spans:      []types.PageSpan{{Page: 1, Start: 0, End: 14}},
			},
		},
		{
			name: "strips a repeated header before computing spans",
			pages: []types.Page{
				{Number: 1, Text: "ACME Confidential\nAlpha body"},
				{Number: 2, Text: "ACME Confidential\nBeta body"},
				{Number: 3, Text: "ACME Confidential\nGamma body"},
			},
			expected: expected{
				normalized: "Alpha body\n\nBeta body\n\nGamma body",
				spans: []types.PageSpan{
					{Page: 1, Start: 0, End: 10},
					{Page: 2, Start: 12, End: 21},
					{Page: 3, Start: 23, End: 33},
				},
			},
		},
		{
			name: "all-empty pages yield empty text and no spans",
			pages: []types.Page{
				{Number: 1, Text: ""},
				{Number: 2, Text: ""},
				{Number: 3, Text: ""},
			},
			expected: expected{
				normalized: "",
				spans:      []types.PageSpan{},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := buildProcessedDocument(tt.pages)

			assert.Equal(t, tt.expected.normalized, result.NormalizedText)
			assert.Equal(t, tt.expected.spans, result.PageSpans)

			for _, span := range result.PageSpans {
				assert.GreaterOrEqual(t, span.Start, 0)
				assert.LessOrEqual(t, span.End, len(result.NormalizedText))
				assert.NotEmpty(t, result.NormalizedText[span.Start:span.End],
					"span for page %d must locate non-empty text", span.Page)
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
