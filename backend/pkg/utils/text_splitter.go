package utils

import (
	"strings"
	"unicode/utf8"
)

type TextSplitter struct {
	ChunkSize    int
	ChunkOverlap int
}

func NewTextSplitter(chunkSize, chunkOverlap int) *TextSplitter {
	return &TextSplitter{
		ChunkSize:    chunkSize,
		ChunkOverlap: chunkOverlap,
	}
}

// SplitText splits the input text into chunks based on the configured ChunkSize and ChunkOverlap.
func (ts *TextSplitter) SplitText(text string) []string {
	if utf8.RuneCountInString(text) <= ts.ChunkSize {
		return []string{text}
	}

	var chunks []string
	runes := []rune(text)
	start := 0

	for start < len(runes) {
		end := min(start+ts.ChunkSize, len(runes))

		chunk := string(runes[start:end])
		chunks = append(chunks, strings.TrimSpace(chunk))

		if end >= len(runes) {
			break
		}

		start = max(end-ts.ChunkOverlap, 0)
	}

	return chunks
}
