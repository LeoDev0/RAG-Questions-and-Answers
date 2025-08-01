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

func (ts *TextSplitter) SplitText(text string) []string {
	if utf8.RuneCountInString(text) <= ts.ChunkSize {
		return []string{text}
	}

	var chunks []string
	runes := []rune(text)
	start := 0

	for start < len(runes) {
		end := start + ts.ChunkSize
		if end > len(runes) {
			end = len(runes)
		}

		chunk := string(runes[start:end])
		chunks = append(chunks, strings.TrimSpace(chunk))

		if end >= len(runes) {
			break
		}

		start = end - ts.ChunkOverlap
		if start < 0 {
			start = 0
		}
	}

	return chunks
}
