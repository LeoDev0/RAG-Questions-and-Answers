package utils

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewTextSplitter(t *testing.T) {
	ts := NewTextSplitter(1000, 200)

	assert.NotNil(t, ts)
	assert.Equal(t, 1000, ts.ChunkSize)
	assert.Equal(t, 200, ts.ChunkOverlap)
}

func TestSplitText(t *testing.T) {
	const japaneseText = "日本語テスト"

	type splitter struct {
		chunkSize    int
		chunkOverlap int
	}

	type expected struct {
		chunks []string
	}

	tests := []struct {
		name     string
		splitter splitter
		text     string
		expected expected
	}{
		{
			name:     "returns single chunk when text shorter than chunk size",
			splitter: splitter{chunkSize: 100, chunkOverlap: 10},
			text:     "hello world",
			expected: expected{chunks: []string{"hello world"}},
		},
		{
			name:     "returns single chunk when text length equals chunk size",
			splitter: splitter{chunkSize: 5, chunkOverlap: 1},
			text:     "hello",
			expected: expected{chunks: []string{"hello"}},
		},
		{
			name:     "returns single chunk for empty text",
			splitter: splitter{chunkSize: 100, chunkOverlap: 10},
			text:     "",
			expected: expected{chunks: []string{""}},
		},
		{
			name:     "preserves whitespace when text fits in single chunk",
			splitter: splitter{chunkSize: 100, chunkOverlap: 10},
			text:     "  hello  ",
			expected: expected{chunks: []string{"  hello  "}},
		},
		{
			name:     "splits text into multiple chunks without overlap",
			splitter: splitter{chunkSize: 5, chunkOverlap: 0},
			text:     "abcdefghij",
			expected: expected{chunks: []string{"abcde", "fghij"}},
		},
		{
			name:     "splits text into multiple chunks with overlap",
			splitter: splitter{chunkSize: 5, chunkOverlap: 2},
			text:     "abcdefghij",
			expected: expected{chunks: []string{"abcde", "defgh", "ghij"}},
		},
		{
			name:     "trims whitespace from chunks when splitting",
			splitter: splitter{chunkSize: 5, chunkOverlap: 0},
			text:     "abc  fghij",
			expected: expected{chunks: []string{"abc", "fghij"}},
		},
		{
			name:     "respects unicode runes when splitting multibyte text",
			splitter: splitter{chunkSize: 3, chunkOverlap: 0},
			text:     japaneseText,
			expected: expected{chunks: []string{"日本語", "テスト"}},
		},
		{
			name:     "produces final chunk smaller than chunk size",
			splitter: splitter{chunkSize: 4, chunkOverlap: 1},
			text:     "abcdefgh",
			expected: expected{chunks: []string{"abcd", "defg", "gh"}},
		},
		{
			name:     "counts unicode by rune count not byte length for early return",
			splitter: splitter{chunkSize: 6, chunkOverlap: 0},
			text:     japaneseText,
			expected: expected{chunks: []string{japaneseText}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := NewTextSplitter(tt.splitter.chunkSize, tt.splitter.chunkOverlap)

			result := ts.SplitText(tt.text)

			assert.Equal(t, tt.expected.chunks, result)
		})
	}
}
