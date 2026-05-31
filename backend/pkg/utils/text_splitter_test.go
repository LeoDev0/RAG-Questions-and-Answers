package utils

import (
	"strings"
	"testing"
	"unicode/utf8"

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
		{
			name:     "splits on paragraph boundaries before finer separators",
			splitter: splitter{chunkSize: 6, chunkOverlap: 0},
			text:     "AAAA\n\nBBBB\n\nCCCC",
			expected: expected{chunks: []string{"AAAA", "BBBB", "CCCC"}},
		},
		{
			name:     "splits on sentence boundaries keeping the trailing period",
			splitter: splitter{chunkSize: 15, chunkOverlap: 0},
			text:     "Alpha is one. Beta is two. Gamma is three.",
			expected: expected{chunks: []string{"Alpha is one.", "Beta is two.", "Gamma is three."}},
		},
		{
			name:     "snaps overlap to a whole word instead of cutting mid word",
			splitter: splitter{chunkSize: 16, chunkOverlap: 6},
			text:     "alpha beta gamma delta",
			expected: expected{chunks: []string{"alpha beta", "beta gamma delta"}},
		},
		{
			name:     "splits on word boundaries with no overlap",
			splitter: splitter{chunkSize: 12, chunkOverlap: 0},
			text:     "alpha beta gamma delta",
			expected: expected{chunks: []string{"alpha beta", "gamma delta"}},
		},
		{
			name:     "drops empty pieces from leading trailing and duplicate separators",
			splitter: splitter{chunkSize: 3, chunkOverlap: 0},
			text:     "\n\n\nA\n\nB\n\n",
			expected: expected{chunks: []string{"A", "B"}},
		},
		{
			name:     "returns whole text when chunk size is not positive",
			splitter: splitter{chunkSize: 0, chunkOverlap: 0},
			text:     "abcdef",
			expected: expected{chunks: []string{"abcdef"}},
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

func TestSplitTextChunkSizeBound(t *testing.T) {
	type splitter struct {
		chunkSize    int
		chunkOverlap int
	}

	tests := []struct {
		name     string
		splitter splitter
		text     string
	}{
		{
			name:     "structured document with paragraphs and sentences",
			splitter: splitter{chunkSize: 1000, chunkOverlap: 200},
			text:     strings.Repeat("This is a sentence about chunking. It has structure.\n\n", 100),
		},
		{
			name:     "single long token without any separator",
			splitter: splitter{chunkSize: 1000, chunkOverlap: 200},
			text:     strings.Repeat("abcdefghij", 250),
		},
		{
			name:     "overlap larger than chunk size still terminates",
			splitter: splitter{chunkSize: 5, chunkOverlap: 100},
			text:     "abcdefghij",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := NewTextSplitter(tt.splitter.chunkSize, tt.splitter.chunkOverlap)

			result := ts.SplitText(tt.text)

			assert.NotEmpty(t, result)
			for _, chunk := range result {
				assert.LessOrEqual(t, utf8.RuneCountInString(chunk), tt.splitter.chunkSize)
				assert.NotEmpty(t, chunk)
			}
		})
	}
}

func TestSplitTextLongTokenReproducesSlidingWindow(t *testing.T) {
	ts := NewTextSplitter(1000, 200)
	text := strings.Repeat("abcdefghij", 250)

	result := ts.SplitText(text)

	assert.Len(t, result, 3)
	assert.Equal(t, 1000, utf8.RuneCountInString(result[0]))
	assert.Equal(t, 1000, utf8.RuneCountInString(result[1]))
	assert.Equal(t, 900, utf8.RuneCountInString(result[2]))

	for i := 0; i < len(result)-1; i++ {
		current := []rune(result[i])
		next := []rune(result[i+1])
		overlap := string(current[len(current)-200:])
		assert.Equal(t, overlap, string(next[:200]))
	}
}
