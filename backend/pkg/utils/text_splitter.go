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

// separators descends from paragraphs to lines to sentence boundaries to words to
// characters. The list is ASCII-only: CJK enders such as 。？！ are not included, so
// CJK text falls through to the character-level fallback. The empty string always
// matches and guarantees termination.
var separators = []string{"\n\n", "\n", ". ", "? ", "! ", " ", ""}

// SplitText splits the input text into chunks that respect the separator hierarchy,
// only breaking at a finer level when a piece still exceeds ChunkSize. Every emitted
// chunk has at most ChunkSize runes.
func (ts *TextSplitter) SplitText(text string) []string {
	if ts.ChunkSize < 1 || utf8.RuneCountInString(text) <= ts.ChunkSize {
		return []string{text}
	}

	overlap := min(ts.ChunkOverlap, ts.ChunkSize-1)
	return ts.splitText(text, separators, overlap)
}

func (ts *TextSplitter) splitText(text string, seps []string, overlap int) []string {
	sep, rest := pickSeparator(text, seps)
	pieces := splitKeepSeparator(text, sep)

	var chunks []string
	var goodSplits []string

	for _, piece := range pieces {
		if utf8.RuneCountInString(piece) <= ts.ChunkSize {
			goodSplits = append(goodSplits, piece)
			continue
		}

		chunks = append(chunks, ts.mergeSplits(goodSplits, overlap)...)
		goodSplits = nil
		chunks = append(chunks, ts.splitText(piece, rest, overlap)...)
	}

	chunks = append(chunks, ts.mergeSplits(goodSplits, overlap)...)
	return chunks
}

func pickSeparator(text string, seps []string) (string, []string) {
	for i, sep := range seps {
		if sep == "" || strings.Contains(text, sep) {
			return sep, seps[i+1:]
		}
	}
	return "", nil
}

func splitKeepSeparator(text, sep string) []string {
	if sep == "" {
		runes := []rune(text)
		pieces := make([]string, 0, len(runes))
		for _, r := range runes {
			pieces = append(pieces, string(r))
		}
		return pieces
	}

	parts := strings.Split(text, sep)
	pieces := make([]string, 0, len(parts))
	for i, part := range parts {
		if i < len(parts)-1 {
			part += sep
		}
		if part != "" {
			pieces = append(pieces, part)
		}
	}
	return pieces
}

func (ts *TextSplitter) mergeSplits(splits []string, overlap int) []string {
	var chunks []string
	var buffer []string
	total := 0

	for _, split := range splits {
		length := utf8.RuneCountInString(split)
		if total+length > ts.ChunkSize && total > 0 {
			if chunk := strings.TrimSpace(strings.Join(buffer, "")); chunk != "" {
				chunks = append(chunks, chunk)
			}
			for len(buffer) > 0 && (total > overlap || (total+length > ts.ChunkSize && total > 0)) {
				total -= utf8.RuneCountInString(buffer[0])
				buffer = buffer[1:]
			}
		}
		buffer = append(buffer, split)
		total += length
	}

	if chunk := strings.TrimSpace(strings.Join(buffer, "")); chunk != "" {
		chunks = append(chunks, chunk)
	}
	return chunks
}
