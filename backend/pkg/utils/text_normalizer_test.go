package utils

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"rag-backend/pkg/types"
)

func TestNormalize(t *testing.T) {
	const cleanProse = "The Go Programming Language\n\nGo was designed at Google in 2007. It reached version 1.0 in March 2012.\n\nConcurrency is built around goroutines and channels."

	tests := []struct {
		name     string
		text     string
		expected string
	}{
		{
			name:     "joins word hyphenated across a line break",
			text:     "exam-\nple",
			expected: "example",
		},
		{
			name:     "joins hyphenated word ignoring stray spaces",
			text:     "exam- \n  ple",
			expected: "example",
		},
		{
			name:     "joins a word hyphenated across a page break",
			text:     "compre-\n\nhensive",
			expected: "comprehensive",
		},
		{
			name:     "preserves a real mid-line hyphen",
			text:     "well-known",
			expected: "well-known",
		},
		{
			name:     "does not join digits split across a line break",
			text:     "page 5-\n3",
			expected: "page 5- 3",
		},
		{
			name:     "collapses runs of horizontal whitespace",
			text:     "a    b\tc",
			expected: "a b c",
		},
		{
			name:     "unwraps a single newline into a space",
			text:     "line one\nline two",
			expected: "line one line two",
		},
		{
			name:     "preserves paragraph boundary",
			text:     "Para one.\n\nPara two.",
			expected: "Para one.\n\nPara two.",
		},
		{
			name:     "collapses three or more newlines to a paragraph break",
			text:     "A\n\n\n\nB",
			expected: "A\n\nB",
		},
		{
			name:     "collapses spaces surrounding a paragraph break",
			text:     "A  \n  \n  B",
			expected: "A\n\nB",
		},
		{
			name:     "normalizes windows line endings",
			text:     "a\r\nb",
			expected: "a b",
		},
		{
			name:     "trims leading and trailing whitespace",
			text:     "  hi  ",
			expected: "hi",
		},
		{
			name:     "returns empty string for empty input",
			text:     "",
			expected: "",
		},
		{
			name:     "leaves already clean prose unchanged",
			text:     cleanProse,
			expected: cleanProse,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, Normalize(tt.text))
		})
	}
}

func TestStripRepeatedHeadersFooters(t *testing.T) {
	tests := []struct {
		name     string
		pages    []string
		expected []string
	}{
		{
			name: "strips a repeated header line",
			pages: []string{
				"ACME Confidential\nAlpha body content",
				"ACME Confidential\nBeta body content",
				"ACME Confidential\nGamma body content",
			},
			expected: []string{"Alpha body content", "Beta body content", "Gamma body content"},
		},
		{
			name: "strips a repeated footer with varying page numbers",
			pages: []string{
				"Alpha body content\nPage 1",
				"Beta body content\nPage 2",
				"Gamma body content\nPage 3",
			},
			expected: []string{"Alpha body content", "Beta body content", "Gamma body content"},
		},
		{
			name: "strips bare page-number footers",
			pages: []string{
				"Alpha body content\n12",
				"Beta body content\n13",
				"Gamma body content\n14",
			},
			expected: []string{"Alpha body content", "Beta body content", "Gamma body content"},
		},
		{
			name: "preserves non-repeating first lines",
			pages: []string{
				"Unique alpha title\nAlpha body content",
				"Unique beta title\nBeta body content",
				"Unique gamma title\nGamma body content",
			},
			expected: []string{
				"Unique alpha title\nAlpha body content",
				"Unique beta title\nBeta body content",
				"Unique gamma title\nGamma body content",
			},
		},
		{
			name: "keeps identical lines when below minimum page count",
			pages: []string{
				"Repeated header\nAlpha body content",
				"Repeated header\nBeta body content",
			},
			expected: []string{
				"Repeated header\nAlpha body content",
				"Repeated header\nBeta body content",
			},
		},
		{
			name: "keeps a line repeated below the threshold",
			pages: []string{
				"Shared header\nAlpha body content",
				"Distinct beta head\nBeta body content",
				"Distinct gamma head\nGamma body content",
				"Distinct delta head\nDelta body content",
			},
			expected: []string{
				"Shared header\nAlpha body content",
				"Distinct beta head\nBeta body content",
				"Distinct gamma head\nGamma body content",
				"Distinct delta head\nDelta body content",
			},
		},
		{
			name:     "handles empty pages without panicking",
			pages:    []string{"", "", ""},
			expected: []string{"", "", ""},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := make([]types.Page, len(tt.pages))
			for i, text := range tt.pages {
				input[i] = types.Page{Number: i + 1, Text: text}
			}
			want := make([]types.Page, len(tt.expected))
			for i, text := range tt.expected {
				want[i] = types.Page{Number: i + 1, Text: text}
			}
			assert.Equal(t, want, StripRepeatedHeadersFooters(input))
		})
	}
}
