package utils

import (
	"regexp"
	"strings"
)

const (
	minPagesForDetection = 3
	scanLines            = 2
	minRepeats           = 3
	repeatFraction       = 0.5
)

const paragraphSentinel = "\x00"

var (
	reLineEndings  = regexp.MustCompile(`\r\n?`)
	reHyphenBreak  = regexp.MustCompile(`(\p{L})-[ \t]*\n[ \t]*(\p{L})`)
	reHorizontalWS = regexp.MustCompile(`[ \t]+`)
	reParagraph    = regexp.MustCompile(`[ \t]*\n[ \t]*\n[ \t\n]*`)
	reSingleNL     = regexp.MustCompile(`[ \t]*\n[ \t]*`)
	reDigits       = regexp.MustCompile(`\d+`)
)

// Normalize cleans extracted text before chunking: it joins words hyphenated
// across line breaks, collapses irregular whitespace, and unwraps single
// newlines into spaces while preserving blank-line paragraph boundaries the
// text splitter relies on.
func Normalize(text string) string {
	text = reLineEndings.ReplaceAllString(text, "\n")
	text = reHyphenBreak.ReplaceAllString(text, "$1$2")
	text = reHorizontalWS.ReplaceAllString(text, " ")
	text = reParagraph.ReplaceAllString(text, paragraphSentinel)
	text = reSingleNL.ReplaceAllString(text, " ")
	text = strings.ReplaceAll(text, paragraphSentinel, "\n\n")
	text = reHorizontalWS.ReplaceAllString(text, " ")
	return strings.TrimSpace(text)
}

// StripRepeatedHeadersFooters removes page headers and footers that repeat
// across the per-page text of a PDF, then joins the surviving pages with blank
// lines so page breaks become paragraph boundaries. Detection is skipped for
// documents with too few pages to provide a reliable signal.
func StripRepeatedHeadersFooters(pages []string) string {
	if len(pages) < minPagesForDetection {
		return strings.Join(pages, "\n\n")
	}

	pageLines := make([][]string, len(pages))
	headerCounts := map[string]int{}
	footerCounts := map[string]int{}

	for i, page := range pages {
		lines := splitLines(page)
		pageLines[i] = lines

		for _, line := range topLines(lines, scanLines) {
			if sig := lineSignature(line); sig != "" {
				headerCounts[sig]++
			}
		}
		for _, line := range bottomLines(lines, scanLines) {
			if sig := lineSignature(line); sig != "" {
				footerCounts[sig]++
			}
		}
	}

	threshold := int(repeatFraction * float64(len(pages)))
	if threshold < minRepeats {
		threshold = minRepeats
	}

	headers := frequentSignatures(headerCounts, threshold)
	footers := frequentSignatures(footerCounts, threshold)

	cleaned := make([]string, 0, len(pages))
	for _, lines := range pageLines {
		lines = trimLeadingMatches(lines, headers)
		lines = trimTrailingMatches(lines, footers)
		cleaned = append(cleaned, strings.Join(lines, "\n"))
	}

	return strings.Join(cleaned, "\n\n")
}

func splitLines(page string) []string {
	normalized := reLineEndings.ReplaceAllString(page, "\n")
	return strings.Split(normalized, "\n")
}

func topLines(lines []string, n int) []string {
	if len(lines) < n {
		return lines
	}
	return lines[:n]
}

func bottomLines(lines []string, n int) []string {
	if len(lines) < n {
		return lines
	}
	return lines[len(lines)-n:]
}

func lineSignature(line string) string {
	line = reDigits.ReplaceAllString(line, "")
	return strings.Join(strings.Fields(strings.ToLower(line)), " ")
}

func frequentSignatures(counts map[string]int, threshold int) map[string]bool {
	frequent := map[string]bool{}
	for sig, count := range counts {
		if count >= threshold {
			frequent[sig] = true
		}
	}
	return frequent
}

func trimLeadingMatches(lines []string, sigs map[string]bool) []string {
	limit := scanLines
	for len(lines) > 0 && limit > 0 && sigs[lineSignature(lines[0])] {
		lines = lines[1:]
		limit--
	}
	return lines
}

func trimTrailingMatches(lines []string, sigs map[string]bool) []string {
	limit := scanLines
	for len(lines) > 0 && limit > 0 && sigs[lineSignature(lines[len(lines)-1])] {
		lines = lines[:len(lines)-1]
		limit--
	}
	return lines
}
