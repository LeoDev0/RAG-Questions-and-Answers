package services

import (
	"context"
	"encoding/json"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/stretchr/testify/assert"

	"rag-backend/internal/config"
	"rag-backend/internal/repositories/vectorstore"
	"rag-backend/internal/repositories/vectorstore/memory"
	"rag-backend/pkg/types"
	"rag-backend/pkg/utils"
)

const embeddingDims = 256

type hashingEmbedder struct {
	dims int
}

func (e *hashingEmbedder) New(_ context.Context, body openai.EmbeddingNewParams, _ ...option.RequestOption) (*openai.CreateEmbeddingResponse, error) {
	var texts []string
	if body.Input.OfString.Valid() {
		texts = []string{body.Input.OfString.Value}
	} else {
		texts = body.Input.OfArrayOfStrings
	}

	data := make([]openai.Embedding, len(texts))
	for i, text := range texts {
		data[i] = openai.Embedding{Index: int64(i), Embedding: embedText(text, e.dims)}
	}
	return &openai.CreateEmbeddingResponse{Data: data}, nil
}

func embedText(text string, dims int) []float64 {
	vec := make([]float64, dims)
	for _, token := range tokenize(text) {
		h := fnv.New32a()
		_, _ = h.Write([]byte(token))
		vec[h.Sum32()%uint32(dims)]++
	}

	var norm float64
	for _, v := range vec {
		norm += v * v
	}
	if norm == 0 {
		return vec
	}
	norm = math.Sqrt(norm)
	for i := range vec {
		vec[i] /= norm
	}
	return vec
}

var stopwords = map[string]bool{
	"a": true, "an": true, "and": true, "are": true, "as": true, "at": true,
	"be": true, "but": true, "by": true, "do": true, "does": true, "for": true,
	"from": true, "how": true, "in": true, "into": true, "is": true, "it": true,
	"its": true, "of": true, "on": true, "or": true, "over": true, "that": true,
	"the": true, "their": true, "them": true, "there": true, "this": true,
	"to": true, "us": true, "was": true, "were": true, "what": true, "when": true,
	"where": true, "which": true, "who": true, "why": true, "with": true, "you": true,
}

func tokenize(text string) []string {
	raw := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	tokens := raw[:0]
	for _, t := range raw {
		if !stopwords[t] {
			tokens = append(tokens, t)
		}
	}
	return tokens
}

func newEvalPipeline(ec EmbeddingCreator, vs vectorstore.VectorStore) *RAGPipeline {
	return &RAGPipeline{
		config:           &config.Config{Port: "3001", OpenAIAPIKey: "test-key", DeepSeekAPIKey: "test-key"},
		embeddingCreator: ec,
		vectorStore:      vs,
		textSplitter:     utils.NewTextSplitter(chunkSize, chunkOverlap),
	}
}

type goldenCase struct {
	ID             string `json:"id"`
	Document       string `json:"document"`
	Question       string `json:"question"`
	ExpectedSource string `json:"expected_source"`
	ExpectedAnswer string `json:"expected_answer"` // reserved for a future answer-quality eval; the current harness is retrieval-only and does not assert on it.
}

func loadGoldenSet(t *testing.T) []goldenCase {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "eval", "golden.json"))
	assert.NoError(t, err)

	var cases []goldenCase
	assert.NoError(t, json.Unmarshal(raw, &cases))
	return cases
}

func loadDocument(t *testing.T, name string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "eval", "docs", name))
	assert.NoError(t, err)
	return string(raw)
}

func ingestGoldenDocs(t *testing.T, pipeline *RAGPipeline, cases []goldenCase) {
	t.Helper()
	ingested := make(map[string]bool)
	for _, gc := range cases {
		if ingested[gc.Document] {
			continue
		}
		content := loadDocument(t, gc.Document)
		doc := types.ProcessedDocument{NormalizedText: utils.Normalize(content)}
		chunks, err := pipeline.ProcessDocument(doc, map[string]string{"source": gc.Document})
		assert.NoError(t, err)
		assert.NoError(t, pipeline.AddDocumentToVectorStore(chunks))
		ingested[gc.Document] = true
	}
}

func normalizeWhitespace(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

func retrievedRank(t *testing.T, store vectorstore.VectorStore, dims int, question, expectedSource string, k int) int {
	t.Helper()
	scored, err := store.Search(embedText(question, dims), k)
	assert.NoError(t, err)

	needle := normalizeWhitespace(strings.ToLower(expectedSource))
	for i, sc := range scored {
		if strings.Contains(normalizeWhitespace(strings.ToLower(sc.Chunk.Content)), needle) {
			return i + 1
		}
	}
	return 0
}

func hitRateAt1(ranks []int) float64 {
	if len(ranks) == 0 {
		return 0
	}
	var hits int
	for _, r := range ranks {
		if r == 1 {
			hits++
		}
	}
	return float64(hits) / float64(len(ranks))
}

func recallAtK(ranks []int, k int) float64 {
	if len(ranks) == 0 {
		return 0
	}
	var hits int
	for _, r := range ranks {
		if r >= 1 && r <= k {
			hits++
		}
	}
	return float64(hits) / float64(len(ranks))
}

type evalMetrics struct {
	Cases     int     `json:"cases"`
	K         int     `json:"k"`
	HitAt1    float64 `json:"hit_at_1"`
	RecallAtK float64 `json:"recall_at_k"`
	MRR       float64 `json:"mrr"`
}

func writeEvalMetrics(t *testing.T, m evalMetrics) {
	t.Helper()
	path := os.Getenv("EVAL_METRICS_OUT")
	if path == "" {
		return
	}
	raw, err := json.Marshal(m)
	assert.NoError(t, err)
	assert.NoError(t, os.WriteFile(path, raw, 0o644))
}

func meanReciprocalRank(ranks []int) float64 {
	if len(ranks) == 0 {
		return 0
	}
	var sum float64
	for _, r := range ranks {
		if r >= 1 {
			sum += 1 / float64(r)
		}
	}
	return sum / float64(len(ranks))
}

func TestEvalRetrieval(t *testing.T) {
	const evalSearchK = maxContentChunks
	type threshold struct {
		hitRateAt1 float64
		recallAtK  float64
		mrr        float64
	}
	// Thresholds are calibrated just below the current baseline so the gate
	// fails on retrieval regressions; raise them as chunking/retrieval improves.
	want := threshold{hitRateAt1: 0.65, recallAtK: 0.85, mrr: 0.75}

	cases := loadGoldenSet(t)
	assert.NotEmpty(t, cases)

	embedder := &hashingEmbedder{dims: embeddingDims}
	store := memory.NewMemoryVectorStore()
	pipeline := newEvalPipeline(embedder, store)
	ingestGoldenDocs(t, pipeline, cases)

	ranks := make([]int, 0, len(cases))
	for _, gc := range cases {
		rank := retrievedRank(t, store, embeddingDims, gc.Question, gc.ExpectedSource, evalSearchK)
		ranks = append(ranks, rank)
		if rank == 0 {
			t.Logf("miss: %s not retrieved within top %d for %q", gc.ID, evalSearchK, gc.Question)
		}
	}

	got := threshold{
		hitRateAt1: hitRateAt1(ranks),
		recallAtK:  recallAtK(ranks, evalSearchK),
		mrr:        meanReciprocalRank(ranks),
	}
	t.Logf("retrieval eval over %d cases (k=%d): hit@1=%.3f recall@%d=%.3f mrr=%.3f",
		len(ranks), evalSearchK, got.hitRateAt1, evalSearchK, got.recallAtK, got.mrr)

	writeEvalMetrics(t, evalMetrics{
		Cases:     len(ranks),
		K:         evalSearchK,
		HitAt1:    got.hitRateAt1,
		RecallAtK: got.recallAtK,
		MRR:       got.mrr,
	})

	assert.GreaterOrEqual(t, got.hitRateAt1, want.hitRateAt1)
	assert.GreaterOrEqual(t, got.recallAtK, want.recallAtK)
	assert.GreaterOrEqual(t, got.mrr, want.mrr)
}

func TestRetrievalThreshold(t *testing.T) {
	type threshold struct {
		filteredRecall    float64
		negativeRejection float64
	}
	// Calibrated just below the current baseline so the gate fails on
	// regressions; raise them as retrieval improves (same ratchet philosophy as
	// TestEvalRetrieval). filteredRecall is lower than the unfiltered recall@K
	// because the deterministic bag-of-words embedder scores hard paraphrases
	// well below similarityThreshold; the production text-embedding-3-small model
	// separates paraphrases far better, so this floor is a regression guard, not
	// a quality target. negativeRejection must stay perfect: clearly off-domain
	// questions must never leak chunks into the prompt.
	want := threshold{filteredRecall: 0.30, negativeRejection: 1.0}

	cases := loadGoldenSet(t)
	assert.NotEmpty(t, cases)

	embedder := &hashingEmbedder{dims: embeddingDims}
	store := memory.NewMemoryVectorStore()
	pipeline := newEvalPipeline(embedder, store)
	ingestGoldenDocs(t, pipeline, cases)

	// Off-domain questions whose answers live in none of the golden documents
	// (Go, photosynthesis, HTTP caching, the water cycle, TCP/IP, the French
	// Revolution, embeddings). similarityThreshold should filter these to nothing.
	negatives := []string{
		"What is the capital city of Australia?",
		"How do you bake chocolate chip cookies from scratch?",
		"What are the official rules of basketball?",
		"Which composer wrote the Moonlight Sonata?",
	}

	var positiveHits int
	for _, gc := range cases {
		_, contextInfo, confidence, err := pipeline.retrieveContext(gc.Question)
		assert.NoError(t, err)
		needle := normalizeWhitespace(strings.ToLower(gc.ExpectedSource))
		if strings.Contains(normalizeWhitespace(strings.ToLower(contextInfo)), needle) {
			positiveHits++
			assert.Greaterf(t, confidence, 0.0, "retained positive %s should have non-zero confidence", gc.ID)
		}
	}

	var negativesRejected int
	for _, q := range negatives {
		docs, _, confidence, err := pipeline.retrieveContext(q)
		assert.NoError(t, err)
		if len(docs) == 0 {
			negativesRejected++
			assert.Equalf(t, 0.0, confidence, "filtered question should have zero confidence: %q", q)
		}
	}

	got := threshold{
		filteredRecall:    float64(positiveHits) / float64(len(cases)),
		negativeRejection: float64(negativesRejected) / float64(len(negatives)),
	}
	t.Logf("threshold eval (similarityThreshold=%.2f) over %d positives / %d negatives: filtered_recall=%.3f negative_rejection=%.3f",
		similarityThreshold, len(cases), len(negatives), got.filteredRecall, got.negativeRejection)

	assert.GreaterOrEqual(t, got.filteredRecall, want.filteredRecall)
	assert.GreaterOrEqual(t, got.negativeRejection, want.negativeRejection)
}

func TestEvalNeighborExpansionRecoversCrossChunkAnswer(t *testing.T) {
	densePara := func() string {
		return strings.Repeat("quantum entanglement correlation analysis. ", 14)
	}
	answerNeedle := "zero point eight seven"
	answerPara := "The recorded measurement value equaled " + answerNeedle + " units. " +
		strings.Repeat("laboratory staff logged ambient temperature during sessions. ", 9)

	paragraphs := []string{
		densePara(),
		densePara(),
		densePara(),
		answerPara,
		densePara(),
		densePara(),
	}
	content := strings.Join(paragraphs, "\n\n")

	embedder := &hashingEmbedder{dims: embeddingDims}
	store := memory.NewMemoryVectorStore()
	pipeline := newEvalPipeline(embedder, store)

	doc := types.ProcessedDocument{NormalizedText: utils.Normalize(content)}
	chunks, err := pipeline.ProcessDocument(doc, map[string]string{"source": "physics"})
	assert.NoError(t, err)
	assert.NoError(t, pipeline.AddDocumentToVectorStore(chunks))
	assert.Greater(t, len(chunks), maxContentChunks, "need more chunks than k so the answer chunk falls outside top-k")

	scored, err := store.Search(embedText("quantum entanglement correlation", embeddingDims), maxContentChunks)
	assert.NoError(t, err)

	hits := make([]types.DocumentChunk, len(scored))
	var bareContext strings.Builder
	for i, sc := range scored {
		hits[i] = sc.Chunk
		bareContext.WriteString(sc.Chunk.Content)
		bareContext.WriteString("\n\n")
	}

	expanded := pipeline.expandContext(hits)

	assert.NotContains(t, bareContext.String(), answerNeedle,
		"baseline top-k retrieval should miss the cross-chunk answer detail")
	assert.Contains(t, expanded, answerNeedle,
		"neighbor expansion should recover the adjacent answer chunk")
}

func TestEvalMetrics(t *testing.T) {
	type input struct {
		ranks []int
		k     int
	}
	type expected struct {
		hitRateAt1 float64
		recallAtK  float64
		mrr        float64
	}

	tests := []struct {
		name     string
		input    input
		expected expected
	}{
		{
			name:     "all retrieved at rank one",
			input:    input{ranks: []int{1, 1, 1}, k: 5},
			expected: expected{hitRateAt1: 1, recallAtK: 1, mrr: 1},
		},
		{
			name:     "mixed ranks within and beyond k",
			input:    input{ranks: []int{1, 2, 6, 0}, k: 5},
			expected: expected{hitRateAt1: 0.25, recallAtK: 0.5, mrr: 0.4166666667},
		},
		{
			name:     "nothing retrieved",
			input:    input{ranks: []int{0, 0}, k: 5},
			expected: expected{hitRateAt1: 0, recallAtK: 0, mrr: 0},
		},
		{
			name:     "empty input",
			input:    input{ranks: []int{}, k: 5},
			expected: expected{hitRateAt1: 0, recallAtK: 0, mrr: 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.InDelta(t, tt.expected.hitRateAt1, hitRateAt1(tt.input.ranks), 1e-9)
			assert.InDelta(t, tt.expected.recallAtK, recallAtK(tt.input.ranks, tt.input.k), 1e-9)
			assert.InDelta(t, tt.expected.mrr, meanReciprocalRank(tt.input.ranks), 1e-9)
		})
	}
}
