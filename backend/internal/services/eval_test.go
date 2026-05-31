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
		data[i] = openai.Embedding{Embedding: embedText(text, e.dims)}
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

func tokenize(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
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
	want := threshold{hitRateAt1: 0.45, recallAtK: 0.7, mrr: 0.55}

	cases := loadGoldenSet(t)
	assert.NotEmpty(t, cases)

	embedder := &hashingEmbedder{dims: embeddingDims}
	store := memory.NewMemoryVectorStore()
	pipeline := newEvalPipeline(embedder, store)

	ingested := make(map[string]bool)
	for _, gc := range cases {
		if ingested[gc.Document] {
			continue
		}
		content := loadDocument(t, gc.Document)
		chunks, err := pipeline.ProcessDocument(content, map[string]string{"source": gc.Document})
		assert.NoError(t, err)
		assert.NoError(t, pipeline.AddDocumentToVectorStore(chunks))
		ingested[gc.Document] = true
	}

	ranks := make([]int, 0, len(cases))
	for _, gc := range cases {
		t.Run(gc.ID, func(t *testing.T) {
			rank := retrievedRank(t, store, embeddingDims, gc.Question, gc.ExpectedSource, evalSearchK)
			assert.NotZero(t, rank, "expected source not retrieved within top %d for %q", evalSearchK, gc.Question)
			ranks = append(ranks, rank)
		})
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
