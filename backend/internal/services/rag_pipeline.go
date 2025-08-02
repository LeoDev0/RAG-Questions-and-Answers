package services

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"

	"github.com/sashabaranov/go-openai"

	"rag-backend/internal/config"
	"rag-backend/pkg/types"
	"rag-backend/pkg/utils"
)

const (
	defaultConfidence = 0.8
	chunkSize         = 1000
	chunkOverlap      = 200
	maxContentChunks  = 4
)

type RAGPipeline struct {
	config       *config.Config
	openaiClient *openai.Client
	vectorStore  *MemoryVectorStore
	textSplitter *utils.TextSplitter
	mutex        sync.RWMutex
}

type MemoryVectorStore struct {
	documents []types.DocumentChunk
	mutex     sync.RWMutex
}

func NewMemoryVectorStore() *MemoryVectorStore {
	return &MemoryVectorStore{
		documents: make([]types.DocumentChunk, 0),
	}
}

func (mvs *MemoryVectorStore) AddDocuments(chunks []types.DocumentChunk) {
	mvs.mutex.Lock()
	defer mvs.mutex.Unlock()
	mvs.documents = append(mvs.documents, chunks...)
}

func (mvs *MemoryVectorStore) SimilaritySearch(queryEmbedding []float64) []types.DocumentChunk {
	mvs.mutex.RLock()
	defer mvs.mutex.RUnlock()

	if len(mvs.documents) == 0 {
		return []types.DocumentChunk{}
	}

	// Calculate cosine similarity for each document
	type ScoredChunk struct {
		Chunk types.DocumentChunk
		Score float64
	}

	scored := make([]ScoredChunk, 0, len(mvs.documents))
	for _, chunk := range mvs.documents {
		if len(chunk.Embedding) == 0 {
			continue
		}
		score := cosineSimilarity(queryEmbedding, chunk.Embedding)
		scored = append(scored, ScoredChunk{Chunk: chunk, Score: score})
	}

	// Sort by score (descending)
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score > scored[j].Score
	})

	// Return top k results
	k := maxContentChunks
	if k > len(scored) {
		k = len(scored)
	}

	results := make([]types.DocumentChunk, k)
	for i := 0; i < k; i++ {
		results[i] = scored[i].Chunk
	}

	return results
}

func NewRAGPipeline(cfg *config.Config) *RAGPipeline {
	return &RAGPipeline{
		config:       cfg,
		openaiClient: openai.NewClient(cfg.OpenAIAPIKey),
		vectorStore:  NewMemoryVectorStore(),
		textSplitter: utils.NewTextSplitter(chunkSize, chunkOverlap),
	}
}

func (rp *RAGPipeline) ProcessDocument(content string, metadata map[string]string) ([]types.DocumentChunk, error) {
	textChunks := rp.textSplitter.SplitText(content)

	chunks := make([]types.DocumentChunk, len(textChunks))

	for i, textChunk := range textChunks {
		// Generate embedding for this chunk
		embedding, err := rp.generateEmbedding(textChunk)
		if err != nil {
			return nil, fmt.Errorf("failed to generate embedding for chunk %d: %w", i, err)
		}

		chunks[i] = types.DocumentChunk{
			ID:        fmt.Sprintf("%s-chunk-%d", metadata["source"], i),
			Content:   textChunk,
			Embedding: embedding,
			Metadata:  metadata,
		}
	}

	return chunks, nil
}

func (rp *RAGPipeline) AddDocumentToVectorStore(chunks []types.DocumentChunk) {
	rp.vectorStore.AddDocuments(chunks)
}

func (rp *RAGPipeline) Query(question string) (*types.RAGResponse, error) {
	queryEmbedding, err := rp.generateEmbedding(question)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding for query: %w", err)
	}

	relevantDocs := rp.vectorStore.SimilaritySearch(queryEmbedding)

	// Build context from relevant documents
	var contextBuilder strings.Builder
	for i, doc := range relevantDocs {
		if i > 0 {
			contextBuilder.WriteString("\n\n")
		}
		contextBuilder.WriteString(doc.Content)
	}

	answer, err := rp.generateResponse(contextBuilder.String(), question)
	if err != nil {
		return nil, fmt.Errorf("failed to generate response: %w", err)
	}

	return &types.RAGResponse{
		Answer:     answer,
		Sources:    relevantDocs,
		Confidence: defaultConfidence, // Static confidence for now
	}, nil
}

func (rp *RAGPipeline) generateEmbedding(text string) ([]float64, error) {
	req := openai.EmbeddingRequest{
		Input: []string{text},
		Model: openai.AdaEmbeddingV2,
	}

	resp, err := rp.openaiClient.CreateEmbeddings(context.TODO(), req)
	if err != nil {
		return nil, err
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	// fix return type to float32 so I dont have to make this ridiculous conversion
	embedding32 := resp.Data[0].Embedding
	embedding64 := make([]float64, len(embedding32))
	for i, v := range embedding32 {
		embedding64[i] = float64(v)
	}
	return embedding64, nil
}

func (rp *RAGPipeline) generateResponse(context, question string) (string, error) {
	prompt := fmt.Sprintf(`Context information:
%s

Question: %s

Please answer the question based on the context provided. If the answer is not in the context, say "I don't have enough information to answer this question."`, context, question)

	payload := map[string]interface{}{
		"model": "deepseek-chat",
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": prompt,
			},
		},
		"temperature": 0,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", "https://api.deepseek.com/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+rp.config.DeepSeekAPIKey)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var response DeepSeekResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", err
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("no response from DeepSeek API")
	}

	return response.Choices[0].Message.Content, nil
}

// cosineSimilarity calculates cosine similarity between two vectors
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (normA * normB)
}

type DeepSeekResponse struct {
	Choices []Choice `json:"choices"`
}

type Choice struct {
	Message Message `json:"message"`
}

type Message struct {
	Content string `json:"content"`
}
