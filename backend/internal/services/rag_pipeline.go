package services

import (
	"context"
	"fmt"
	"rag-backend/internal/repositories/vectorstore"
	"strings"
	"sync"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"

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
	config         *config.Config
	openaiClient   openai.Client
	deepseekClient openai.Client
	vectorStore    vectorstore.VectorStore
	textSplitter   *utils.TextSplitter
	mutex          sync.RWMutex
}

func NewRAGPipeline(cfg *config.Config, vectorStore vectorstore.VectorStore) *RAGPipeline {
	return &RAGPipeline{
		config:       cfg,
		openaiClient: openai.NewClient(option.WithAPIKey(cfg.OpenAIAPIKey)),
		deepseekClient: openai.NewClient(
			option.WithAPIKey(cfg.DeepSeekAPIKey),
			option.WithBaseURL("https://api.deepseek.com/v1"),
		),
		vectorStore:  vectorStore,
		textSplitter: utils.NewTextSplitter(chunkSize, chunkOverlap),
	}
}

func (rp *RAGPipeline) ProcessDocument(content string, metadata map[string]string) ([]types.DocumentChunk, error) {
	textChunks := rp.textSplitter.SplitText(content)

	// Generate embeddings for all chunks in a single batch API call
	embeddings, err := rp.generateEmbeddingBatch(textChunks)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embeddings: %w", err)
	}

	// Create chunks with the batch-generated embeddings
	chunks := make([]types.DocumentChunk, len(textChunks))
	for i, textChunk := range textChunks {
		chunks[i] = types.DocumentChunk{
			ID:        fmt.Sprintf("%s-chunk-%d", metadata["source"], i),
			Content:   textChunk,
			Embedding: embeddings[i],
			Metadata:  metadata,
		}
	}

	return chunks, nil
}

func (rp *RAGPipeline) AddDocumentToVectorStore(chunks []types.DocumentChunk) error {
	if err := rp.vectorStore.Store(chunks); err != nil {
		return fmt.Errorf("failed to store chunks: %w", err)
	}
	return nil
}

func (rp *RAGPipeline) Query(question string) (*types.RAGResponse, error) {
	queryEmbedding, err := rp.generateEmbedding(question)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding for query: %w", err)
	}

	scoredChunks, err := rp.vectorStore.Search(queryEmbedding, maxContentChunks)
	if err != nil {
		return nil, fmt.Errorf("failed to search vector store: %w", err)
	}

	// Build context from relevant documents
	var contextBuilder strings.Builder
	relevantDocs := make([]types.DocumentChunk, len(scoredChunks))
	for i, scored := range scoredChunks {
		if i > 0 {
			contextBuilder.WriteString("\n\n")
		}
		contextBuilder.WriteString(scored.Chunk.Content)
		relevantDocs[i] = scored.Chunk
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
	embedding, err := rp.openaiClient.Embeddings.New(context.TODO(), openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{
			OfString: openai.String(text),
		},
		Model: openai.EmbeddingModelTextEmbedding3Small,
	})
	if err != nil {
		return nil, err
	}

	if len(embedding.Data) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	// TODO Handle different embedding types if needed so I dont have to make this conversion
	embedding32 := embedding.Data[0].Embedding
	embedding64 := make([]float64, len(embedding32))
	for i, v := range embedding32 {
		embedding64[i] = float64(v)
	}
	return embedding64, nil
}

func (rp *RAGPipeline) generateEmbeddingBatch(texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("no texts provided for batch embedding")
	}

	embedding, err := rp.openaiClient.Embeddings.New(context.TODO(), openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: texts,
		},
		Model: openai.EmbeddingModelTextEmbedding3Small,
	})
	if err != nil {
		return nil, err
	}

	if len(embedding.Data) != len(texts) {
		return nil, fmt.Errorf("expected %d embeddings, got %d", len(texts), len(embedding.Data))
	}

	// TODO Handle different embedding types if needed so I dont have to make this conversion
	embeddings := make([][]float64, len(embedding.Data))
	for i, embData := range embedding.Data {
		embedding32 := embData.Embedding
		embedding64 := make([]float64, len(embedding32))
		for j, v := range embedding32 {
			embedding64[j] = float64(v)
		}
		embeddings[i] = embedding64
	}

	return embeddings, nil
}

func (rp *RAGPipeline) generateResponse(contextInfo, question string) (string, error) {
	prompt := fmt.Sprintf(`Context information:
%s

Question: %s

Please answer the question based on the context provided. If the answer is not in the context, say "I don't have enough information to answer this question."`, contextInfo, question)

	completion, err := rp.deepseekClient.Chat.Completions.New(context.TODO(), openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(prompt),
		},
		Model:       "deepseek-chat",
		Temperature: openai.Float(0.0), // The model will be deterministic and always choose the most likely next token. Same question = same answer every time
	})
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}

	if len(completion.Choices) == 0 {
		return "", fmt.Errorf("no response from DeepSeek API")
	}

	return completion.Choices[0].Message.Content, nil
}
