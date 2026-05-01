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
	defaultConfidence      = 0.8
	chunkSize              = 1000
	chunkOverlap           = 200
	maxContentChunks       = 4
	maxBatchSize           = 40
	maxConcurrency         = 5
	maxHistoryTurns        = 10
	retrievalRewriteWindow = 2
)

type RAGPipeline struct {
	config           *config.Config
	embeddingCreator EmbeddingCreator
	chatCompleter    ChatCompletionCreator
	vectorStore      vectorstore.VectorStore
	textSplitter     *utils.TextSplitter
	mutex            sync.RWMutex
}

func NewRAGPipeline(cfg *config.Config, vectorStore vectorstore.VectorStore) *RAGPipeline {
	openaiClient := openai.NewClient(option.WithAPIKey(cfg.OpenAIAPIKey))
	deepseekClient := openai.NewClient(
		option.WithAPIKey(cfg.DeepSeekAPIKey),
		option.WithBaseURL("https://api.deepseek.com/v1"),
	)
	return &RAGPipeline{
		config:           cfg,
		embeddingCreator: &openaiClient.Embeddings,
		chatCompleter:    &chatCompletionsAdapter{inner: &deepseekClient.Chat.Completions},
		vectorStore:      vectorStore,
		textSplitter:     utils.NewTextSplitter(chunkSize, chunkOverlap),
	}
}

func (rp *RAGPipeline) ProcessDocument(content string, metadata map[string]string) ([]types.DocumentChunk, error) {
	textChunks := rp.textSplitter.SplitText(content)

	var embeddings [][]float64
	var err error

	if len(textChunks) > maxBatchSize {
		// Use parallel batch processing for large documents
		embeddings, err = rp.generateEmbeddingParallel(textChunks)
	} else {
		// Use single batch processing for small documents
		embeddings, err = rp.generateEmbeddingBatch(textChunks)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to generate embeddings: %w", err)
	}

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

type StreamEvent struct {
	Sources    []types.DocumentChunk
	Confidence float64
	Token      string
	Err        error
	Done       bool
}

func (rp *RAGPipeline) QueryStream(ctx context.Context, question string, history []types.Message) (<-chan StreamEvent, error) {
	history = trimHistory(history)
	retrievalQuery := rewriteQueryForRetrieval(history, question)

	relevantDocs, contextInfo, err := rp.retrieveContext(retrievalQuery)
	if err != nil {
		return nil, err
	}

	events := make(chan StreamEvent)
	go rp.streamCompletion(ctx, relevantDocs, contextInfo, history, question, events)
	return events, nil
}

func (rp *RAGPipeline) retrieveContext(question string) ([]types.DocumentChunk, string, error) {
	queryEmbedding, err := rp.generateEmbedding(question)
	if err != nil {
		return nil, "", fmt.Errorf("failed to generate embedding for query: %w", err)
	}

	scoredChunks, err := rp.vectorStore.Search(queryEmbedding, maxContentChunks)
	if err != nil {
		return nil, "", fmt.Errorf("failed to search vector store: %w", err)
	}

	var contextBuilder strings.Builder
	relevantDocs := make([]types.DocumentChunk, len(scoredChunks))
	for i, scored := range scoredChunks {
		if i > 0 {
			contextBuilder.WriteString("\n\n")
		}
		contextBuilder.WriteString(scored.Chunk.Content)
		relevantDocs[i] = scored.Chunk
	}

	return relevantDocs, contextBuilder.String(), nil
}

func (rp *RAGPipeline) streamCompletion(ctx context.Context, sources []types.DocumentChunk, contextInfo string, history []types.Message, question string, events chan<- StreamEvent) {
	defer close(events)

	send := func(ev StreamEvent) bool {
		select {
		case <-ctx.Done():
			return false
		case events <- ev:
			return true
		}
	}

	if !send(StreamEvent{Sources: sources, Confidence: defaultConfidence}) {
		return
	}

	stream := rp.chatCompleter.NewStreamingIter(ctx, chatCompletionParams(buildSystemPrompt(contextInfo), history, question))
	defer stream.Close()

	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) == 0 {
			continue
		}
		content := chunk.Choices[0].Delta.Content
		if content == "" {
			continue
		}
		if !send(StreamEvent{Token: content}) {
			return
		}
	}

	if err := stream.Err(); err != nil {
		send(StreamEvent{Err: fmt.Errorf("stream failed: %w", err)})
		return
	}

	send(StreamEvent{Done: true})
}

func (rp *RAGPipeline) Query(question string, history []types.Message) (*types.RAGResponse, error) {
	history = trimHistory(history)
	retrievalQuery := rewriteQueryForRetrieval(history, question)

	relevantDocs, contextInfo, err := rp.retrieveContext(retrievalQuery)
	if err != nil {
		return nil, err
	}

	answer, err := rp.generateResponse(contextInfo, history, question)
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
	embedding, err := rp.embeddingCreator.New(context.TODO(), openai.EmbeddingNewParams{
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

	embedding, err := rp.embeddingCreator.New(context.TODO(), openai.EmbeddingNewParams{
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

func (rp *RAGPipeline) generateEmbeddingParallel(texts []string) ([][]float64, error) {
	// Split texts into batches of size equals to maxBatchSize
	batches := make([][]string, 0)
	for i := 0; i < len(texts); i += maxBatchSize {
		end := i + maxBatchSize
		end = min(end, len(texts))
		batches = append(batches, texts[i:end])
	}

	// Process batches in parallel with concurrency control
	// How It Works:
	//
	// 1. Channel as Gatekeeper: make(chan struct{}, N) creates a channel that can hold N "tokens"
	// 2. Acquire Token: semaphore <- struct{}{} - Goroutine waits here if channel is full
	// 3. Do Work: Only when there's space, the goroutine proceeds to make API call
	// 4. Release Token: <-semaphore - Frees up space for the next waiting goroutine one by one like a traffic light for goroutines
	//
	// This way I get parallel processing speed while staying within API limits and avoiding
	semaphore := make(chan struct{}, maxConcurrency)
	resultChan := make(chan batchResult, len(batches))
	var wg sync.WaitGroup

	for i, batch := range batches {
		wg.Add(1)
		go func(idx int, textBatch []string) {
			defer wg.Done()

			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			embeddings, err := rp.generateEmbeddingBatch(textBatch)
			resultChan <- batchResult{
				index:      idx,
				embeddings: embeddings,
				err:        err,
			}
		}(i, batch)
	}

	wg.Wait()
	close(resultChan)

	// Collect results in order
	results := make([]batchResult, len(batches))
	for result := range resultChan {
		if result.err != nil {
			return nil, fmt.Errorf("failed to generate embeddings for batch %d: %w", result.index, result.err)
		}
		results[result.index] = result
	}

	// Combine all embeddings in the correct order
	allEmbeddings := make([][]float64, 0, len(texts))
	for _, result := range results {
		allEmbeddings = append(allEmbeddings, result.embeddings...)
	}

	return allEmbeddings, nil
}

func buildSystemPrompt(contextInfo string) string {
	return fmt.Sprintf(`You are answering questions about a provided document.

Context information:
%s

Answer using only the context above. If the answer is not in the context, say "I don't have enough information to answer this question."`, contextInfo)
}

func chatCompletionParams(systemPrompt string, history []types.Message, question string) openai.ChatCompletionNewParams {
	msgs := make([]openai.ChatCompletionMessageParamUnion, 0, len(history)+2)
	msgs = append(msgs, openai.SystemMessage(systemPrompt))
	for _, m := range history {
		if m.Role == types.RoleAssistant {
			msgs = append(msgs, openai.AssistantMessage(m.Content))
		} else {
			msgs = append(msgs, openai.UserMessage(m.Content))
		}
	}
	msgs = append(msgs, openai.UserMessage(question))
	return openai.ChatCompletionNewParams{
		Messages:    msgs,
		Model:       "deepseek-chat",
		Temperature: openai.Float(0.0), // Deterministic: same question = same answer.
	}
}

func trimHistory(history []types.Message) []types.Message {
	if len(history) <= maxHistoryTurns {
		return history
	}
	return history[len(history)-maxHistoryTurns:]
}

// Concatenates the last retrievalRewriteWindow user turns with the current
// question so the embedding has more anchor for follow-up disambiguation.
// Centralized so the rewrite can later be swapped to an LLM-based one.
func rewriteQueryForRetrieval(history []types.Message, question string) string {
	var parts []string
	userTurns := 0
	for i := len(history) - 1; i >= 0 && userTurns < retrievalRewriteWindow; i-- {
		if history[i].Role == types.RoleUser {
			parts = append([]string{history[i].Content}, parts...)
			userTurns++
		}
	}
	parts = append(parts, question)
	return strings.Join(parts, " ")
}

func (rp *RAGPipeline) generateResponse(contextInfo string, history []types.Message, question string) (string, error) {
	completion, err := rp.chatCompleter.New(context.TODO(), chatCompletionParams(buildSystemPrompt(contextInfo), history, question))
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}

	if len(completion.Choices) == 0 {
		return "", fmt.Errorf("no response from DeepSeek API")
	}

	return completion.Choices[0].Message.Content, nil
}

type batchResult struct {
	index      int
	embeddings [][]float64
	err        error
}
