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
	maxBatchSize      = 40
	maxConcurrency    = 5
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

// StreamEvent is a unit emitted by QueryStream. Exactly one of Sources (with
// Confidence), Token, Err, or Done is meaningful per event.
type StreamEvent struct {
	Sources    []types.DocumentChunk
	Confidence float64
	Token      string
	Err        error
	Done       bool
}

// QueryStream runs the retrieval stage synchronously (so pre-stream errors
// surface normally) and then emits events on the returned channel: first the
// retrieved sources, then one Token event per streamed delta, then either Done
// on clean finish or Err on failure. The channel is closed when the stream
// ends or ctx is cancelled.
func (rp *RAGPipeline) QueryStream(ctx context.Context, question string) (<-chan StreamEvent, error) {
	relevantDocs, contextInfo, err := rp.retrieveContext(question)
	if err != nil {
		return nil, err
	}

	events := make(chan StreamEvent)
	go rp.streamCompletion(ctx, relevantDocs, contextInfo, question, events)
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

func (rp *RAGPipeline) streamCompletion(ctx context.Context, sources []types.DocumentChunk, contextInfo, question string, events chan<- StreamEvent) {
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

	stream := rp.chatCompleter.NewStreamingIter(ctx, chatCompletionParams(buildPrompt(contextInfo, question)))
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

func (rp *RAGPipeline) Query(question string) (*types.RAGResponse, error) {
	relevantDocs, contextInfo, err := rp.retrieveContext(question)
	if err != nil {
		return nil, err
	}

	answer, err := rp.generateResponse(contextInfo, question)
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

func buildPrompt(contextInfo, question string) string {
	return fmt.Sprintf(`Context information:
%s

Question: %s

Please answer the question based on the context provided. If the answer is not in the context, say "I don't have enough information to answer this question."`, contextInfo, question)
}

func chatCompletionParams(prompt string) openai.ChatCompletionNewParams {
	return openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(prompt),
		},
		Model:       "deepseek-chat",
		Temperature: openai.Float(0.0), // Deterministic: same question = same answer.
	}
}

func (rp *RAGPipeline) generateResponse(contextInfo, question string) (string, error) {
	completion, err := rp.chatCompleter.New(context.TODO(), chatCompletionParams(buildPrompt(contextInfo, question)))
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
