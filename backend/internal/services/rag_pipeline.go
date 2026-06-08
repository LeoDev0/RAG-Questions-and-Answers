package services

import (
	"context"
	"fmt"
	"rag-backend/internal/repositories/vectorstore"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"

	"rag-backend/internal/config"
	"rag-backend/pkg/types"
	"rag-backend/pkg/utils"
)

const (
	chunkSize        = 1000
	chunkOverlap     = 200
	maxContentChunks = 4
	maxBatchSize     = 40
	maxConcurrency   = 5
	// neighborRadius controls how many adjacent chunks are pulled in around
	// each search hit to give the LLM fuller surrounding context than the
	// matched fragment alone.
	neighborRadius = 1
	// maxContextChars bounds the assembled context (in bytes) sent to the LLM as
	// a safety rail if neighborRadius grows. Search hits are always kept;
	// neighbors are dropped first when over budget.
	maxContextChars = 24000
	// similarityThreshold is the minimum cosine score a chunk must reach to be
	// fed to the LLM. Chunks below it are dropped so weak matches don't dilute
	// the prompt. It is a starting default for text-embedding-3-small relevance
	// scores; the eval harness (TestRetrievalThreshold) validates the filtering
	// mechanism and the positive/negative score separation offline. A chunk is
	// retained iff its score is >= this value.
	similarityThreshold = 0.3
	// maxHistoryTurns bounds how many prior turns are sent to the LLM as
	// conversational context. retrievalRewriteWindow bounds how many recent
	// user turns are folded into the embedding query for vector search.
	// The retrieval window is much smaller because stuffing many turns into
	// a single embedding dilutes the topic signal; only the last turn or two
	// are needed to resolve follow-up references like "its" or "that".
	maxHistoryTurns        = 10
	retrievalRewriteWindow = 2
)

type RAGPipeline struct {
	config           *config.Config
	embeddingCreator EmbeddingCreator
	chatCompleter    ChatCompletionCreator
	vectorStore      vectorstore.VectorStore
	textSplitter     *utils.TextSplitter
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

func (rp *RAGPipeline) ProcessDocument(doc types.ProcessedDocument, metadata map[string]string) ([]types.DocumentChunk, error) {
	normalized := doc.NormalizedText
	textChunks := rp.textSplitter.SplitText(normalized)

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

	source := metadata["source"]
	chunks := make([]types.DocumentChunk, len(textChunks))
	cursor := 0
	for i, textChunk := range textChunks {
		start, end := 0, 0
		if idx := strings.Index(normalized[cursor:], textChunk); idx >= 0 {
			start = cursor + idx
			end = start + len(textChunk)
			cursor = start + 1
		}
		page := pageForOffset(doc.PageSpans, start)
		chunkMetadata := metadata
		if page > 0 {
			chunkMetadata = cloneMetadata(metadata)
			chunkMetadata["page"] = strconv.Itoa(page)
		}
		chunks[i] = types.DocumentChunk{
			ID:          fmt.Sprintf("%s-chunk-%d", source, i),
			Content:     textChunk,
			Embedding:   embeddings[i],
			Metadata:    chunkMetadata,
			Source:      source,
			ChunkIndex:  i,
			StartOffset: start,
			EndOffset:   end,
			Page:        page,
		}
	}

	return chunks, nil
}

// pageForOffset maps a byte offset in the normalized text to the page whose
// span contains it, attributing a chunk to the page of its StartOffset. The
// "\n\n" gap between page spans falls to the preceding page. Returns 0 when
// the document carries no page spans (e.g. plain-text uploads).
func pageForOffset(spans []types.PageSpan, offset int) int {
	if len(spans) == 0 {
		return 0
	}
	i := sort.Search(len(spans), func(i int) bool { return spans[i].Start > offset }) - 1
	if i < 0 {
		return 0
	}
	return spans[i].Page
}

func cloneMetadata(metadata map[string]string) map[string]string {
	clone := make(map[string]string, len(metadata)+1)
	for k, v := range metadata {
		clone[k] = v
	}
	return clone
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

	relevantDocs, contextInfo, confidence, err := rp.retrieveContext(retrievalQuery)
	if err != nil {
		return nil, err
	}

	events := make(chan StreamEvent)
	go rp.streamCompletion(ctx, relevantDocs, contextInfo, confidence, history, question, events)
	return events, nil
}

func (rp *RAGPipeline) retrieveContext(question string) ([]types.DocumentChunk, string, float64, error) {
	queryEmbedding, err := rp.generateEmbedding(question)
	if err != nil {
		return nil, "", 0, fmt.Errorf("failed to generate embedding for query: %w", err)
	}

	scoredChunks, err := rp.vectorStore.Search(queryEmbedding, maxContentChunks)
	if err != nil {
		return nil, "", 0, fmt.Errorf("failed to search vector store: %w", err)
	}

	retained := retainAboveThreshold(scoredChunks)

	relevantDocs := make([]types.DocumentChunk, 0, len(retained))
	for _, scored := range retained {
		relevantDocs = append(relevantDocs, scored.Chunk)
	}

	return relevantDocs, rp.expandContext(relevantDocs), confidenceFromTopScore(retained), nil
}

func (rp *RAGPipeline) expandContext(hits []types.DocumentChunk) string {
	byID := make(map[string]types.DocumentChunk, len(hits))
	hitIDs := make(map[string]bool, len(hits))
	for _, hit := range hits {
		byID[hit.ID] = hit
		hitIDs[hit.ID] = true
		neighbors, err := rp.vectorStore.Neighbors(hit.Source, hit.ChunkIndex, neighborRadius)
		if err != nil {
			// Degrade gracefully: keep the hit, skip its neighbors.
			continue
		}
		for _, neighbor := range neighbors {
			if _, seen := byID[neighbor.ID]; !seen {
				byID[neighbor.ID] = neighbor
			}
		}
	}

	ordered := make([]types.DocumentChunk, 0, len(byID))
	for _, chunk := range byID {
		ordered = append(ordered, chunk)
	}
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].Source != ordered[j].Source {
			return ordered[i].Source < ordered[j].Source
		}
		if ordered[i].ChunkIndex != ordered[j].ChunkIndex {
			return ordered[i].ChunkIndex < ordered[j].ChunkIndex
		}
		return ordered[i].ID < ordered[j].ID
	})

	selected := selectWithinBudget(ordered, hitIDs, maxContextChars)
	return assembleContext(selected)
}

func selectWithinBudget(ordered []types.DocumentChunk, hitIDs map[string]bool, budget int) []types.DocumentChunk {
	total := 0
	for _, c := range ordered {
		if hitIDs[c.ID] {
			total += len(c.Content)
		}
	}

	selected := make([]types.DocumentChunk, 0, len(ordered))
	for _, c := range ordered {
		if hitIDs[c.ID] {
			selected = append(selected, c)
			continue
		}
		if total+len(c.Content) <= budget {
			total += len(c.Content)
			selected = append(selected, c)
		}
	}
	return selected
}

func assembleContext(chunks []types.DocumentChunk) string {
	var b strings.Builder
	var prevSource string
	var prevEnd int
	started := false

	for _, c := range chunks {
		if started && c.Source == prevSource && prevEnd > 0 && c.EndOffset > 0 && c.StartOffset < prevEnd {
			overlap := prevEnd - c.StartOffset
			if overlap < len(c.Content) {
				b.WriteString(c.Content[overlap:])
			}
			prevEnd = max(prevEnd, c.EndOffset)
			continue
		}

		if started {
			b.WriteString("\n\n")
		}
		b.WriteString(c.Content)
		prevSource = c.Source
		prevEnd = c.EndOffset
		started = true
	}

	return b.String()
}

func retainAboveThreshold(scored []types.ScoredChunk) []types.ScoredChunk {
	retained := make([]types.ScoredChunk, 0, len(scored))
	for _, sc := range scored {
		if sc.Score >= similarityThreshold {
			retained = append(retained, sc)
		}
	}
	return retained
}

func confidenceFromTopScore(retained []types.ScoredChunk) float64 {
	if len(retained) == 0 {
		return 0.0
	}
	top := retained[0].Score
	for _, sc := range retained[1:] {
		if sc.Score > top {
			top = sc.Score
		}
	}
	if top < 0 {
		return 0.0
	}
	if top > 1 {
		return 1.0
	}
	return top
}

func (rp *RAGPipeline) streamCompletion(ctx context.Context, sources []types.DocumentChunk, contextInfo string, confidence float64, history []types.Message, question string, events chan<- StreamEvent) {
	defer close(events)

	send := func(ev StreamEvent) bool {
		select {
		case <-ctx.Done():
			return false
		case events <- ev:
			return true
		}
	}

	if !send(StreamEvent{Sources: sources, Confidence: confidence}) {
		return
	}

	stream := rp.chatCompleter.NewStreamingIter(ctx, chatCompletionParams(buildSystemPrompt(contextInfo), history, question))
	defer func() { _ = stream.Close() }()

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

	relevantDocs, contextInfo, confidence, err := rp.retrieveContext(retrievalQuery)
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
		Confidence: confidence,
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

	return embedding.Data[0].Embedding, nil
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

	embeddings := make([][]float64, len(embedding.Data))
	for _, embData := range embedding.Data {
		embeddings[embData.Index] = embData.Embedding
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

	results := make([]batchResult, len(batches))
	for result := range resultChan {
		if result.err != nil {
			return nil, fmt.Errorf("failed to generate embeddings for batch %d: %w", result.index, result.err)
		}
		results[result.index] = result
	}

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
	const nonHistoryMessages = 2 // system prompt + current user question
	msgs := make([]openai.ChatCompletionMessageParamUnion, 0, len(history)+nonHistoryMessages)
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
