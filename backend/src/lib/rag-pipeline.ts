import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { Document as LangchainDocument } from '@langchain/core/documents';
import { DocumentChunk, RAGResponse } from '@/types';

export class RAGPipeline {
  private readonly embeddings: OpenAIEmbeddings;
  private readonly llm: ChatOpenAI;
  private vectorStore: MemoryVectorStore | null = null;
  private readonly textSplitter: RecursiveCharacterTextSplitter;

  constructor() {
    // OpenAI embeddings
    this.embeddings = new OpenAIEmbeddings({
      apiKey: process.env.OPENAI_API_KEY,
    });

    // DeepSeek LLM
    this.llm = new ChatOpenAI({
      apiKey: process.env.DEEPSEEK_API_KEY,
      configuration: {
        baseURL: 'https://api.deepseek.com/v1',
      },
      model: 'deepseek-chat',
      temperature: 0,
    });

    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
  }

  async initializeVectorStore() {
    if (!this.vectorStore) {
      this.vectorStore = new MemoryVectorStore(this.embeddings);
    }
    return this.vectorStore;
  }

  async processDocument(content: string, metadata: { source: string }): Promise<DocumentChunk[]> {
    const docs = await this.textSplitter.createDocuments([content], [metadata]);

    const chunks: DocumentChunk[] = docs.map((doc, index) => ({
      id: `${metadata.source}-chunk-${index}`,
      content: doc.pageContent,
      metadata: {
        source: metadata.source,
        ...doc.metadata,
      },
    }));

    return chunks;
  }

  async addDocumentToVectorStore(chunks: DocumentChunk[]) {
    const vectorStore = await this.initializeVectorStore();

    const langchainDocs = chunks.map(chunk =>
      new LangchainDocument({
        pageContent: chunk.content,
        metadata: chunk.metadata,
      })
    );

    await vectorStore.addDocuments(langchainDocs);
  }

  async query(question: string, k: number = 4): Promise<RAGResponse> {
    const vectorStore = await this.initializeVectorStore();

    const relevantDocs = await vectorStore.similaritySearch(question, k);

    const context = relevantDocs.map(doc => doc.pageContent).join('\n\n');

    const prompt = `Context information:
${context}

Question: ${question}

Please answer the question based on the context provided and always try to make direct citations from the document (always in the original language of the document, you don't have to translate citations) to endorse your answer with the source, make it more reliable and avoid hallucinations. If the answer is not in the context, say you don't have enough information to answer the question. It's important to always answer the question in the language used by the user to ask it`;

    const response = await this.llm.invoke(prompt);

    const sources: DocumentChunk[] = relevantDocs.map((doc, index) => ({
      id: `result-${index}`,
      content: doc.pageContent,
      metadata: doc.metadata as { source: string },
    }));

    return {
      answer: response.content as string,
      sources,
      confidence: 0.8,
    };
  }
}