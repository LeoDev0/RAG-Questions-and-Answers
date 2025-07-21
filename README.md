# RAG Q&A

This is a Document Q&A Bot built with that implements Retrieval-Augmented Generation (RAG). Users can upload PDF or text documents and ask questions about their content.

## Development Commands

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run linting
npm run lint
```

## Environment Setup

Copy `.env.example` to `.env` and configure:
- `DEEPSEEK_API_KEY` - For LLM responses (DeepSeek Chat API)
- `OPENAI_API_KEY` - For document embeddings only

## Architecture

### Core RAG Pipeline (`src/lib/rag-pipeline.ts`)
- **LLM**: DeepSeek Chat API via ChatOpenAI with custom baseURL configuration
- **Embeddings**: OpenAI embeddings for vector similarity search
- **Vector Store**: MemoryVectorStore (documents reset on server restart)
- **Text Splitting**: 1000 character chunks with 200 character overlap

### API Endpoints
- `POST /api/upload` - Processes and stores documents (PDF/text, max 10MB)
- `POST /api/query` - Performs RAG queries against uploaded documents

### Data Flow
1. Documents uploaded via `/api/upload` → processed by `DocumentProcessor` → chunked → embedded → stored in vector store
2. Questions via `/api/query` → similarity search → context retrieval → LLM prompt → response with sources

### Key Components
- `RAGPipeline` - Core RAG logic with DeepSeek + OpenAI integration
- `DocumentProcessor` - PDF and text file processing utilities
- `MemoryVectorStore` - In-memory vector storage (non-persistent)
- Type definitions in `src/types/index.ts` for Document, DocumentChunk, ChatMessage, RAGResponse

### Frontend
- Single-page React app with file upload and chat interface
- Real-time Q&A with source citations
- Tailwind CSS for styling

## Configuration Notes

- File upload limit: 10MB (configured in `next.config.ts`)
- Supported file types: PDF and plain text
- Vector store is ephemeral - documents are lost on server restart