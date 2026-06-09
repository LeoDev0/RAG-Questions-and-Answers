# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Document Q&A Bot** with decoupled architecture implementing Retrieval-Augmented Generation (RAG). The project consists of two separate applications:

- **Backend**: Go API server (in `/backend` directory)
- **Frontend**: Next.js React application (in `/frontend` directory)

Users can upload PDF or text documents and ask questions about their content through a web interface.

## Development Commands

### Backend (Go API)
```bash
cd backend

# Install dependencies
go mod download

# Run the server
go run cmd/main.go

# Build for production
go build -o bin/server cmd/main.go

# Run tests
go test ./...

# Format code
gofmt -w .

# Vet code
go vet ./...

# Lint (golangci-lint, pinned version auto-installed on first run)
make lint

# Lint with autofixes where supported
make lint-fix
```

### Frontend (Next.js App)
```bash
cd frontend

# Install dependencies
npm install

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

### Backend Environment
Copy `backend/.env.example` to `backend/.env` and configure:
- `DEEPSEEK_API_KEY` - For LLM responses (DeepSeek Chat API)
- `OPENAI_API_KEY` - For document embeddings only
- `PORT` - Backend server port (default: 3001)

### Frontend Environment
Copy `frontend/.env.example` to `frontend/.env.local` and configure:
- `NEXT_PUBLIC_BACKEND_URL` - Backend API URL (default: http://localhost:3001)

## Architecture

This project follows a **decoupled architecture** with separate backend and frontend applications:

```
/
├── backend/          # Go API server
│   ├── cmd/
│   │   └── main.go   # Application entry point
│   ├── internal/
│   │   ├── config/   # Configuration handling
│   │   ├── handlers/ # HTTP handlers (REST API with proper status codes)
│   │   ├── services/ # Business logic (RAG pipeline, document processing)
│   │   └── repositories/
│   │       └── vectorstore/ # Vector store interface and implementations
│   │           ├── interface.go # VectorStore interface
│   │           └── memory/      # In-memory implementation
│   │               └── memory_store.go
│   ├── pkg/
│   │   ├── codes/      # API error codes
│   │   ├── similarity/ # Cosine similarity
│   │   ├── types/      # Data structures (REST response types)
│   │   └── utils/      # Text splitting and normalization
│   ├── go.mod        # Uses openai-go v1.12.0 (official client)
│   ├── Makefile      # lint / lint-fix / test / fmt / vet targets
│   ├── .golangci.yml
│   ├── .env.example
│   └── .gitignore
├── frontend/         # Next.js React application
│   ├── src/
│   │   ├── app/      # Next.js pages and components
│   │   ├── lib/
│   │   │   └── api/  # Query-mode dispatcher (streaming vs single-shot)
│   │   └── types/    # Frontend type definitions (REST-compliant)
│   ├── package.json
│   ├── tsconfig.json
│   ├── .env.example
│   └── .gitignore
├── README.md         # Project documentation
└── CLAUDE.md         # This file
```

### Backend Components (`/backend`)
- **Core RAG Pipeline** (`backend/internal/services/rag_pipeline.go`)
  - **LLM**: DeepSeek Chat API via official OpenAI client (v1.12.0)
  - **Embeddings**: OpenAI embeddings for vector similarity search
  - **Vector Store**: Interface-based design with in-memory implementation
  - **Text Splitting**: 1000 character chunks with 200 character overlap
  - **Text Normalization**: Extracted text is normalized before chunking (`pkg/utils/text_normalizer.go`) — page-aware header/footer stripping and de-hyphenation across page breaks
  - **Page Attribution**: PDF page numbers are tracked via `PageSpan`s and attached to each chunk (`DocumentChunk.Page` + `page` metadata)
  - **Relevance Filtering & Confidence**: Search hits below `similarityThreshold` (0.3) are dropped so weak matches don't dilute the prompt; the response `Confidence` is derived from the retained scores
  - **Neighbor Context Expansion**: `neighborRadius` adjacent chunks are pulled in around each search hit (bounded by `maxContextChars`) to give the LLM fuller surrounding context than the matched fragment alone
  - **Conversation History**: Optional `history` is trimmed to the last `maxHistoryTurns` turns before being passed as prior chat messages to the LLM. A separate, smaller `retrievalRewriteWindow` folds only the most recent user turns into the embedding query — keeping retrieval focused on the current topic while still resolving follow-up references like "it" or "that".

- **Vector Store Architecture** (`backend/internal/repositories/vectorstore/`)
  - **Interface**: `VectorStore` interface (`Store`, `Search`, `Neighbors`) for pluggable implementations
  - **Memory Implementation**: In-memory storage with cosine similarity (`pkg/similarity`)
  - **Future-Ready**: Easy to add Redis, Pinecone, or other vector stores

- **REST API Endpoints** (Proper HTTP status codes, no `success` field)
  - `POST /api/upload` - Processes and stores documents (PDF/text, max 10MB)
    - Success: HTTP 200 with `UploadResponse`
    - Error: HTTP 400/500 with `ErrorResponse`
  - `POST /api/query` - Performs RAG queries against uploaded documents (single response)
    - Request body: `QueryRequest` with required `question` and optional `history` (array of `{role: "user"|"assistant", content: string}`)
    - Success: HTTP 200 with `QueryResponse` (`answer`, `sources`, `confidence`)
    - Error: HTTP 400/500 with `ErrorResponse`
  - `POST /api/query/stream` - Same as `/api/query` but streams the answer via Server-Sent Events
    - Request body: same `QueryRequest` shape as `/api/query` (including optional `history`)
    - Success: HTTP 200 with `text/event-stream`; emits `sources` (carries `confidence`), `token`, `done`, and `error` events (each as `data: {...}\n\n`)
    - Error: HTTP 400/500 with `ErrorResponse` (before the stream begins) or an inline `error` SSE event
  - `GET /health` - Health check endpoint

- **Key Components**
  - `RAGPipeline` - Core RAG logic using VectorStore interface
  - `DocumentProcessor` - PDF and text file processing utilities
  - `VectorStore` interface with `MemoryVectorStore` implementation
  - **Official OpenAI Client**: Uses `github.com/openai/openai-go` v1.12.0
  - Type definitions in `backend/pkg/types/models.go` (REST-compliant)
  - API error codes in `backend/pkg/codes/errors.go`
  - Cosine similarity in `backend/pkg/similarity`, text utilities in `backend/pkg/utils`

### Frontend Components (`/frontend`)
- **Next.js Application** (`frontend/src/app/`)
  - **Document Upload Interface** - File upload with validation
  - **Chat Interface** - Real-time Q&A with message history and animated "thinking..." 
  - **Clean UI** - No technical source citations shown to users
  - **Responsive Design** - Tailwind CSS styling

- **Key Features**
  - File upload with drag-and-drop support
  - Real-time chat interface for questions
  - **REST API Integration** - Uses `response.ok` instead of `success` field
  - **Proper Error Handling** - HTTP status code based error handling
  - **Animated Loading State** - "Thinking..." with animated dots
  - Type definitions in `frontend/src/types/index.ts` (REST-compliant)

### Data Flow
1. **Document Upload**: Frontend uploads files → Backend `/api/upload` → `DocumentProcessor` → chunked → embedded → stored in `VectorStore` interface
2. **Question Answering**: Frontend sends question + prior chat `history` (capped client-side by `buildHistory`) → Backend `/api/query` (single response) or `/api/query/stream` (SSE) → history is trimmed to `maxHistoryTurns` and the last `retrievalRewriteWindow` user turns are folded into the embedding query → `VectorStore.Search()` → low-similarity hits filtered out and neighbor chunks expanded for context → LLM prompt (system prompt + prior history + current question) → response (with `confidence` derived from retrieval scores) → Frontend displays clean answer (rendered all at once or incrementally as tokens stream in, selectable via the response-mode toggle in the UI)

## Configuration Notes

- **File upload limit**: 10MB (configured in Go Gin middleware)
- **Supported file types**: PDF and plain text
- **Vector store**: Ephemeral - documents are lost on server restart
- **Ports**: Backend runs on port 3001, Frontend on port 3000
- **CORS**: Backend configured to allow requests from frontend
- **API Configuration**: DeepSeek API accessed via OpenAI-compatible client
- **Communication**: Frontend communicates with backend via proper HTTP REST API calls (no `success` fields)
- **OpenAI Client**: Uses official `github.com/openai/openai-go` v1.12.0 for both embeddings and DeepSeek

## Development Workflow

1. **Setup**: Install dependencies in both `backend/` and `frontend/` directories
2. **Environment**: Configure environment variables in both directories
3. **Development**: 
   - Start backend first: `cd backend && go run cmd/main.go`
   - Start frontend: `cd frontend && npm run dev`
4. **Testing**: Upload documents via frontend, test Q&A functionality
5. **Code Quality**: Run `gofmt -w . && go vet ./...` in backend, `npm run lint` in frontend

## Important Files to Check

When working on this project, pay attention to:

### Backend
- `backend/CLAUDE.md` - Backend-specific conventions (testing, code style)
- `backend/internal/services/rag_pipeline.go` - Core RAG logic with VectorStore interface
- `backend/internal/services/document_processor.go` - File processing
- `backend/internal/handlers/` - REST API endpoint implementations (proper HTTP status codes)
- `backend/internal/repositories/vectorstore/interface.go` - VectorStore interface definition
- `backend/internal/repositories/vectorstore/memory/memory_store.go` - In-memory implementation
- `backend/pkg/types/models.go` - Backend type definitions (REST-compliant, no `success` fields)
- `backend/pkg/codes/errors.go` - API error code constants
- `backend/pkg/similarity/cosine.go` - Cosine similarity used by the memory store
- `backend/pkg/utils/text_splitter.go` - Chunking; `backend/pkg/utils/text_normalizer.go` - pre-chunk normalization
- `backend/cmd/main.go` - Application entry point

### Frontend
- `frontend/src/app/page.tsx` - Main application interface with REST API integration
- `frontend/src/lib/api/query.ts` - `sendQuery(mode, question, history, backendUrl, callbacks)` dispatcher and `QueryCallbacks` contract; pick `'stream'` or `'single'`
- `frontend/src/lib/api/queryStream.ts` - SSE parser for `/api/query/stream` (sends `question` + `history`)
- `frontend/src/lib/api/queryOnce.ts` - Single-shot client for `/api/query` (sends `question` + `history`)
- `frontend/src/lib/api/history.ts` - `buildHistory` helper that filters empty messages and caps payload to `MAX_TRANSPORTED_TURNS`; independent of the backend's own history cap
- `frontend/src/types/index.ts` - Frontend type definitions (REST-compliant)
- Frontend environment configuration for backend URL

### Cross-cutting
- Environment variable configuration in both projects
- Type consistency between frontend and backend (both REST-compliant)
- **REST API Standards**: Use HTTP status codes, `response.ok` pattern, no `success` fields
- **Error Handling**: Structured `ErrorResponse` with error codes and details

## Git conventions

- Always use conventional commits (e.g., `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`).
