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
│   │   └── vectorstore/ # Vector store interface and implementations
│   │       ├── interface.go # VectorStore interface
│   │       └── memory/      # In-memory implementation
│   │           └── memory_store.go
│   ├── pkg/
│   │   ├── types/    # Data structures (REST response types)
│   │   └── utils/    # Utilities
│   ├── go.mod        # Uses openai-go v1.12.0 (official client)
│   ├── .env.example
│   └── .gitignore
├── frontend/         # Next.js React application
│   ├── src/
│   │   ├── app/      # Next.js pages and components
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

- **Vector Store Architecture** (`backend/internal/vectorstore/`)
  - **Interface**: `VectorStore` interface for pluggable implementations
  - **Memory Implementation**: In-memory storage with cosine similarity
  - **Future-Ready**: Easy to add Redis, Pinecone, or other vector stores

- **REST API Endpoints** (Proper HTTP status codes, no `success` field)
  - `POST /api/upload` - Processes and stores documents (PDF/text, max 10MB)
    - Success: HTTP 200 with `UploadResponse`
    - Error: HTTP 400/500 with `ErrorResponse`
  - `POST /api/query` - Performs RAG queries against uploaded documents
    - Success: HTTP 200 with `QueryResponse`  
    - Error: HTTP 400/500 with `ErrorResponse`
  - `GET /health` - Health check endpoint

- **Key Components**
  - `RAGPipeline` - Core RAG logic using VectorStore interface
  - `DocumentProcessor` - PDF and text file processing utilities
  - `VectorStore` interface with `MemoryVectorStore` implementation
  - **Official OpenAI Client**: Uses `github.com/openai/openai-go` v1.12.0
  - Type definitions in `backend/pkg/types/models.go` (REST-compliant)

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
2. **Question Answering**: Frontend sends question → Backend `/api/query` → `VectorStore.Search()` → context retrieval → LLM prompt → response → Frontend displays clean answer

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
- `backend/internal/vectorstore/interface.go` - VectorStore interface definition
- `backend/internal/vectorstore/memory/memory_store.go` - In-memory implementation
- `backend/pkg/types/models.go` - Backend type definitions (REST-compliant, no `success` fields)
- `backend/cmd/main.go` - Application entry point

### Frontend
- `frontend/src/app/page.tsx` - Main application interface with REST API integration
- `frontend/src/types/index.ts` - Frontend type definitions (REST-compliant)
- Frontend environment configuration for backend URL

### Cross-cutting
- Environment variable configuration in both projects
- Type consistency between frontend and backend (both REST-compliant)
- **REST API Standards**: Use HTTP status codes, `response.ok` pattern, no `success` fields
- **Error Handling**: Structured `ErrorResponse` with error codes and details

## Git conventions

- Always use conventional commits (e.g., `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`).
