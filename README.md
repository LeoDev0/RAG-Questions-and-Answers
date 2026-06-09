# RAG Q&A

Document Q&A Bot that implements Retrieval-Augmented Generation (RAG). Users can upload PDF or text documents and ask questions about their content.

## Features

- Upload PDF and text documents (max 10MB)
- Ask natural language questions about uploaded content
- Real-time Q&A with source citations and per-answer confidence scoring
- PDF page attribution: answers track which page each retrieved chunk came from
- Multi-turn conversations: follow-up questions use prior chat history for both LLM context and retrieval disambiguation
- Streaming responses (Server-Sent Events) with a UI toggle to fall back to single-shot replies
- Vector-based document similarity search with low-relevance filtering and neighbor-chunk context expansion
- DeepSeek LLM integration for responses
- OpenAI embeddings for document processing

## Project Structure

```
/
├── backend/          # Go API server
│   ├── cmd/
│   │   └── main.go   # Application entry point
│   ├── internal/
│   │   ├── config/   # Configuration handling
│   │   ├── handlers/ # HTTP handlers
│   │   ├── services/ # Business logic (RAG pipeline, document processing)
│   │   └── repositories/
│   │       └── vectorstore/ # Vector store interface + in-memory implementation
│   ├── pkg/
│   │   ├── codes/      # API error codes
│   │   ├── similarity/ # Cosine similarity search
│   │   ├── types/      # Data structures
│   │   └── utils/      # Utilities
│   ├── go.mod
│   ├── Makefile
│   ├── .env.example
│   └── .gitignore
└── frontend/         # Next.js React application
    ├── src/
    │   ├── app/      # Next.js pages
    │   ├── lib/
    │   │   └── api/  # Query-mode dispatcher (streaming vs single-shot)
    │   └── types/
    ├── package.json
    ├── tsconfig.json
    ├── .env.example
    └── .gitignore
```

## Quick Start

### Backend Setup (Go API)

```bash
cd backend

# Install dependencies
go mod download

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
# DEEPSEEK_API_KEY=your_deepseek_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here
# PORT=3001

# Start development server
go run cmd/main.go
```

The backend will run on `http://localhost:3001`

### Frontend Setup (Next.js App)

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Edit .env.local with:
# NEXT_PUBLIC_BACKEND_URL=http://localhost:3001

# Start development server
npm run dev
```

The frontend will run on `http://localhost:3000`

## Testing

```bash
cd backend

# Run the full Go test suite
go test ./...

# Run the offline retrieval evaluation harness (no API keys required)
go test ./internal/services/ -run TestEvalRetrieval -v
```

The retrieval harness scores chunking/retrieval changes against a golden set
(`hit@1`, `recall@k`, `MRR`) using a deterministic local embedder, so it runs
fully offline and acts as a regression gate in CI. See `backend/CLAUDE.md` for
how to read the metrics and add golden cases.

## API Endpoints

- **POST** `/api/upload` - Upload and process documents
- **POST** `/api/query` - Ask questions about uploaded documents (single response)
- **POST** `/api/query/stream` - Same as `/api/query` but streams the answer via Server-Sent Events
- **GET** `/health` - Health check

## Environment Variables

### Backend (.env)
- `DEEPSEEK_API_KEY` - DeepSeek Chat API key for LLM responses
- `OPENAI_API_KEY` - OpenAI API key for document embeddings
- `PORT` - Server port (default: 3001)

### Frontend (.env)
- `NEXT_PUBLIC_BACKEND_URL` - Backend API URL (default: http://localhost:3001)

## Technology Stack

### Backend
- **Go 1.25.0** - Programming language
- **Gin** - Web framework
- **DeepSeek API** - Language model for responses
- **OpenAI Embeddings** - Document vectorization
- **ledongthuc/pdf** - PDF text extraction
- **In-memory Vector Store** - Document similarity search

### Frontend
- **Next.js** - React framework
- **React** - UI library
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling

## Architecture

### RAG Pipeline
1. **Document Upload**: Extracted text is normalized (header/footer stripping, de-hyphenation across page breaks), then chunked into 1000-character segments with 200-character overlap. PDF page numbers are attributed to each chunk.
2. **Embedding**: Text chunks are converted to vectors using OpenAI embeddings
3. **Storage**: Vectors stored in memory (ephemeral - resets on restart)
4. **Query**: User questions trigger similarity search to find relevant chunks. Recent user turns from the conversation history are folded into the embedding query to disambiguate follow-up references like "it" or "that". Low-similarity hits are filtered out and adjacent chunks are pulled in to give the LLM fuller context.
5. **Generation**: DeepSeek LLM generates responses based on retrieved context and prior chat history, along with a confidence score derived from the retrieval similarity scores

### Data Flow
```
Frontend → Go Backend /api/upload → Document Processing → Vector Storage
Frontend → Go Backend /api/query → Similarity Search → DeepSeek LLM → Response
```
