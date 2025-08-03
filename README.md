# RAG Q&A

Document Q&A Bot that implements Retrieval-Augmented Generation (RAG). Users can upload PDF or text documents and ask questions about their content.

## Features

- Upload PDF and text documents (max 10MB)
- Ask natural language questions about uploaded content
- Real-time Q&A with source citations
- Vector-based document similarity search
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
│   │   └── services/ # Business logic (RAG pipeline, document processing)
│   ├── pkg/
│   │   ├── types/      # Data structures
│   │   └── utils/      # Utilities
│   │   └── similarity/ # Similarity search algorithm
│   ├── go.mod
│   ├── Makefile
│   ├── .env.example
│   └── .gitignore
└── frontend/         # Next.js React application
    ├── src/
    │   ├── app/      # Next.js pages
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

## API Endpoints

- **POST** `/api/upload` - Upload and process documents
- **POST** `/api/query` - Ask questions about uploaded documents
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
- **Go 1.24.4** - Programming language
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
1. **Document Upload**: Files are processed and chunked into 1000-character segments with 200-character overlap
2. **Embedding**: Text chunks are converted to vectors using OpenAI embeddings
3. **Storage**: Vectors stored in memory (ephemeral - resets on restart)
4. **Query**: User questions trigger similarity search to find relevant chunks
5. **Generation**: DeepSeek LLM generates responses based on retrieved context

### Data Flow
```
Frontend → Go Backend /api/upload → Document Processing → Vector Storage
Frontend → Go Backend /api/query → Similarity Search → DeepSeek LLM → Response
```
