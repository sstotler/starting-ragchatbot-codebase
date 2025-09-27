# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Course Materials RAG (Retrieval-Augmented Generation) System** - a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses using semantic search and Anthropic's Claude AI.

## Development Commands

### Quick Start
```bash
chmod +x run.sh && ./run.sh
```

### Manual Development
```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Dependencies
```bash
# Install/update dependencies
uv sync

# Add new dependency
uv add package-name
```

## Architecture Overview

### Core System Flow
1. **Document Ingestion**: Course files → DocumentProcessor → ChromaDB vector storage
2. **Query Processing**: User query → RAGSystem → AI tools → Claude response
3. **Tool-Based AI**: Claude uses CourseSearchTool for semantic search rather than direct RAG
4. **Dual Vector Storage**: Separate collections for course metadata (`course_catalog`) and content (`course_content`)

### Key Components

**Backend (`/backend/`)**:
- `app.py` - FastAPI application and API endpoints
- `rag_system.py` - Main orchestrator coordinating all components
- `vector_store.py` - ChromaDB integration with dual collections
- `ai_generator.py` - Claude API integration with tool-based interaction
- `document_processor.py` - Document chunking and metadata extraction
- `search_tools.py` - AI tool system for semantic search
- `session_manager.py` - Conversation state and history management

**Frontend (`/frontend/`)**:
- Static HTML/CSS/JS single-page chat interface
- Served directly by FastAPI at root path

**Data (`/docs/`)**:
- Course materials auto-loaded on startup
- Supports TXT, PDF, DOCX formats

### Technology Stack
- **FastAPI** + **ChromaDB** + **Anthropic Claude** + **Sentence Transformers**
- **uv** package manager (Python 3.13 required)
- Static frontend with Marked.js for markdown rendering

### Configuration
Key environment variables:
- `ANTHROPIC_API_KEY` (required)
- `ANTHROPIC_MODEL: "claude-sonnet-4-20250514"`
- `EMBEDDING_MODEL: "all-MiniLM-L6-v2"`

### API Endpoints
- `POST /api/query` - Process user questions with optional session_id
- `GET /api/courses` - Get course statistics
- `GET /` - Serve frontend

### Session Management
- Stateful conversations with session IDs
- Conversation history limited to 2 exchanges
- Context preserved for follow-up queries

### Vector Search Architecture
- **Course Resolution**: Fuzzy matching for course names in queries
- **Semantic Chunking**: 800-character chunks with 100-character overlap
- **Metadata Filtering**: Search can be filtered by course or lesson
- **Source Attribution**: All responses include source file references

### Development Notes
- No test framework currently configured
- ChromaDB data persisted in `./chroma_db/`
- Frontend assets served from `/frontend/static/`
- CORS enabled for development