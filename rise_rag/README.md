# RISE RAG Chatbot

A Retrieval-Augmented Generation chatbot for the RISE (Revitalization Intelligence and Smart Empowerment) system, using ChromaDB for vector storage and Groq for AI-generated answers.

No local server required. No paid APIs. Runs entirely from the cloud using Groq's free tier.

## Architecture

```
rise_rag/
├── app/                    # Main package
│   ├── __init__.py
│   ├── config.py           # All configuration constants
│   ├── ingestion.py        # Parse & chunk knowledge base files
│   ├── embeddings.py       # ChromaDB vector store wrapper
│   ├── retriever.py        # Similarity search logic
│   ├── llm.py              # LLM interface (Groq + fallback)
│   └── chatbot.py          # Orchestrator — ties everything together
├── data/                   # Place your .txt knowledge base files here
├── scripts/
│   └── ingest.py           # One-time ingestion script
├── chat.py                 # CLI entry point
├── api.py                  # FastAPI server entry point
├── .env.example            # Template for your API key
├── requirements.txt
└── README.md
```

## Setup

### 1. Prerequisites

- Python 3.11+
- A free Groq API key (get one at https://console.groq.com — no credit card required)

### 2. Activate your venv and install dependencies

```powershell
# Windows (VS Code terminal)
.venvRise\Scripts\activate

pip install -r requirements.txt
```

### 3. Set your Groq API key

Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

Then open `.env` in VS Code and replace the placeholder with your real key:

```
GROQ_API_KEY=your_actual_key_here
```

Your `.env` file is already in `.gitignore` — it will never be pushed to GitHub.

### 4. Add your knowledge base files

Copy all 9 `.txt` files into the `data/` folder:
- `community_context.txt`
- `data_sources.txt`
- `glossary_and_scope.txt`
- `grants_detail.txt`
- `parcel_a_heritage.txt`
- `parcel_b_ix_hub.txt`
- `parcel_c_food_desert.txt`
- `qa_pairs.txt`
- `scoring_engine.txt`

### 5. Ingest the knowledge base (run once)

```powershell
python scripts/ingest.py
```

This parses all documents, creates chunks, generates embeddings with `sentence-transformers`, and stores them in a local ChromaDB database. Only needs to be run once unless you change the knowledge base files.

### 6. Start chatting (CLI)

```powershell
python chat.py
```

### 7. Start the API server

```powershell
uvicorn api:app --reload --port 8000
```

Then open http://localhost:8000/docs for the interactive API docs.

## Configuration

Edit `app/config.py` to change:
- `GROQ_MODEL` — which model to use (default: `llama-3.3-70b-versatile`)
- `GROQ_TEMPERATURE` — how creative vs factual the answers are (default: `0.1`)
- `TOP_K_RESULTS` — how many chunks to retrieve per query (default: `5`)
- `CHROMA_PERSIST_DIR` — where ChromaDB stores its data
- `EMBEDDING_MODEL` — which sentence-transformers model to use

## How it works

1. **Ingestion** (`scripts/ingest.py`): Reads each `.txt` file, splits on `---` separators, extracts metadata (PARCEL_ID, TOPIC, document title), and stores chunks in ChromaDB with semantic embeddings.

2. **Retrieval** (`app/retriever.py`): For each user query, the top-K most semantically similar chunks are retrieved from ChromaDB.

3. **Generation** (`app/llm.py`): Retrieved chunks are assembled into a context-grounded prompt and sent to Groq. The model is instructed to answer only from the provided context.

4. **Fallback**: If the Groq API key is missing or unavailable, the chatbot falls back to a context-only mode that surfaces the raw retrieved chunks with a clear message.

## Free tier note

- **Embeddings**: `all-MiniLM-L6-v2` via sentence-transformers — runs locally, downloaded once from HuggingFace, no API key needed.
- **LLM**: Groq free tier — 14,400 requests/day, 30 requests/minute, no credit card required.
- **Vector DB**: ChromaDB — runs locally, no cost.
