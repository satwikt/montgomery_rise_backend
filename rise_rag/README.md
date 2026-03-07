# RISE RAG Chatbot

A Retrieval-Augmented Generation chatbot for the RISE (Revitalization Intelligence and Smart Empowerment) system, using ChromaDB for vector storage and free local LLMs via Ollama.

## Architecture

```
rise_rag/
├── rise_rag/               # Main package
│   ├── __init__.py
│   ├── config.py           # All configuration constants
│   ├── ingestion.py        # Parse & chunk knowledge base files
│   ├── embeddings.py       # ChromaDB vector store wrapper
│   ├── retriever.py        # Similarity search logic
│   ├── llm.py              # LLM interface (Ollama + fallback)
│   └── chatbot.py          # Orchestrator — ties everything together
├── data/                   # Place your .txt knowledge base files here
├── scripts/
│   └── ingest.py           # One-time ingestion script
├── chat.py                 # Entry point — run this to chat
├── requirements.txt
└── README.md
```

## Setup

### 1. Prerequisites

- Python 3.11+
- Your virtual environment (`venvRise`) already created
- [Ollama](https://ollama.com) installed on your machine

### 2. Install Ollama and pull a model

```bash
# Install Ollama from https://ollama.com/download
# Then pull a lightweight model (choose one):

ollama pull mistral          # ~4GB  — best quality, recommended
ollama pull llama3.2         # ~2GB  — fast, good quality
ollama pull phi3             # ~2GB  — very fast, lightweight
ollama pull gemma2:2b        # ~1.6GB — smallest, still capable
```

### 3. Activate your venv and install dependencies

```bash
# In VS Code terminal:
venvRise\Scripts\activate        # Windows
# or
source venvRise/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

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

```bash
python scripts/ingest.py
```

This parses all documents, creates chunks, generates embeddings with `sentence-transformers`, and stores them in a local ChromaDB database.

### 6. Start chatting

```bash
python chat.py
```

## Configuration

Edit `rise_rag/config.py` to change:
- `OLLAMA_MODEL` — which Ollama model to use
- `OLLAMA_BASE_URL` — if Ollama runs on a different port
- `TOP_K_RESULTS` — how many chunks to retrieve per query
- `CHROMA_PERSIST_DIR` — where ChromaDB stores its data
- `EMBEDDING_MODEL` — which sentence-transformers model to use

## How it works

1. **Ingestion** (`scripts/ingest.py`): Reads each `.txt` file, splits on the `---` document separators, extracts metadata (PARCEL_ID, TOPIC, document title), and stores chunks in ChromaDB with semantic embeddings.

2. **Retrieval** (`rise_rag/retriever.py`): For each user query, the top-K most semantically similar chunks are retrieved from ChromaDB.

3. **Generation** (`rise_rag/llm.py`): Retrieved chunks are assembled into a context-grounded prompt and sent to Ollama. The LLM is instructed to answer only from the provided context.

4. **Fallback**: If Ollama is not running, the chatbot falls back to a context-only mode that surfaces the raw retrieved chunks with a clear message.

## Free LLM note

All LLMs run locally via Ollama — no API keys, no cloud costs, no rate limits. The embedding model (`all-MiniLM-L6-v2`) is downloaded once from HuggingFace and cached locally.
