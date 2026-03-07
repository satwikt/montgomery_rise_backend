"""
RISE RAG Knowledge Base Loader
================================
Loads all knowledge base documents into ChromaDB with structured
chunking and metadata for retrieval-augmented generation.

Usage:
    pip install chromadb sentence-transformers
    python rag_loader.py

    # On first run: builds the vector store (~30 seconds)
    # On subsequent runs: loads existing store instantly

Directory structure expected:
    rag_kb/
        parcel_a_heritage.txt
        parcel_b_ix_hub.txt
        parcel_c_food_desert.txt
        scoring_engine.txt
        community_context.txt
        grants_detail.txt
        data_sources.txt
        qa_pairs.txt
        glossary_and_scope.txt

Output:
    rise_chroma_db/   ← persistent ChromaDB vector store
"""

import os
import re
import chromadb
from chromadb.utils import embedding_functions

# ── CONFIG ──────────────────────────────────────────────────────────
KB_DIR       = "rag_kb"
CHROMA_DIR   = "rise_chroma_db"
COLLECTION   = "rise_knowledge"

# Sentence-transformers model — free, runs locally, no API key
# all-MiniLM-L6-v2: fast, small, good for semantic similarity
# all-mpnet-base-v2: slower, larger, better accuracy (recommended for prod)
EMBED_MODEL  = "all-MiniLM-L6-v2"

# ── DOCUMENT MANIFEST ───────────────────────────────────────────────
# Maps filename → metadata applied to ALL chunks from that file
DOCUMENT_MANIFEST = {
    "parcel_a_heritage.txt": {
        "parcel":   "A",
        "doc_type": "parcel_sheet",
        "story":    "heritage",
        "priority": "high",   # high = boosted in retrieval for parcel-specific Qs
    },
    "parcel_b_ix_hub.txt": {
        "parcel":   "B",
        "doc_type": "parcel_sheet",
        "story":    "ix_hub",
        "priority": "high",
    },
    "parcel_c_food_desert.txt": {
        "parcel":   "C",
        "doc_type": "parcel_sheet",
        "story":    "food_desert",
        "priority": "high",
    },
    "scoring_engine.txt": {
        "parcel":   "general",
        "doc_type": "scoring",
        "story":    "general",
        "priority": "high",
    },
    "community_context.txt": {
        "parcel":   "general",
        "doc_type": "context",
        "story":    "general",
        "priority": "medium",
    },
    "grants_detail.txt": {
        "parcel":   "general",
        "doc_type": "grants",
        "story":    "general",
        "priority": "high",   # grant questions are common in demos
    },
    "data_sources.txt": {
        "parcel":   "general",
        "doc_type": "technical",
        "story":    "general",
        "priority": "medium",
    },
    "qa_pairs.txt": {
        "parcel":   "general",
        "doc_type": "qa",
        "story":    "general",
        "priority": "high",   # direct Q&A pairs = highest retrieval value
    },
    "glossary_and_scope.txt": {
        "parcel":   "general",
        "doc_type": "reference",
        "story":    "general",
        "priority": "low",
    },
}


# ── CHUNKING ─────────────────────────────────────────────────────────
def parse_chunks(filepath: str, file_meta: dict) -> list[dict]:
    """
    Parse a knowledge base .txt file into chunks.

    Chunking strategy:
    - Split on '---' separators (each section is a named chunk)
    - Each chunk keeps its DOCUMENT header as context
    - Minimum chunk length: 50 chars (skip empty sections)
    - Metadata extracted from DOCUMENT header line:
        PARCEL_ID: X
        TOPIC: scoring_heritage

    Returns list of dicts:
        { "id": str, "text": str, "metadata": dict }
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split on '---' separator lines
    sections = re.split(r"\n---\n", raw)

    chunks = []
    current_doc_header = ""
    current_parcel_id  = file_meta.get("parcel", "general")
    current_topic      = "general"

    for i, section in enumerate(sections):
        section = section.strip()
        if not section or len(section) < 50:
            continue

        # Extract DOCUMENT header metadata if present
        doc_match = re.search(r"DOCUMENT:\s*(.+)", section)
        pid_match = re.search(r"PARCEL_ID:\s*(\S+)", section)
        top_match = re.search(r"TOPIC:\s*(\S+)", section)

        if doc_match:
            current_doc_header = doc_match.group(1).strip()
        if pid_match:
            current_parcel_id = pid_match.group(1).strip()
        if top_match:
            current_topic = top_match.group(1).strip()

        # Clean the text: remove the header metadata lines
        # but keep the document title as context for the chunk
        text_lines = []
        for line in section.split("\n"):
            if line.startswith("DOCUMENT:") or line.startswith("PARCEL_ID:") or line.startswith("TOPIC:"):
                continue
            text_lines.append(line)
        clean_text = "\n".join(text_lines).strip()

        if not clean_text or len(clean_text) < 50:
            continue

        # Prepend document title so chunk is self-contained for retrieval
        if current_doc_header:
            chunk_text = f"[{current_doc_header}]\n\n{clean_text}"
        else:
            chunk_text = clean_text

        # Build chunk metadata — merge file-level meta with chunk-level
        chunk_meta = {
            **file_meta,
            "parcel":       current_parcel_id,
            "topic":        current_topic,
            "doc_title":    current_doc_header,
            "source_file":  os.path.basename(filepath),
            "chunk_index":  i,
        }

        # Stable chunk ID from file + section index
        chunk_id = f"{os.path.basename(filepath).replace('.txt','')}_{i:03d}"

        chunks.append({
            "id":       chunk_id,
            "text":     chunk_text,
            "metadata": chunk_meta,
        })

    return chunks


# ── LOAD INTO CHROMADB ───────────────────────────────────────────────
def build_knowledge_base(force_rebuild: bool = False) -> chromadb.Collection:
    """
    Build (or load) the ChromaDB vector store from the knowledge base files.

    force_rebuild=True: wipes existing store and rebuilds from scratch.
    force_rebuild=False: loads existing store if available.
    """
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Check if collection already exists
    existing = [c.name for c in client.list_collections()]

    if COLLECTION in existing and not force_rebuild:
        print(f"[RAG] Loading existing collection '{COLLECTION}' from {CHROMA_DIR}")
        collection = client.get_collection(
            name=COLLECTION,
            embedding_function=embed_fn,
        )
        print(f"[RAG] ✅ Collection loaded — {collection.count()} chunks ready")
        return collection

    # Build from scratch
    if COLLECTION in existing:
        print(f"[RAG] Rebuilding — deleting existing collection '{COLLECTION}'")
        client.delete_collection(COLLECTION)

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for semantic search
    )

    print(f"[RAG] Building knowledge base from {KB_DIR}/")
    total_chunks = 0

    for filename, file_meta in DOCUMENT_MANIFEST.items():
        filepath = os.path.join(KB_DIR, filename)

        if not os.path.exists(filepath):
            print(f"  ⚠️  File not found: {filepath} — skipping")
            continue

        chunks = parse_chunks(filepath, file_meta)

        if not chunks:
            print(f"  ⚠️  No chunks parsed from {filename}")
            continue

        # Add to ChromaDB in one batch per file
        collection.add(
            ids        = [c["id"]       for c in chunks],
            documents  = [c["text"]     for c in chunks],
            metadatas  = [c["metadata"] for c in chunks],
        )

        total_chunks += len(chunks)
        print(f"  ✅ {filename:<35s} → {len(chunks)} chunks")

    print(f"\n[RAG] ✅ Knowledge base built — {total_chunks} total chunks")
    print(f"[RAG]    Stored at: {CHROMA_DIR}/")
    return collection


# ── QUERY FUNCTION ───────────────────────────────────────────────────
def query_knowledge_base(
    collection:   chromadb.Collection,
    question:     str,
    n_results:    int = 5,
    parcel_filter: str | None = None,
) -> list[dict]:
    """
    Retrieve the most relevant chunks for a question.

    Args:
        collection:     ChromaDB collection (from build_knowledge_base())
        question:       The user's question string
        n_results:      Number of chunks to retrieve (default 5)
        parcel_filter:  Optional — filter to a specific parcel ("A", "B", "C")
                        If None, searches all chunks including general docs

    Returns:
        List of dicts: { "text": str, "metadata": dict, "distance": float }
        Sorted by relevance (lowest distance = most similar)
    """
    where_clause = None
    if parcel_filter:
        # Return chunks for the specific parcel AND general docs
        where_clause = {
            "$or": [
                {"parcel": {"$eq": parcel_filter}},
                {"parcel": {"$eq": "general"}},
            ]
        }

    results = collection.query(
        query_texts  = [question],
        n_results    = n_results,
        where        = where_clause,
        include      = ["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":     doc,
            "metadata": meta,
            "distance": round(dist, 4),
        })

    return chunks


# ── GEMINI RAG RESPONSE ──────────────────────────────────────────────
def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build the prompt sent to Gemini with retrieved context.
    The model is instructed to answer from context only — no hallucination.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        meta  = chunk["metadata"]
        label = f"[Source {i} — {meta.get('doc_title', meta.get('topic', 'general'))}]"
        context_blocks.append(f"{label}\n{chunk['text']}")

    context_str = "\n\n---\n\n".join(context_blocks)

    return f"""You are the RISE assistant — an AI advisor for Montgomery Alabama's Revitalization Intelligence and Smart Empowerment tool.

Answer the user's question using ONLY the context provided below. 

Rules:
- If the answer is in the context, answer confidently and specifically — cite numbers, parcel names, and grant details exactly as they appear.
- If the answer is partially in the context, answer what you can and clearly note what you do not have data for.
- If the answer is not in the context at all, say: "I don't have that specific information in my knowledge base. I can tell you about the three RISE hero parcels, the scoring model, active grants, and the Montgomery community context."
- Never invent numbers, dates, grant names, or statistics.
- Keep answers concise — 3 to 6 sentences unless a detailed breakdown is asked for.
- Use plain English. Avoid jargon unless the user asked a technical question.

CONTEXT:
{context_str}

USER QUESTION:
{question}

ANSWER:"""


# ── MAIN — test the pipeline ─────────────────────────────────────────
if __name__ == "__main__":
    import requests
    import json
    import os

    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_KEY")
    GEMINI_URL = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-pro:generateContent"
    )

    # ── Step 1: Build or load the vector store ───────────────────────
    collection = build_knowledge_base(force_rebuild=False)

    # ── Step 2: Run test questions ───────────────────────────────────
    TEST_QUESTIONS = [
        ("What is the most urgent grant right now?",                   None),
        ("Why does Parcel A score high on heritage?",                  "A"),
        ("How much would the city pay for a grocery on Parcel C?",     "C"),
        ("Where does the foot traffic data come from?",               None),
        ("What does the 311 distress score measure?",                 None),
        ("What is MGMix and why does it matter?",                     None),
        ("Can RISE scale to all 200 Montgomery parcels?",             None),
    ]

    print("\n" + "=" * 65)
    print("  RISE RAG — Test Query Results")
    print("=" * 65)

    for question, parcel in TEST_QUESTIONS:
        print(f"\nQ: {question}")
        if parcel:
            print(f"   (filtered to Parcel {parcel} + general docs)")

        # Retrieve relevant chunks
        chunks = query_knowledge_base(collection, question, n_results=4, parcel_filter=parcel)

        print(f"   Retrieved {len(chunks)} chunks:")
        for c in chunks:
            print(f"     [{c['distance']:.3f}] {c['metadata'].get('doc_title', c['metadata'].get('topic'))}")

        # Generate answer via Gemini (only if key is set)
        if GEMINI_API_KEY != "YOUR_GEMINI_KEY":
            prompt = build_rag_prompt(question, chunks)
            resp   = requests.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 300, "temperature": 0.2},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                answer = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                print(f"\n   A: {answer}\n")
            else:
                print(f"   [Gemini error {resp.status_code}]")
        else:
            print("   [Set GEMINI_API_KEY to see AI answers]")

        print("─" * 65)
