from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import ollama

from app.utils.chunker import chunk_text

app = FastAPI()

# Load the embedding model once at startup (downloads on first run ~90MB)
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")

# ChromaDB client — persists data to ./chroma_store inside the project
chroma_client = chromadb.PersistentClient(path="./chroma_store")

# Get or create a collection (like a table) to store our embeddings
collection = chroma_client.get_or_create_collection(name="rag_documents")

# Ollama model to use for generating answers
LLM_MODEL = "llama3.2"


class SaveRequest(BaseModel):
    text: str
    source: str = "unknown"     # optional: tag where the text came from (e.g. filename)
    chunk_size: int = 500       # max words per chunk
    overlap: int = 50           # words overlapping between chunks


class ChunkEmbedding(BaseModel):
    chunk_index: int
    text: str
    embedding: list[float]
    chroma_id: str


@app.post("/save", response_model=dict)
def save_embedding(req: SaveRequest):
    # Step 1: Chunk the text
    chunks = chunk_text(req.text, chunk_size=req.chunk_size, overlap=req.overlap)

    # Step 2: Embed each chunk
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    # Step 3: Generate unique IDs for each chunk
    ids = [str(uuid.uuid4()) for _ in chunks]

    # Step 4: Store in ChromaDB
    # - documents: original text of each chunk (for retrieval later)
    # - embeddings: the vectors (for similarity search)
    # - metadatas: extra info like source and chunk index
    # - ids: unique identifier per chunk
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": req.source, "chunk_index": i} for i in range(len(chunks))],
        ids=ids,
    )

    # Step 5: Build response
    result = [
        ChunkEmbedding(
            chunk_index=i,
            text=chunk,
            embedding=embedding,
            chroma_id=chroma_id,
        )
        for i, (chunk, embedding, chroma_id) in enumerate(zip(chunks, embeddings, ids))
    ]

    return {
        "status": "ok",
        "total_chunks": len(chunks),
        "source": req.source,
        "data": [r.model_dump() for r in result],
    }


class SearchRequest(BaseModel):
    question: str
    top_k: int = 5              # how many similar chunks to retrieve from ChromaDB
    llm_model: str = LLM_MODEL  # which Ollama model to use


@app.post("/search", response_model=dict)
def search(req: SearchRequest):
    # Step 1: Embed the user's question using the same model
    question_embedding = model.encode(req.question, show_progress_bar=False).tolist()

    # Step 2: Query ChromaDB — find top_k most similar chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=req.top_k,
        include=["documents", "metadatas", "distances"],
    )

    # results["documents"][0] → list of matching chunk texts
    # results["metadatas"][0] → list of metadata dicts for each chunk
    # results["distances"][0] → similarity distances (lower = more similar)
    matched_chunks = results["documents"][0]
    matched_metadata = results["metadatas"][0]
    matched_distances = results["distances"][0]

    # Step 3: Build context from retrieved chunks
    context = "\n\n".join(
        f"[Chunk {i+1} | source: {meta.get('source', 'unknown')}]\n{chunk}"
        for i, (chunk, meta) in enumerate(zip(matched_chunks, matched_metadata))
    )

    # Step 4: Build the RAG prompt
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {req.question}

Answer:"""

    # Step 5: Send prompt to Ollama LLM and get the answer
    response = ollama.chat(
        model=req.llm_model,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.message.content

    return {
        "status": "ok",
        "question": req.question,
        "answer": answer,
        "sources": [
            {
                "chunk_index": meta.get("chunk_index"),
                "source": meta.get("source", "unknown"),
                "text": chunk,
                "distance": round(distance, 4),
            }
            for chunk, meta, distance in zip(matched_chunks, matched_metadata, matched_distances)
        ],
    }
