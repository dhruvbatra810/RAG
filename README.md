# RAG API with FastAPI, ChromaDB, and Ollama

A small retrieval-augmented generation (RAG) API. Upload text to the API, which splits it into chunks and stores their embeddings in ChromaDB. Ask a question later and the API retrieves the most relevant chunks before asking a local Ollama model to answer from that context only.

## How it works

```text
Text document
  -> /save
  -> chunk text + create embeddings (all-MiniLM-L6-v2)
  -> ChromaDB

Question
  -> /search
  -> retrieve relevant chunks from ChromaDB
  -> Ollama / Llama 3.2
  -> answer with sources
```

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally

## Run locally

1. Install project dependencies:

   ```bash
   uv sync
   ```

2. Download the LLM used to answer questions. The smaller 1B model is a good choice on a CPU-only laptop:

   ```bash
   ollama pull llama3.2:1b
   ```

3. In `app/routes/router.py`, update `LLM_MODEL` if necessary:

   ```python
   LLM_MODEL = "llama3.2:1b"
   ```

4. Start the API:

   ```bash
   uv run python main.py
   ```

The interactive API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

On first startup, the embedding model (`all-MiniLM-L6-v2`) is downloaded and cached in `models/`.

## API usage

### Add text to the knowledge base

```bash
curl -X POST http://localhost:8000/save \
  -H "Content-Type: application/json" \
  -d '{
    "source": "company-handbook",
    "text": "Employees receive 20 days of paid leave each year.",
    "chunk_size": 500,
    "overlap": 50
  }'
```

### Ask a question

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How much paid leave do employees receive?",
    "top_k": 5,
    "llm_model": "llama3.2:1b"
  }'
```

The `/search` response includes the answer and the retrieved source chunks, including their similarity distances.

## Data and models

- `models/` contains the downloaded embedding model cache.
- `chroma_store/` contains the persistent local ChromaDB knowledge base.
- Both are deliberately excluded from Git. Back up `chroma_store/` separately if its contents matter.

## Deployment notes

Ollama is a separate model service; it is not bundled into the Python API. In deployment, run it on the same server or as a private service on the same network. The API then calls Ollama over its internal API.

For a low-traffic, no-cost prototype, an Oracle Cloud Always Free VM can run FastAPI, ChromaDB, and a small Ollama model together. It will be CPU-based and comparatively slow. A paid GPU server or an external LLM API is the better choice for higher traffic or faster responses.

Do not expose Ollama's port (`11434`) directly to the internet. Expose only the FastAPI application through HTTPS, and add authentication and rate limits before accepting public traffic.
