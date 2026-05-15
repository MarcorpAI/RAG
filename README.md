---
title: RAG Document Q&A API
emoji: 📚
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# RAG-Powered Document Q&A API

A production-style Retrieval-Augmented Generation API for asking grounded questions over uploaded PDF or text documents. The backend is FastAPI, the vector search layer is FAISS, embeddings come from Hugging Face sentence-transformers, and answer generation uses Hugging Face Inference Providers.

The implementation intentionally avoids LangChain and LlamaIndex so the RAG pipeline is visible end to end.

## What It Does

- Accepts `.pdf` and `.txt` uploads through `POST /ingest`.
- Extracts text, chunks it with overlap, and embeds each chunk.
- Stores one local FAISS index plus metadata per uploaded document.
- Retrieves the most relevant chunks for a question.
- Sends only those chunks as context to a Hugging Face chat model.
- Returns the answer plus source chunks and similarity scores.
- Serves a minimal browser UI from the FastAPI app.

## Architecture

```text
Browser UI / cURL
  -> FastAPI routes
    -> document loader
    -> chunker
    -> sentence-transformer embedder
    -> FAISS document store
    -> Hugging Face router chat completions
```

Each ingested document gets a local directory under `DATA_DIR` containing:

- `index.faiss`
- `metadata.json`

Restarting the API reloads existing document indexes from `DATA_DIR`.

## Local Setup

Use Python 3.11. The requirements pin CPU PyTorch wheels to avoid pulling GPU/CUDA packages.

```bash
cd /path/to/RAG
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set your Hugging Face token:

```env
HF_API_KEY=your_huggingface_token
```

Your token needs permission to make Inference Provider calls.

Default models:

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Generation: `meta-llama/Llama-3.2-1B-Instruct`

## Run Locally With FastAPI

```bash
cd /path/to/RAG
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8011 --reload
```

Open:

- Frontend: `http://127.0.0.1:8011/`
- API docs: `http://127.0.0.1:8011/docs`
- Health check: `http://127.0.0.1:8011/health`

If you open `frontend/index.html` directly from disk, pass the API URL:

```text
file:///path/to/RAG/frontend/index.html?api=http://127.0.0.1:8011
```

## Demo Flow

Use a short document with facts that are easy to verify, such as a project brief, policy note, or article excerpt.

1. Start the API and open the frontend.
2. Upload a `.txt` or `.pdf` file.
3. Copy the returned document ID if you want to test from cURL.
4. Ask a question that is answered directly in the document.
5. Ask a question that is not in the document and confirm the assistant refuses to invent an answer.
6. Expand the source chunks in the response and compare them with the generated answer.

The key behavior to show is that the answer is grounded in retrieved chunks, not in the model's general memory.

## API Usage

Ingest a document:

```bash
curl -X POST http://127.0.0.1:8011/ingest \
  -F "file=@sample.txt"
```

Query a document:

```bash
curl -X POST http://127.0.0.1:8011/query \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "returned-doc-id",
    "question": "What are the key findings?",
    "top_k": 5
  }'
```

Check metadata:

```bash
curl http://127.0.0.1:8011/docs/returned-doc-id
```

Delete a document:

```bash
curl -X DELETE http://127.0.0.1:8011/docs/returned-doc-id
```

## Hugging Face Spaces

This repo is ready for a Hugging Face Docker Space. Hugging Face Docker Spaces support FastAPI-style apps and expose the app on port `7860`.

Create a new Space:

1. Go to Hugging Face Spaces and create a Space.
2. Choose **Docker** as the SDK.
3. Push this repo to the Space repository, or connect your GitHub repo if you prefer.
4. Add a Space secret:
   - Name: `HF_API_KEY`
   - Value: your Hugging Face token
5. Make sure the Space uses the included `Dockerfile`.

The app will run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Once the Space is running, users can visit:

```text
https://<your-username>-<your-space-name>.hf.space/
```

API docs will be available at:

```text
https://<your-username>-<your-space-name>.hf.space/docs
```

Notes for Spaces:

- Do not commit `.env`; use Space secrets for `HF_API_KEY`.
- Free Spaces may sleep and restart, so local FAISS data can disappear unless persistent storage is enabled.
- The first ingest can be slow because the embedding model may need to download.
- If the generation model is unsupported for your HF account/provider settings, change `GENERATION_MODEL` in the Space variables.

## Configuration

```text
HF_API_KEY=your_huggingface_inference_api_key
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GENERATION_MODEL=meta-llama/Llama-3.2-1B-Instruct
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_DEFAULT=5
DATA_DIR=data
MAX_UPLOAD_MB=10
HF_TIMEOUT_SECONDS=60
```

## Tests

```bash
source .venv/bin/activate
pytest
```

The tests mock model inference and FAISS where needed so core API and pipeline behavior can be checked without downloading models or calling external services.

## Limitations

- No authentication; this is intended as a portfolio/demo app.
- FAISS indexes are local files, not a distributed vector database.
- Chunking uses whitespace tokenization for clarity rather than tokenizer-specific token counts.
- Query generation requires a valid Hugging Face token with Inference Provider access.
- Hugging Face Spaces without persistent storage may lose uploaded document indexes after restart.

## References

- Hugging Face Docker Spaces documentation: https://huggingface.co/docs/hub/spaces-sdks-docker
- Hugging Face Inference Providers documentation: https://huggingface.co/docs/inference-providers/main/index
