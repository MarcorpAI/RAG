import numpy as np
import pytest
from fastapi import HTTPException

from app.pipeline.generator import RetrievedContext
from app.routers.docs import delete_document, get_document_metadata
from app.routers.ingest import ingest_document
from app.routers.query import query_document
from app.models.schemas import QueryRequest
from app.state import AppState


class FakeSettings:
    chunk_size = 4
    chunk_overlap = 1
    top_k_default = 2
    max_upload_mb = 1


class FakeEmbedder:
    model_name = "fake-embedding-model"

    def embed_texts(self, texts):
        return np.ones((len(texts), 2), dtype="float32")

    def embed_query(self, text):
        return np.ones((1, 2), dtype="float32")


class FakeGenerator:
    async def generate(self, question, contexts):
        return f"answer for {question}"


class FakeVectorStore:
    def __init__(self):
        self.record = None
        self.deleted = False

    def load_existing(self):
        return None

    def add_document(self, filename, chunks, embeddings, embedding_model):
        self.record = type(
            "Record",
            (),
            {
                "doc_id": "doc-1",
                "filename": filename,
                "chunks": chunks,
                "embedding_model": embedding_model,
                "created_at": "2026-05-15T00:00:00+00:00",
                "chunks_indexed": len(chunks),
            },
        )()
        return self.record

    def get_document(self, doc_id):
        return self.record

    def search(self, doc_id, query_embedding, top_k):
        return [RetrievedContext(chunk_index=0, text="source text", score=0.75)]

    def delete_document(self, doc_id):
        self.deleted = True


class FakeUploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self._content = content

    async def read(self):
        return self._content


def fake_state():
    return AppState(
        settings=FakeSettings(),
        embedder=FakeEmbedder(),
        generator=FakeGenerator(),
        vector_store=FakeVectorStore(),
    )


@pytest.mark.anyio
async def test_ingest_query_metadata_and_delete():
    state = fake_state()
    upload = FakeUploadFile("sample.txt", b"alpha beta gamma delta epsilon")

    ingest = await ingest_document(file=upload, state=state)
    assert ingest.doc_id == "doc-1"
    assert ingest.chunks_indexed == 2

    query = await query_document(
        QueryRequest(doc_id="doc-1", question="What is inside?", top_k=1),
        state=state,
    )
    assert query.answer == "answer for What is inside?"
    assert query.sources[0].text == "source text"

    metadata = get_document_metadata("doc-1", state=state)
    assert metadata.filename == "sample.txt"

    deleted = delete_document("doc-1", state=state)
    assert deleted.doc_id == "doc-1"
    assert deleted.status == "deleted"


@pytest.mark.anyio
async def test_ingest_rejects_unsupported_file_type():
    upload = FakeUploadFile("sample.docx", b"alpha beta")

    with pytest.raises(HTTPException) as exc_info:
        await ingest_document(file=upload, state=fake_state())

    assert exc_info.value.status_code == 415
