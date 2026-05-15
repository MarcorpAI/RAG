import json
import sys
from pathlib import Path

import numpy as np
import pytest

from app.pipeline.chunker import TextChunk, chunk_text
from app.pipeline.document_loader import EmptyDocumentError, UnsupportedDocumentType, extract_text
from app.pipeline.generator import RetrievedContext, build_prompt
from app.pipeline.vector_store import FaissDocumentStore


class FakeIndex:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.vectors = None

    def add(self, vectors):
        self.vectors = vectors

    def search(self, query, top_k):
        scores = self.vectors @ query[0]
        order = np.argsort(scores)[::-1][:top_k]
        return np.array([scores[order]], dtype="float32"), np.array([order], dtype="int64")


class FakeFaiss:
    @staticmethod
    def IndexFlatIP(dimensions):
        return FakeIndex(dimensions)

    @staticmethod
    def write_index(index, path):
        payload = {"dimensions": index.dimensions, "vectors": index.vectors.tolist()}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def read_index(path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = FakeIndex(payload["dimensions"])
        index.add(np.asarray(payload["vectors"], dtype="float32"))
        return index


def test_chunk_text_uses_overlap():
    text = " ".join(f"tok{i}" for i in range(10))

    chunks = chunk_text(text, chunk_size=4, overlap=1)

    assert [chunk.text for chunk in chunks] == [
        "tok0 tok1 tok2 tok3",
        "tok3 tok4 tok5 tok6",
        "tok6 tok7 tok8 tok9",
    ]
    assert chunks[1].start_token == 3


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("hello world", chunk_size=4, overlap=4)


def test_extract_text_supports_txt_and_rejects_empty_or_unknown():
    assert extract_text("note.txt", b"hello\nworld") == "hello world"

    with pytest.raises(UnsupportedDocumentType):
        extract_text("note.docx", b"hello")

    with pytest.raises(EmptyDocumentError):
        extract_text("empty.txt", b"   ")


def test_build_prompt_enforces_grounding_instruction():
    prompt = build_prompt(
        "What is the answer?",
        [RetrievedContext(chunk_index=0, text="Only this fact exists.", score=0.9)],
    )

    assert "using ONLY the context" in prompt
    assert "I don't know based on the provided document" in prompt
    assert "Only this fact exists." in prompt
    assert "Question: What is the answer?" in prompt


def test_vector_store_persists_searches_and_deletes(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", FakeFaiss)
    store = FaissDocumentStore(tmp_path)
    chunks = [
        TextChunk(chunk_index=0, text="alpha", start_token=0, end_token=1),
        TextChunk(chunk_index=1, text="beta", start_token=1, end_token=2),
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")

    record = store.add_document("doc.txt", chunks, embeddings, "test-model")
    results = store.search(record.doc_id, np.asarray([[0.9, 0.1]], dtype="float32"), 1)

    assert results[0].chunk_index == 0

    reloaded = FaissDocumentStore(tmp_path)
    reloaded.load_existing()
    assert reloaded.get_document(record.doc_id).filename == "doc.txt"

    reloaded.delete_document(record.doc_id)
    assert not (tmp_path / record.doc_id).exists()
