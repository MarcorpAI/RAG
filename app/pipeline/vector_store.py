import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from app.pipeline.chunker import TextChunk
from app.pipeline.generator import RetrievedContext


class VectorStoreError(RuntimeError):
    pass


class DocumentNotFoundError(KeyError):
    pass


@dataclass
class DocumentRecord:
    doc_id: str
    filename: str
    chunks: list[TextChunk]
    embedding_model: str
    created_at: str

    @property
    def chunks_indexed(self) -> int:
        return len(self.chunks)


class FaissDocumentStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.documents: dict[str, DocumentRecord] = {}
        self.indexes: dict[str, object] = {}

    def load_existing(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for doc_dir in self.data_dir.iterdir():
            if not doc_dir.is_dir():
                continue
            metadata_path = doc_dir / "metadata.json"
            index_path = doc_dir / "index.faiss"
            if not metadata_path.exists() or not index_path.exists():
                continue
            try:
                record = self._read_metadata(metadata_path)
                self.documents[record.doc_id] = record
                self.indexes[record.doc_id] = self._read_index(index_path)
            except Exception:
                continue

    def add_document(
        self,
        filename: str,
        chunks: list[TextChunk],
        embeddings: np.ndarray,
        embedding_model: str,
    ) -> DocumentRecord:
        if not chunks:
            raise VectorStoreError("Cannot index a document with no chunks")
        if embeddings.shape[0] != len(chunks):
            raise VectorStoreError("Embedding count must match chunk count")

        doc_id = str(uuid4())
        doc_dir = self.data_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=False)

        index = self._new_index(embeddings)
        record = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            chunks=chunks,
            embedding_model=embedding_model,
            created_at=datetime.now(UTC).isoformat(),
        )
        self._write_index(index, doc_dir / "index.faiss")
        self._write_metadata(record, doc_dir / "metadata.json")
        self.documents[doc_id] = record
        self.indexes[doc_id] = index
        return record

    def get_document(self, doc_id: str) -> DocumentRecord:
        try:
            return self.documents[doc_id]
        except KeyError as exc:
            raise DocumentNotFoundError(doc_id) from exc

    def delete_document(self, doc_id: str) -> None:
        self.get_document(doc_id)
        self.documents.pop(doc_id, None)
        self.indexes.pop(doc_id, None)
        shutil.rmtree(self.data_dir / doc_id, ignore_errors=True)

    def search(self, doc_id: str, query_embedding: np.ndarray, top_k: int) -> list[RetrievedContext]:
        record = self.get_document(doc_id)
        index = self.indexes.get(doc_id)
        if index is None:
            raise VectorStoreError("Document index is not loaded")
        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            raise VectorStoreError("Query embedding must have shape (1, dimensions)")

        limit = min(top_k, len(record.chunks))
        scores, indices = index.search(query_embedding.astype("float32"), limit)
        contexts: list[RetrievedContext] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            chunk = record.chunks[int(idx)]
            contexts.append(
                RetrievedContext(
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=float(score),
                )
            )
        return contexts

    def _new_index(self, embeddings: np.ndarray):
        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError("faiss-cpu is required for vector search") from exc
        if embeddings.ndim != 2:
            raise VectorStoreError("Embeddings must be a 2D array")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype("float32"))
        return index

    def _read_index(self, path: Path):
        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError("faiss-cpu is required for vector search") from exc
        return faiss.read_index(str(path))

    def _write_index(self, index, path: Path) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError("faiss-cpu is required for vector search") from exc
        faiss.write_index(index, str(path))

    @staticmethod
    def _write_metadata(record: DocumentRecord, path: Path) -> None:
        payload = {
            "doc_id": record.doc_id,
            "filename": record.filename,
            "embedding_model": record.embedding_model,
            "created_at": record.created_at,
            "chunks": [asdict(chunk) for chunk in record.chunks],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _read_metadata(path: Path) -> DocumentRecord:
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [TextChunk(**chunk) for chunk in payload["chunks"]]
        return DocumentRecord(
            doc_id=payload["doc_id"],
            filename=payload["filename"],
            chunks=chunks,
            embedding_model=payload["embedding_model"],
            created_at=payload["created_at"],
        )
