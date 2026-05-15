from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_indexed: int
    status: Literal["success"]


class QueryRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceChunk(BaseModel):
    chunk_index: int
    text: str
    score: float


class QueryResponse(BaseModel):
    doc_id: str
    answer: str
    sources: list[SourceChunk]


class DocumentMetadataResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_indexed: int
    embedding_model: str
    created_at: datetime


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    status: Literal["deleted"]
