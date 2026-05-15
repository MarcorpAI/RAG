from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.models.schemas import IngestResponse
from app.pipeline.chunker import chunk_text
from app.pipeline.document_loader import (
    EmptyDocumentError,
    UnsupportedDocumentType,
    extract_text,
)
from app.pipeline.embedder import EmbeddingDependencyError
from app.pipeline.vector_store import VectorStoreError
from app.routers.dependencies import get_app_state
from app.state import AppState


router = APIRouter(tags=["documents"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    state: AppState = Depends(get_app_state),
) -> IngestResponse:
    content = await file.read()
    max_bytes = state.settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {state.settings.max_upload_mb} MB limit",
        )

    try:
        text = extract_text(file.filename or "upload", content)
    except UnsupportedDocumentType as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    chunks = chunk_text(
        text,
        chunk_size=state.settings.chunk_size,
        overlap=state.settings.chunk_overlap,
    )
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document did not produce any chunks",
        )

    try:
        embeddings = state.embedder.embed_texts([chunk.text for chunk in chunks])
        record = state.vector_store.add_document(
            filename=file.filename or "upload",
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=state.embedder.model_name,
        )
    except (EmbeddingDependencyError, VectorStoreError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return IngestResponse(
        doc_id=record.doc_id,
        filename=record.filename,
        chunks_indexed=record.chunks_indexed,
        status="success",
    )
