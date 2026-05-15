from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from app.models.schemas import DeleteDocumentResponse, DocumentMetadataResponse
from app.pipeline.vector_store import DocumentNotFoundError
from app.routers.dependencies import get_app_state
from app.state import AppState


router = APIRouter(prefix="/docs", tags=["documents"])


@router.get("/{doc_id}", response_model=DocumentMetadataResponse)
def get_document_metadata(
    doc_id: str,
    state: AppState = Depends(get_app_state),
) -> DocumentMetadataResponse:
    try:
        record = state.vector_store.get_document(doc_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found") from exc

    return DocumentMetadataResponse(
        doc_id=record.doc_id,
        filename=record.filename,
        chunks_indexed=record.chunks_indexed,
        embedding_model=record.embedding_model,
        created_at=datetime.fromisoformat(record.created_at),
    )


@router.delete("/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document(
    doc_id: str,
    state: AppState = Depends(get_app_state),
) -> DeleteDocumentResponse:
    try:
        state.vector_store.delete_document(doc_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found") from exc
    return DeleteDocumentResponse(doc_id=doc_id, status="deleted")
