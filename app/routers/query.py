from fastapi import APIRouter, Depends, HTTPException, status

from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.pipeline.embedder import EmbeddingDependencyError
from app.pipeline.generator import GenerationConfigError, GenerationError
from app.pipeline.vector_store import DocumentNotFoundError, VectorStoreError
from app.routers.dependencies import get_app_state
from app.state import AppState


router = APIRouter(tags=["queries"])


@router.post("/query", response_model=QueryResponse)
async def query_document(
    payload: QueryRequest,
    state: AppState = Depends(get_app_state),
) -> QueryResponse:
    top_k = payload.top_k or state.settings.top_k_default

    try:
        query_embedding = state.embedder.embed_query(payload.question)
        contexts = state.vector_store.search(payload.doc_id, query_embedding, top_k)
        answer = await state.generator.generate(payload.question, contexts)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found") from exc
    except EmbeddingDependencyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except (VectorStoreError, GenerationConfigError, GenerationError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return QueryResponse(
        doc_id=payload.doc_id,
        answer=answer,
        sources=[
            SourceChunk(
                chunk_index=context.chunk_index,
                text=context.text,
                score=context.score,
            )
            for context in contexts
        ],
    )
