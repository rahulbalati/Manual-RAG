from fastapi import APIRouter, HTTPException

from service_manual_rag.api.dependencies import (
    chunk_source,
    index_ready,
    resolve_document_id,
)
from service_manual_rag.api.schemas import RetrieveRequest, RetrieveResponse
from service_manual_rag.retrieval.search import retrieve
from service_manual_rag.storage import chunk_map

router = APIRouter(tags=["retrieve"])


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(body: RetrieveRequest) -> RetrieveResponse:
    document_id = resolve_document_id(body.document_id)
    if not index_ready(document_id):
        raise HTTPException(
            status_code=503,
            detail=f"Index not ready for document {document_id}. Run indexing first.",
        )

    result = retrieve(
        body.query,
        document_id=document_id,
        top_k_chunks=body.chunk_k,
    )
    chunks_by_id = chunk_map(document_id)
    return RetrieveResponse(
        query=result.query,
        document_id=document_id,
        chunks=[
            chunk_source(hit, document_id=document_id, chunks_by_id=chunks_by_id)
            for hit in result.chunks
        ],
    )
