import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from service_manual_rag.api.dependencies import (
    build_sources,
    index_ready,
    resolve_document_id,
)
from service_manual_rag.api.schemas import AskRequest, AskResponse
from service_manual_rag.clients.azure_openai import chat_completion_stream
from service_manual_rag.generation.answer import answer, prepare_answer

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest) -> AskResponse:
    document_id = resolve_document_id(body.document_id)
    if not index_ready(document_id):
        raise HTTPException(
            status_code=503,
            detail=f"Index not ready for document {document_id}. Run indexing first.",
        )

    try:
        result = answer(
            body.query,
            document_id=document_id,
            top_k_chunks=body.chunk_k,
            top_k_figures=body.figure_k,
            include_images=body.include_images,
        )
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        query=result.query,
        document_id=document_id,
        answer=result.answer,
        sources=build_sources(result, document_id),
    )


def _ask_stream_events(body: AskRequest, document_id: str):
    yield (
        "event: status\n"
        f"data: {json.dumps({'message': 'Searching manual...'})}\n\n"
    )

    try:
        _doc_id, result, messages = prepare_answer(
            body.query,
            document_id=document_id,
            top_k_chunks=body.chunk_k,
            top_k_figures=body.figure_k,
            include_images=body.include_images,
        )
    except Exception as exc:
        yield (
            "event: error\n"
            f"data: {json.dumps({'message': str(exc)})}\n\n"
        )
        return

    sources = build_sources(result, document_id)
    yield (
        "event: sources\n"
        f"data: {json.dumps(sources.model_dump())}\n\n"
    )

    yield (
        "event: status\n"
        f"data: {json.dumps({'message': 'Generating answer...'})}\n\n"
    )

    try:
        for token in chat_completion_stream(messages):
            yield (
                "event: token\n"
                f"data: {json.dumps({'content': token})}\n\n"
            )
    except Exception as exc:
        yield (
            "event: error\n"
            f"data: {json.dumps({'message': str(exc)})}\n\n"
        )
        return

    yield "event: done\ndata: {}\n\n"


@router.post("/ask/stream")
def ask_stream_endpoint(body: AskRequest) -> StreamingResponse:
    document_id = resolve_document_id(body.document_id)
    if not index_ready(document_id):
        raise HTTPException(
            status_code=503,
            detail=f"Index not ready for document {document_id}. Run indexing first.",
        )

    return StreamingResponse(
        _ask_stream_events(body, document_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
