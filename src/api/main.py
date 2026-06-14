"""Phase 15: FastAPI service for RAG retrieval and answers."""

import json
from pathlib import Path

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    AskRequest,
    AskResponse,
    ChunkSource,
    DocumentInfo,
    FigureSource,
    HealthResponse,
    HighlightRect,
    RetrieveRequest,
    RetrieveResponse,
    Sources,
)
from src.phases.phase11_index_text import COLLECTION_NAME as TEXT_COLLECTION
from src.phases.phase12_index_figures import COLLECTION_NAME as FIGURE_COLLECTION
from src.phases.phase14_answer import answer, prepare_answer
from src.azure_openai_client import chat_completion_stream
from src.phases.phase13_retrieve import retrieve
from src.utils import (
    ASSETS_DIR,
    PDF_PATH,
    chroma_path,
    chunk_map,
    document_id_for_pdf,
    load_document,
    pdf_path_for_document,
    PROCESSED_DIR,
)

UI_DIR = Path(__file__).resolve().parents[2] / "ui"


def _resolve_document_id(document_id: str | None) -> str:
    if document_id:
        return document_id
    if PDF_PATH.exists():
        return document_id_for_pdf(PDF_PATH)
    raise HTTPException(
        status_code=404,
        detail="No default document found. Run ingestion first.",
    )


def _image_url(image_path: str) -> str:
    normalized = str(image_path).replace("\\", "/")
    if normalized.startswith("/"):
        return normalized
    return f"/{normalized}"


def _pdf_url(document_id: str) -> str:
    return f"/documents/{document_id}/pdf"


def _chunk_source(
    hit: dict,
    *,
    document_id: str,
    chunks_by_id: dict,
) -> ChunkSource:
    chunk = chunks_by_id.get(hit["chunk_id"])
    highlights: list[HighlightRect] = []
    if chunk is not None:
        for item in chunk.metadata.get("highlights", []):
            highlights.append(
                HighlightRect(
                    page=int(item["page"]),
                    rect=[float(v) for v in item["rect"]],
                )
            )

    page_start = int(
        hit.get("page_start")
        or (chunk.page_start if chunk else 0)
        or str(hit.get("page_range", "0-0")).split("-")[0]
    )
    page_end = int(
        hit.get("page_end")
        or (chunk.page_end if chunk else page_start)
        or str(hit.get("page_range", f"{page_start}-{page_start}")).split("-")[-1]
    )

    return ChunkSource(
        chunk_id=hit["chunk_id"],
        title=hit["title"],
        page_start=page_start,
        page_end=page_end,
        page_range=str(hit.get("page_range", f"{page_start}-{page_end}")),
        snippet=str(hit.get("snippet", ""))[:300],
        chunk_type=hit["chunk_type"],
        distance=float(hit["distance"]),
        pdf_url=_pdf_url(document_id),
        highlights=highlights,
    )


def _figure_source(hit: dict, *, document_id: str) -> FigureSource:
    page_number = int(hit["page_number"])
    return FigureSource(
        figure_id=hit["figure_id"],
        procedure_title=hit["procedure_title"],
        page_number=page_number,
        page_start=page_number,
        page_end=page_number,
        image_path=str(hit["image_path"]),
        image_url=_image_url(str(hit["image_path"])),
        heading_path=str(hit["heading_path"]),
        distance=float(hit["distance"]),
        pdf_url=_pdf_url(document_id),
    )


def _index_ready(document_id: str) -> bool:
    path = chroma_path(document_id)
    if not path.exists():
        return False
    try:
        chroma = chromadb.PersistentClient(path=str(path))
        chroma.get_collection(TEXT_COLLECTION)
        chroma.get_collection(FIGURE_COLLECTION)
        return True
    except Exception:
        return False


def list_documents() -> list[DocumentInfo]:
    if not PROCESSED_DIR.exists():
        return []

    documents: list[DocumentInfo] = []
    for entry in sorted(PROCESSED_DIR.iterdir()):
        doc_path = entry / "document.json"
        if not entry.is_dir() or not doc_path.exists():
            continue
        document = load_document(entry.name)
        documents.append(
            DocumentInfo(
                document_id=document.document_id,
                title=document.title,
                indexed=_index_ready(document.document_id),
                pdf_url=_pdf_url(document.document_id),
            )
        )
    return documents


app = FastAPI(
    title="Service Manual RAG API",
    description="Retrieval and Q&A over technical service manuals",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    documents = list_documents()
    default_id = None
    index_ready = False
    if PDF_PATH.exists():
        default_id = document_id_for_pdf(PDF_PATH)
        index_ready = _index_ready(default_id)

    return HealthResponse(
        status="ok",
        documents=len(documents),
        default_document_id=default_id,
        index_ready=index_ready,
        pdf_url=_pdf_url(default_id) if default_id else None,
    )


@app.get("/documents/{document_id}/pdf")
def get_document_pdf(document_id: str) -> FileResponse:
    try:
        pdf_path = pdf_path_for_document(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
    )


@app.get("/documents", response_model=list[DocumentInfo])
def get_documents() -> list[DocumentInfo]:
    return list_documents()


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(body: RetrieveRequest) -> RetrieveResponse:
    document_id = _resolve_document_id(body.document_id)
    if not _index_ready(document_id):
        raise HTTPException(
            status_code=503,
            detail=f"Index not ready for document {document_id}. Run phases 11-12.",
        )

    result = retrieve(
        body.query,
        document_id=document_id,
        top_k_chunks=body.chunk_k,
        top_k_figures=body.figure_k,
    )
    chunks_by_id = chunk_map(document_id)
    return RetrieveResponse(
        query=result.query,
        document_id=document_id,
        chunks=[
            _chunk_source(hit, document_id=document_id, chunks_by_id=chunks_by_id)
            for hit in result.chunks
        ],
        figures=[
            _figure_source(hit, document_id=document_id)
            for hit in result.figures
        ],
    )


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest) -> AskResponse:
    document_id = _resolve_document_id(body.document_id)
    if not _index_ready(document_id):
        raise HTTPException(
            status_code=503,
            detail=f"Index not ready for document {document_id}. Run phases 11-12.",
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
        sources=_build_sources(result, document_id),
    )


def _build_sources(result, document_id: str) -> Sources:
    chunks_by_id = chunk_map(document_id)
    return Sources(
        chunks=[
            _chunk_source(hit, document_id=document_id, chunks_by_id=chunks_by_id)
            for hit in result.chunks
        ],
        figures=[
            _figure_source(hit, document_id=document_id)
            for hit in result.figures
        ],
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

    sources = _build_sources(result, document_id)
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


@app.post("/ask/stream")
def ask_stream_endpoint(body: AskRequest) -> StreamingResponse:
    document_id = _resolve_document_id(body.document_id)
    if not _index_ready(document_id):
        raise HTTPException(
            status_code=503,
            detail=f"Index not ready for document {document_id}. Run phases 11-12.",
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


def main() -> None:
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
