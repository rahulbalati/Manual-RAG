"""FastAPI dependency helpers and response mappers."""

import chromadb
from fastapi import HTTPException

from service_manual_rag.api.schemas import (
    ChunkSource,
    DocumentInfo,
    FigureSource,
    HighlightRect,
    Sources,
)
from service_manual_rag.config import get_settings
from service_manual_rag.indexing.figures import COLLECTION_NAME as FIGURE_COLLECTION
from service_manual_rag.indexing.text import COLLECTION_NAME as TEXT_COLLECTION
from service_manual_rag.storage import chunk_map, load_document, pdf_path_for_document
from service_manual_rag.storage.paths import chroma_path, document_id_for_pdf


def resolve_document_id(document_id: str | None) -> str:
    if document_id:
        return document_id
    settings = get_settings()
    if settings.default_pdf.exists():
        return document_id_for_pdf(settings.default_pdf)
    raise HTTPException(
        status_code=404,
        detail="No default document found. Run ingestion first.",
    )


def image_url(image_path: str) -> str:
    normalized = str(image_path).replace("\\", "/")
    if normalized.startswith("/"):
        return normalized
    return f"/{normalized}"


def pdf_url(document_id: str) -> str:
    return f"/documents/{document_id}/pdf"


def chunk_source(
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
        pdf_url=pdf_url(document_id),
        highlights=highlights,
    )


def figure_source(hit: dict, *, document_id: str) -> FigureSource:
    page_number = int(hit["page_number"])
    return FigureSource(
        figure_id=hit["figure_id"],
        procedure_title=hit["procedure_title"],
        page_number=page_number,
        page_start=page_number,
        page_end=page_number,
        image_path=str(hit["image_path"]),
        image_url=image_url(str(hit["image_path"])),
        heading_path=str(hit["heading_path"]),
        distance=float(hit["distance"]),
        pdf_url=pdf_url(document_id),
    )


def index_ready(document_id: str) -> bool:
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
    settings = get_settings()
    if not settings.processed_dir.exists():
        return []

    documents: list[DocumentInfo] = []
    for entry in sorted(settings.processed_dir.iterdir()):
        doc_path = entry / "document.json"
        if not entry.is_dir() or not doc_path.exists():
            continue
        document = load_document(entry.name)
        documents.append(
            DocumentInfo(
                document_id=document.document_id,
                title=document.title,
                indexed=index_ready(document.document_id),
                pdf_url=pdf_url(document.document_id),
            )
        )
    return documents


def build_sources(result, document_id: str) -> Sources:
    chunks_by_id = chunk_map(document_id)
    return Sources(
        chunks=[
            chunk_source(hit, document_id=document_id, chunks_by_id=chunks_by_id)
            for hit in result.chunks
        ],
        figures=[
            figure_source(hit, document_id=document_id)
            for hit in result.figures
        ],
    )
