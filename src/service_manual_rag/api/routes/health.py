from fastapi import APIRouter

from service_manual_rag.api.dependencies import (
    index_ready,
    list_documents,
    pdf_url,
)
from service_manual_rag.api.schemas import HealthResponse
from service_manual_rag.config import get_settings
from service_manual_rag.storage.paths import document_id_for_pdf

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    documents = list_documents()
    default_id = None
    index_ready_flag = False
    if settings.default_pdf.exists():
        default_id = document_id_for_pdf(settings.default_pdf)
        index_ready_flag = index_ready(default_id)

    return HealthResponse(
        status="ok",
        documents=len(documents),
        default_document_id=default_id,
        index_ready=index_ready_flag,
        pdf_url=pdf_url(default_id) if default_id else None,
    )
