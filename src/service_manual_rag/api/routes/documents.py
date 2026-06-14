from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from service_manual_rag.api.dependencies import list_documents
from service_manual_rag.api.schemas import DocumentInfo
from service_manual_rag.storage import pdf_path_for_document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentInfo])
def get_documents() -> list[DocumentInfo]:
    return list_documents()


@router.get("/{document_id}/pdf")
def get_document_pdf(document_id: str) -> FileResponse:
    try:
        path = pdf_path_for_document(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FileResponse(
        path,
        media_type="application/pdf",
        filename=path.name,
    )
