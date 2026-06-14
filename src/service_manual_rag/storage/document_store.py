"""Document persistence."""

from pathlib import Path

from service_manual_rag.config import get_settings
from service_manual_rag.domain.models import Document
from service_manual_rag.storage.paths import (
    default_document_id,
    document_path,
)


def save_document(document: Document) -> Path:
    path = document_path(document.document_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        document.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return path


def load_document(document_id: str | None = None) -> Document:
    if document_id is None:
        document_id = default_document_id()
    path = document_path(document_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the previous ingestion step first."
        )
    return Document.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def pdf_path_for_document(document_id: str | None = None) -> Path:
    document = load_document(document_id)
    source = Path(document.source_file)
    if source.exists():
        return source
    default_pdf = get_settings().default_pdf
    if default_pdf.exists():
        return default_pdf.resolve()
    raise FileNotFoundError(
        f"PDF not found for document {document.document_id}."
    )
