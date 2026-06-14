"""Path builders and stable ID generation."""

import hashlib
from pathlib import Path

from service_manual_rag.config import get_settings


def stable_id(*parts: str) -> str:
    joined = "::".join(parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def document_id_for_pdf(pdf: Path) -> str:
    pdf = pdf.resolve()
    return stable_id(
        pdf.name,
        str(pdf.stat().st_size),
        str(pdf.stat().st_mtime),
    )


def output_dir(document_id: str) -> Path:
    return get_settings().processed_dir / document_id


def document_path(document_id: str) -> Path:
    return output_dir(document_id) / "document.json"


def chunks_path(document_id: str) -> Path:
    return output_dir(document_id) / "chunks.json"


def text_spans_path(document_id: str) -> Path:
    return output_dir(document_id) / "text_spans.json"


def chroma_path(document_id: str) -> Path:
    return get_settings().index_dir / document_id / "chroma"


def default_document_id() -> str:
    return document_id_for_pdf(get_settings().default_pdf)
