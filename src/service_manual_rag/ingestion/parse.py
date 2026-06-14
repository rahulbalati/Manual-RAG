"""Parse PDF with Docling."""

from pathlib import Path
from typing import Any

from service_manual_rag.clients.docling import create_pdf_converter
from service_manual_rag.domain.models import Document
from service_manual_rag.processors.text_spans import extract_text_spans
from service_manual_rag.config import get_settings
from service_manual_rag.storage import save_document, save_text_spans
from service_manual_rag.storage.paths import stable_id

SECTION_HEADER_LABEL = "section_header"


def parse_pdf(pdf_path: Path) -> tuple[Document, list[dict]]:
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    converter = create_pdf_converter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    markdown = doc.export_to_markdown()
    if not markdown.strip():
        raise ValueError(f"No content extracted from {pdf_path.name}")

    page_markdown = _build_page_markdown(doc)
    heading_locations = _extract_heading_locations(doc)
    text_spans = extract_text_spans(doc)

    return Document(
        document_id=stable_id(
            pdf_path.name,
            str(pdf_path.stat().st_size),
            str(pdf_path.stat().st_mtime),
        ),
        source_file=pdf_path,
        title=pdf_path.stem,
        metadata={
            "raw_markdown": markdown,
            "page_markdown": page_markdown,
            "heading_locations": heading_locations,
            "total_pages": len(page_markdown),
            "text_span_count": len(text_spans),
        },
    ), text_spans


def _build_page_markdown(doc: Any) -> dict[int, str]:
    page_markdown: dict[int, str] = {}
    for page_number in doc.pages:
        try:
            page_markdown[int(page_number)] = (
                doc.export_to_markdown(page_no=int(page_number))
            )
        except Exception:
            page_markdown[int(page_number)] = ""
    return page_markdown


def _extract_heading_locations(doc: Any) -> list[dict[str, int | str]]:
    locations: list[dict[str, int | str]] = []
    for item in doc.texts:
        label = str(getattr(item, "label", ""))
        if SECTION_HEADER_LABEL not in label:
            continue
        text = (getattr(item, "text", "") or "").strip()
        if not text or not item.prov:
            continue
        locations.append(
            {
                "title": text,
                "page_no": int(item.prov[0].page_no),
            }
        )
    return locations


def main() -> None:
    print(f"Parsing {get_settings().default_pdf.name}")
    document, text_spans = parse_pdf(get_settings().default_pdf)
    path = save_document(document)
    spans_path = save_text_spans(text_spans, document.document_id)

    page_markdown = document.metadata["page_markdown"]
    nonempty_pages = sum(
        1 for text in page_markdown.values() if text.strip()
    )
    heading_locations = document.metadata["heading_locations"]

    print(f"  document_id: {document.document_id}")
    print(f"  total_pages: {document.metadata['total_pages']}")
    print(f"  markdown_chars: {len(document.metadata['raw_markdown'])}")
    print(f"  pages_with_markdown: {nonempty_pages}")
    print(f"  heading_locations: {len(heading_locations)}")
    print(f"  text_spans: {len(text_spans)}")
    print(f"  saved: {path}")
    print(f"  text_spans saved: {spans_path}")
    print("  preview:")
    preview = document.metadata["raw_markdown"][:400]
    for line in preview.splitlines()[:8]:
        print(f"    {line}")

