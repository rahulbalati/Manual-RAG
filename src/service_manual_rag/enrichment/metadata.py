"""Enrich chunks with metadata."""

from copy import deepcopy

from service_manual_rag.domain.models import Chunk
from service_manual_rag.enrichment.highlights import attach_highlights_to_chunk
from service_manual_rag.storage import (
    load_chunks,
    load_text_spans,
    pdf_path_for_document,
    save_chunks,
)


def enrich_chunks(chunks: list[Chunk]) -> list[Chunk]:
    enriched: list[Chunk] = []
    document_id = chunks[0].document_id if chunks else None
    pdf_path = pdf_path_for_document(document_id) if document_id else None
    text_spans = load_text_spans(document_id) if document_id else []

    for chunk in chunks:
        updated = deepcopy(chunk)
        updated.metadata["hierarchy_context"] = " > ".join(
            updated.heading_path
        )
        updated.metadata["provenance"] = {
            "page_start": updated.page_start,
            "page_end": updated.page_end,
            "page_range": (
                f"{updated.page_start}-{updated.page_end}"
            ),
            "source_document": updated.metadata.get(
                "source_document"
            ),
        }
        updated.metadata["chunk_depth"] = len(
            updated.heading_path
        )
        updated.metadata["image_count"] = len(
            updated.image_ids
        )
        updated.metadata["word_count"] = len(
            updated.content.split()
        )
        if pdf_path is not None:
            updated.metadata["highlights"] = attach_highlights_to_chunk(
                updated,
                pdf_path=pdf_path,
                text_spans=text_spans or None,
            )
        enriched.append(updated)
    return enriched


def main() -> None:
    print("Enriching chunks")
    chunks = load_chunks()
    chunks = enrich_chunks(chunks)
    path = save_chunks(chunks, chunks[0].document_id)

    print(f"  chunks: {len(chunks)}")
    print(f"  saved: {path}")
    if chunks:
        sample = chunks[0]
        print("  first chunk:")
        print(f"    title: {sample.title}")
        print(f"    metadata: {sample.metadata}")

