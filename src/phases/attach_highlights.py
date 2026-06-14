"""Attach PDF highlight rectangles to existing chunks."""

from src.processors.highlights import attach_highlights_to_chunks
from src.utils import (
    load_chunks,
    load_text_spans,
    pdf_path_for_document,
    save_chunks,
)


def main() -> None:
    print("Attach highlights: mapping chunk text to PDF regions")
    chunks = load_chunks()
    document_id = chunks[0].document_id
    pdf_path = pdf_path_for_document(document_id)
    text_spans = load_text_spans(document_id)

    chunks = attach_highlights_to_chunks(
        chunks,
        pdf_path=pdf_path,
        text_spans=text_spans or None,
    )
    path = save_chunks(chunks, document_id)

    with_highlights = sum(
        1 for chunk in chunks if chunk.metadata.get("highlights")
    )
    total_regions = sum(
        len(chunk.metadata.get("highlights", [])) for chunk in chunks
    )

    print(f"  chunks: {len(chunks)}")
    print(f"  chunks with highlights: {with_highlights}")
    print(f"  highlight regions: {total_regions}")
    print(f"  text_spans available: {len(text_spans)}")
    print(f"  saved: {path}")


if __name__ == "__main__":
    main()
