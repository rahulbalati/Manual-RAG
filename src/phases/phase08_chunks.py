"""Phase 8: Generate chunks from all document sections."""

from src.domain.models import Chunk, ChunkType, Document, Section
from src.phases.error_table_split import (
    extract_error_table_rows,
    format_error_row_content,
)
from src.utils import (
    flatten_sections,
    heading_path,
    load_document,
    save_chunks,
    stable_id,
)


def classify_chunk(title: str) -> ChunkType:
    lower = title.lower()
    if any(k in lower for k in ("error", "errors", "jam", "jams")):
        return ChunkType.ERROR_CODE
    if any(
        k in lower
        for k in (
            "print quality",
            "background",
            "blank pages",
            "too dark",
            "too light",
        )
    ):
        return ChunkType.TROUBLESHOOTING
    if any(k in lower for k in ("maintenance", "cleaning", "replace")):
        return ChunkType.MAINTENANCE
    return ChunkType.PROCEDURE


def _normalize_content(content: str) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return "\n".join(lines)


def _section_has_content(section: Section) -> bool:
    if _normalize_content(section.content):
        return True
    return bool(extract_error_table_rows(section.content))


def _build_heading_path(
    section: Section,
    document: Document,
    parent_titles: list[str],
) -> list[str]:
    path = heading_path(section, document.sections)
    if parent_titles:
        return [*parent_titles, *path]
    return path


def _prepend_parent_context(
    content: str,
    parent_titles: list[str],
) -> str:
    if not parent_titles:
        return content
    prefix = f"Parent sections: {' > '.join(parent_titles)}\n\n"
    return prefix + content


def _base_metadata(
    section: Section,
    document: Document,
    parent_titles: list[str],
) -> dict:
    metadata = {
        "section_id": section.section_id,
        "source_document": document.title,
    }
    if parent_titles:
        metadata["parent_section_titles"] = " > ".join(parent_titles)
    return metadata


def _section_chunk(
    section: Section,
    document: Document,
    *,
    content: str,
    title: str | None = None,
    chunk_id_parts: list[str] | None = None,
    extra_metadata: dict | None = None,
    parent_titles: list[str] | None = None,
) -> Chunk:
    parents = parent_titles or []
    metadata = _base_metadata(section, document, parents)
    if extra_metadata:
        metadata.update(extra_metadata)

    parts = chunk_id_parts or [
        "chunk",
        document.document_id,
        section.section_id,
    ]

    return Chunk(
        chunk_id=stable_id(*parts),
        document_id=document.document_id,
        chunk_type=classify_chunk(title or section.title),
        title=title or section.title,
        heading_path=_build_heading_path(
            section,
            document,
            parents,
        ),
        content=_prepend_parent_context(
            _normalize_content(content),
            parents,
        ),
        page_start=section.page_start or 0,
        page_end=section.page_end or 0,
        image_ids=list(section.figure_ids),
        metadata=metadata,
    )


def _chunks_for_section(
    section: Section,
    document: Document,
    *,
    parent_titles: list[str] | None = None,
) -> list[Chunk]:
    parents = parent_titles or []
    normalized = _normalize_content(section.content)
    if not normalized:
        return []

    table_rows = extract_error_table_rows(section.content)
    if not table_rows:
        return [
            _section_chunk(
                section,
                document,
                content=normalized,
                parent_titles=parents,
            ),
        ]

    return [
        _section_chunk(
            section,
            document,
            title=f"Error {code} - {section.title}",
            content=format_error_row_content(
                section_title=section.title,
                error_code=code,
                description=description,
                action=action,
            ),
            chunk_id_parts=[
                "chunk",
                document.document_id,
                section.section_id,
                code,
            ],
            extra_metadata={
                "error_code": code,
                "parent_section_title": section.title,
                "chunk_granularity": "error_row",
            },
            parent_titles=parents,
        )
        for code, description, action in table_rows
    ]


def _ordered_sections(document: Document) -> list[Section]:
    sections = flatten_sections(document.sections)
    return sorted(
        sections,
        key=lambda section: (
            section.page_start if section.page_start is not None else 999999,
            section.title,
        ),
    )


def generate_chunks(document: Document) -> list[Chunk]:
    sections = _ordered_sections(document)

    chunks: list[Chunk] = []
    pending_parents: list[str] = []

    for section in sections:
        if not _section_has_content(section):
            pending_parents.append(section.title)
            continue

        chunks.extend(
            _chunks_for_section(
                section,
                document,
                parent_titles=list(pending_parents),
            )
        )
        pending_parents = []

    return chunks


def main() -> None:
    print("Phase 8: Generating chunks")
    document = load_document()
    sections = _ordered_sections(document)
    chunks = generate_chunks(document)
    path = save_chunks(chunks, document.document_id)

    sections_with_content = sum(
        1 for section in sections if _section_has_content(section)
    )
    row_chunks = sum(
        1
        for c in chunks
        if c.metadata.get("chunk_granularity") == "error_row"
    )
    table_sections = len(
        {
            c.metadata.get("section_id")
            for c in chunks
            if c.metadata.get("chunk_granularity") == "error_row"
        }
    )
    with_parents = sum(
        1 for c in chunks if c.metadata.get("parent_section_titles")
    )

    print(f"  sections: {len(sections)}")
    print(f"  sections with content: {sections_with_content}")
    print(f"  chunks: {len(chunks)}")
    print(f"  error_row_chunks: {row_chunks} (from {table_sections} tables)")
    print(f"  chunks with parent context: {with_parents}")
    print(f"  empty chunks: 0")
    print(f"  saved: {path}")
    print("  newly included sample:")
    for chunk in chunks:
        if chunk.title == "White streaks and voided areas":
            print(f"    {chunk.title} | p{chunk.page_start}-{chunk.page_end}")
            break


if __name__ == "__main__":
    main()
