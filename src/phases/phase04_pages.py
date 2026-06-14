"""Phase 4: Assign page numbers to sections."""

from copy import deepcopy

from src.domain.models import Document, Section
from src.utils import flatten_sections, load_document, save_document


def _collect_heading_pages(document: Document) -> list[tuple[str, int]]:
    heading_pages: list[tuple[str, int]] = []

    for entry in document.metadata.get("heading_locations", []):
        title = str(entry.get("title", "")).strip()
        page_no = entry.get("page_no")
        if title and page_no is not None:
            heading_pages.append((title, int(page_no)))

    if heading_pages:
        return heading_pages

    page_map = document.metadata.get("page_markdown", {})
    for page_number, markdown in page_map.items():
        for line in markdown.splitlines():
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            title = stripped.lstrip("#").strip()
            if title:
                heading_pages.append((title, int(page_number)))

    return heading_pages


def resolve_pages(document: Document) -> Document:
    updated = deepcopy(document)
    heading_pages = _collect_heading_pages(updated)
    total_pages = updated.metadata.get("total_pages", 1)

    sections = flatten_sections(updated.sections)
    for section in sections:
        for title, page_number in heading_pages:
            if title.strip() == section.title.strip():
                section.page_start = page_number
                break

    sections.sort(
        key=lambda s: s.page_start if s.page_start is not None else 999999
    )

    for idx, section in enumerate(sections):
        if section.page_start is None:
            continue
        if idx == len(sections) - 1:
            section.page_end = total_pages
            continue
        next_section = sections[idx + 1]
        if next_section.page_start is None:
            section.page_end = section.page_start
        else:
            section.page_end = max(
                section.page_start,
                next_section.page_start - 1,
            )

    return updated


def main() -> None:
    print("Phase 4: Resolving page boundaries")
    document = load_document()
    document = resolve_pages(document)
    path = save_document(document)

    sections = flatten_sections(document.sections)
    with_pages = [
        s for s in sections if s.page_start is not None
    ]

    print(f"  sections_with_pages: {len(with_pages)}/{len(sections)}")
    print(f"  saved: {path}")
    print("  sample:")
    for section in with_pages[:10]:
        print(
            f"    {section.title} "
            f"[{section.page_start}-{section.page_end}]"
        )
    if not with_pages:
        print("    (no page matches — re-run Phase 2 first)")


if __name__ == "__main__":
    main()
