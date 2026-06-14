"""Phase 6: Link figures to sections by page range."""

from copy import deepcopy

from src.domain.models import Document, Section
from src.utils import (
    flatten_sections,
    heading_path,
    load_document,
    save_document,
)


def associate_figures(document: Document) -> Document:
    updated = deepcopy(document)
    sections = flatten_sections(updated.sections)
    sections.sort(key=lambda s: s.level, reverse=True)

    for figure in updated.figures:
        owner = None
        for section in sections:
            if (
                section.page_start is not None
                and section.page_end is not None
                and section.page_start
                <= figure.page_number
                <= section.page_end
            ):
                owner = section
                break

        if owner:
            owner.figure_ids.append(figure.figure_id)
            figure.heading_path = heading_path(
                owner,
                updated.sections,
            )

    return updated


def main() -> None:
    print("Phase 6: Associating figures to sections")
    document = load_document()
    document = associate_figures(document)
    path = save_document(document)

    linked = sum(
        1 for f in document.figures if f.heading_path
    )
    print(f"  figures_linked: {linked}/{len(document.figures)}")
    print(f"  saved: {path}")
    for figure in document.figures[:5]:
        path_str = " > ".join(figure.heading_path) or "(unlinked)"
        print(f"    {figure.figure_id}: {path_str}")


if __name__ == "__main__":
    main()
