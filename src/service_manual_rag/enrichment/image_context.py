"""Build text context for each figure."""

from copy import deepcopy

from service_manual_rag.domain.models import Document, Section
from service_manual_rag.domain.sections import flatten_sections
from service_manual_rag.storage import load_document, save_document


def build_image_context(document: Document) -> Document:
    updated = deepcopy(document)
    figure_to_section: dict[str, Section] = {}

    for section in flatten_sections(updated.sections):
        for figure_id in section.figure_ids:
            figure_to_section[figure_id] = section

    for figure in updated.figures:
        section = figure_to_section.get(figure.figure_id)
        if not section:
            continue
        hierarchy = " > ".join(figure.heading_path)
        content = section.content.strip()[:3000]
        figure.context_text = (
            f"Hierarchy:\n{hierarchy}\n\n"
            f"Procedure:\n{section.title}\n\n"
            f"Procedure Content:\n{content}"
        )

    return updated


def main() -> None:
    print("Building image context")
    document = load_document()
    document = build_image_context(document)
    path = save_document(document)

    with_context = [
        f for f in document.figures if f.context_text
    ]
    print(f"  figures_with_context: {len(with_context)}")
    print(f"  saved: {path}")

    if with_context:
        print("  first figure context:")
        print(with_context[0].context_text[:500])
    else:
        print("  (no figures to build context for)")

