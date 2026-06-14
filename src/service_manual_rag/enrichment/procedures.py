"""Detect procedure sections."""

from service_manual_rag.domain.models import Document, Section
from service_manual_rag.domain.sections import flatten_sections, heading_path
from service_manual_rag.storage import load_document


PROCEDURE_KEYWORDS = (
    "removal",
    "replacement",
    "replace",
    "replacing",
    "maintenance",
    "cleaning",
    "calibration",
    "adjustment",
    "error",
    "errors",
    "jam",
    "jams",
    "troubleshooting",
    "diagnostics",
    "inspection",
    "installation",
    "configuration",
    "gray background",
    "blank pages",
    "print is too dark",
    "print is too light",
)


def is_procedure(section: Section) -> bool:
    title = section.title.lower().strip()
    return any(keyword in title for keyword in PROCEDURE_KEYWORDS)


def detect_procedures(document: Document) -> list[Section]:
    return [
        section
        for section in flatten_sections(document.sections)
        if is_procedure(section)
    ]


def main() -> None:
    print("Detecting procedures")
    document = load_document()
    procedures = detect_procedures(document)

    print(f"  procedures: {len(procedures)}")
    print("  sample:")
    for section in procedures[:15]:
        path = " > ".join(
            heading_path(section, document.sections)
        )
        print(f"    {section.title}")
        print(f"      path: {path}")
        print(
            f"      pages: {section.page_start}-"
            f"{section.page_end}"
        )

