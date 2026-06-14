"""Build section hierarchy from markdown headings."""

import re
from copy import deepcopy

from service_manual_rag.domain.models import Document, Section
from service_manual_rag.domain.sections import flatten_sections
from service_manual_rag.storage import load_document, save_document
from service_manual_rag.storage.paths import stable_id


HEADING_PATTERNS = [
    (1, re.compile(r"^#\s+(.+)$")),
    (2, re.compile(r"^##\s+(.+)$")),
    (3, re.compile(r"^###\s+(.+)$")),
    (4, re.compile(r"^####\s+(.+)$")),
]


def _detect_heading(line: str) -> tuple[int, str] | None:
    line = line.strip()
    for level, pattern in HEADING_PATTERNS:
        match = pattern.match(line)
        if match:
            return level, match.group(1).strip()
    return None


def build_hierarchy(document: Document) -> Document:
    markdown = document.metadata["raw_markdown"]
    roots: list[Section] = []
    stack: list[Section] = []
    content_lines: list[str] = []

    def flush_content() -> None:
        if not stack or not content_lines:
            return
        text = "\n".join(content_lines).strip()
        if not text:
            return
        if stack[-1].content:
            stack[-1].content += "\n\n"
        stack[-1].content += text

    for line in markdown.splitlines():
        heading = _detect_heading(line)
        if heading:
            flush_content()
            content_lines = []
            level, title = heading
            section = Section(
                section_id=stable_id(
                    document.document_id,
                    title,
                    str(len(roots)),
                ),
                title=title,
                level=level,
            )
            while stack and stack[-1].level >= level:
                stack.pop()
            if stack:
                stack[-1].children.append(section)
            else:
                roots.append(section)
            stack.append(section)
            continue
        content_lines.append(line)

    flush_content()

    updated = deepcopy(document)
    updated.sections = roots
    return updated


def main() -> None:
    print("Building hierarchy")
    document = load_document()
    document = build_hierarchy(document)
    path = save_document(document)

    total = len(flatten_sections(document.sections))
    print(f"  sections: {total}")
    print(f"  saved: {path}")
    print("  tree:")
    lines = 0
    for section in document.sections:
        lines = _print_tree_limited(section, lines, limit=15)
        if lines >= 15:
            break


def _print_tree_limited(
    section: Section,
    lines: int,
    depth: int = 0,
    limit: int = 15,
) -> int:
    if lines >= limit:
        return lines
    print(f"{'  ' * depth}- [{section.level}] {section.title}")
    lines += 1
    for child in section.children:
        lines = _print_tree_limited(
            child, lines, depth + 1, limit
        )
        if lines >= limit:
            break
    return lines

