"""Attach PDF highlight rectangles to chunks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from service_manual_rag.domain.models import Chunk
from service_manual_rag.storage import (
    load_chunks,
    load_text_spans,
    pdf_path_for_document,
    save_chunks,
)


def normalize_for_match(text: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"[|#*\[\](){}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _rects_overlap(
    a: list[float],
    b: list[float],
    *,
    pad: float = 3.0,
) -> bool:
    al, ab, ar, at = a
    bl, bb, br, bt = b
    return not (
        ar + pad < bl
        or br + pad < al
        or at + pad < bb
        or bt + pad < ab
    )


def merge_rects(
    rects: list[list[float]],
    *,
    pad: float = 3.0,
) -> list[list[float]]:
    if not rects:
        return []

    merged = [list(rect) for rect in rects]
    changed = True
    while changed:
        changed = False
        next_pass: list[list[float]] = []
        used = [False] * len(merged)

        for i, rect in enumerate(merged):
            if used[i]:
                continue
            current = rect[:]
            used[i] = True
            for j, other in enumerate(merged):
                if used[j] or i == j:
                    continue
                if _rects_overlap(current, other, pad=pad):
                    current = [
                        min(current[0], other[0]),
                        min(current[1], other[1]),
                        max(current[2], other[2]),
                        max(current[3], other[3]),
                    ]
                    used[j] = True
                    changed = True
            next_pass.append(current)
        merged = next_pass

    return merged


def _search_terms_for_chunk(chunk: Chunk) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        cleaned = re.sub(r"\s+", " ", term).strip()
        if len(cleaned) < 4:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        terms.append(cleaned)

    add(chunk.title)
    if chunk.heading_path:
        add(chunk.heading_path[-1])

    for line in chunk.content.splitlines():
        line = re.sub(r"<!--.*?-->", "", line).strip()
        if not line or line.startswith("|") or line.startswith("-"):
            continue
        if line.lower().startswith("parent sections:"):
            continue
        add(line)
        if len(terms) >= 8:
            break

    return terms


def _charboxes_for_search(
    textpage: pdfium.PdfTextPage,
    term: str,
) -> list[list[float]]:
    rects: list[list[float]] = []
    searcher = textpage.search(term, match_case=False, match_whole_word=False)
    while True:
        hit = searcher.get_next()
        if hit is None:
            break
        start, count = hit
        for index in range(start, start + count):
            box = textpage.get_charbox(index)
            if box is None:
                continue
            left, bottom, right, top = box
            rects.append([float(left), float(bottom), float(right), float(top)])
    return rects


def highlights_from_pdf(
    pdf_path: Path,
    chunk: Chunk,
) -> list[dict[str, Any]]:
    """Build highlight rectangles by searching the PDF for chunk text."""
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        return []

    doc = pdfium.PdfDocument(str(pdf_path))
    highlights: list[dict[str, Any]] = []
    terms = _search_terms_for_chunk(chunk)
    page_count = len(doc)

    start_page = max(1, chunk.page_start)
    end_page = min(chunk.page_end if chunk.page_end > 0 else start_page, page_count)
    if start_page > page_count:
        return []

    for page_number in range(start_page, end_page + 1):
        page = doc[page_number - 1]
        textpage = page.get_textpage()
        page_rects: list[list[float]] = []

        for term in terms:
            try:
                page_rects.extend(_charboxes_for_search(textpage, term))
            except Exception:
                continue

        for rect in merge_rects(page_rects):
            highlights.append({"page": page_number, "rect": rect})

    return highlights


def highlights_from_spans(
    chunk: Chunk,
    spans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build highlight rectangles by matching Docling text spans to chunk content."""
    content_norm = normalize_for_match(chunk.content)
    title_norm = normalize_for_match(chunk.title)
    rects_by_page: dict[int, list[list[float]]] = {}

    for span in spans:
        page = int(span["page"])
        if page < chunk.page_start or page > chunk.page_end:
            continue

        span_norm = normalize_for_match(str(span.get("text", "")))
        if len(span_norm) < 3:
            continue

        if (
            span_norm in content_norm
            or span_norm in title_norm
            or title_norm in span_norm
        ):
            rects_by_page.setdefault(page, []).append(list(span["rect"]))

    highlights: list[dict[str, Any]] = []
    for page, rects in sorted(rects_by_page.items()):
        for rect in merge_rects(rects):
            highlights.append({"page": page, "rect": rect})

    return highlights


def attach_highlights_to_chunk(
    chunk: Chunk,
    *,
    pdf_path: Path,
    text_spans: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return highlight rectangles for a chunk using spans or PDF search."""
    if text_spans:
        highlights = highlights_from_spans(chunk, text_spans)
        if highlights:
            return highlights

    return highlights_from_pdf(pdf_path, chunk)


def attach_highlights_to_chunks(
    chunks: list[Chunk],
    *,
    pdf_path: Path,
    text_spans: list[dict[str, Any]] | None = None,
) -> list[Chunk]:
    updated: list[Chunk] = []
    for chunk in chunks:
        copy = chunk.model_copy(deep=True)
        copy.metadata["highlights"] = attach_highlights_to_chunk(
            copy,
            pdf_path=pdf_path,
            text_spans=text_spans,
        )
        updated.append(copy)
    return updated


def attach_highlights() -> None:
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


def main() -> None:
    attach_highlights()

