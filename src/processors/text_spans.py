"""Extract text spans with bounding boxes from Docling documents."""

from __future__ import annotations

from typing import Any

from docling_core.types.doc.base import CoordOrigin


def _page_sizes(doc: Any) -> dict[int, tuple[float, float]]:
    sizes: dict[int, tuple[float, float]] = {}
    for page_number, page in doc.pages.items():
        width = float(getattr(page.size, "width", 612.0))
        height = float(getattr(page.size, "height", 792.0))
        sizes[int(page_number)] = (width, height)
    return sizes


def _bbox_to_pdf_rect(bbox: Any, page_height: float) -> list[float]:
    """Convert a Docling bbox to PDF user-space [l, b, r, t]."""
    origin = getattr(bbox, "coord_origin", CoordOrigin.TOPLEFT)
    if origin != CoordOrigin.BOTTOMLEFT:
        bbox = bbox.to_bottom_left_origin(page_height)
    return [float(bbox.l), float(bbox.b), float(bbox.r), float(bbox.t)]


def extract_text_spans(doc: Any) -> list[dict[str, Any]]:
    """Extract page-level text spans with PDF coordinates from Docling output."""
    page_sizes = _page_sizes(doc)
    spans: list[dict[str, Any]] = []

    for item in doc.texts:
        text = (getattr(item, "text", "") or "").strip()
        if not text or not getattr(item, "prov", None):
            continue

        for prov in item.prov:
            page_no = int(prov.page_no)
            page_height = page_sizes.get(page_no, (612.0, 792.0))[1]
            rect = _bbox_to_pdf_rect(prov.bbox, page_height)
            spans.append(
                {
                    "page": page_no,
                    "text": text,
                    "rect": rect,
                }
            )

    return spans
