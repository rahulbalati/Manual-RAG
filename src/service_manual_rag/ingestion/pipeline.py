"""Ingestion pipeline orchestration."""

from collections.abc import Callable
from pathlib import Path

from service_manual_rag.config import get_settings

STEPS: list[tuple[str, str]] = [
    ("parse", "Parse PDF with Docling"),
    ("hierarchy", "Build section hierarchy"),
    ("pages", "Assign page numbers"),
    ("figures", "Extract figures"),
    ("associate", "Associate figures to sections"),
    ("procedures", "Detect procedures"),
    ("chunks", "Generate chunks"),
    ("metadata", "Enrich chunk metadata"),
    ("image-context", "Build figure context"),
    ("highlights", "Attach PDF highlights"),
    ("index-text", "Index text chunks"),
    ("index-figures", "Index figures"),
]


def _load_step_handlers() -> dict[str, Callable[[], None]]:
    from service_manual_rag.enrichment import (
        associate,
        chunks,
        highlights,
        image_context,
        metadata,
        procedures,
    )
    from service_manual_rag.indexing import figures as index_figures
    from service_manual_rag.indexing import text as index_text
    from service_manual_rag.ingestion import figures, hierarchy, pages, parse

    return {
        "parse": parse.main,
        "hierarchy": hierarchy.main,
        "pages": pages.main,
        "figures": figures.main,
        "associate": associate.main,
        "procedures": procedures.main,
        "chunks": chunks.main,
        "metadata": metadata.main,
        "image-context": image_context.main,
        "highlights": highlights.main,
        "index-text": index_text.main,
        "index-figures": index_figures.main,
    }


def run_pipeline(
    *,
    pdf: Path | None = None,
    step: str | None = None,
) -> None:
    if pdf is not None:
        get_settings().default_pdf = pdf.resolve()

    handlers = _load_step_handlers()

    if step:
        handlers[step]()
        return

    for step_name, _label in STEPS:
        handlers[step_name]()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the service manual ingestion pipeline",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Path to PDF (default: from config / DEFAULT_PDF)",
    )
    parser.add_argument(
        "--step",
        choices=[name for name, _ in STEPS],
        help="Run a single pipeline step only",
    )
    args = parser.parse_args()
    run_pipeline(pdf=args.pdf, step=args.step)


if __name__ == "__main__":
    main()
