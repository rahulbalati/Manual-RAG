"""Phase 5: Extract images from PDF."""

from pathlib import Path

from PIL import Image

from src.docling_client import create_pdf_converter
from src.domain.models import Document, Figure
from src.utils import ASSETS_DIR, load_document, save_document, stable_id


def _save_pil_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(path, format="PNG")


def extract_figures(
    document: Document,
    pdf_path: Path,
) -> Document:
    converter = create_pdf_converter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    figures: list[Figure] = []
    image_counter = 0

    for picture in doc.pictures:
        if not picture.prov:
            continue

        page_number = int(picture.prov[0].page_no)
        pil_image = picture.get_image(doc)
        if pil_image is None:
            continue

        image_counter += 1
        image_dir = ASSETS_DIR / document.document_id / "images"
        image_path = (
            image_dir / f"page_{page_number}_img_{image_counter}.png"
        )
        _save_pil_image(pil_image, image_path)

        figures.append(
            Figure(
                figure_id=stable_id(
                    document.document_id,
                    str(page_number),
                    str(image_counter),
                ),
                page_number=page_number,
                image_path=image_path,
            )
        )

    document.figures = figures
    return document


def main() -> None:
    print("Phase 5: Extracting figures")
    document = load_document()
    document = extract_figures(
        document,
        document.source_file,
    )
    path = save_document(document)

    print(f"  figures: {len(document.figures)}")
    print(f"  saved: {path}")
    for figure in document.figures[:5]:
        print(
            f"    {figure.figure_id} "
            f"p{figure.page_number} "
            f"{figure.image_path}"
        )
    if len(document.figures) > 5:
        print(f"    ... and {len(document.figures) - 5} more")


if __name__ == "__main__":
    main()
