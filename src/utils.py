import hashlib
import json
from pathlib import Path

from src.domain.models import Chunk, Document, Section

PDF_PATH = Path("data/raw/MX-B468P-Service-Manual.pdf")
PROCESSED_DIR = Path("data/processed")
ASSETS_DIR = Path("assets")
INDEX_DIR = Path("data/index")


def stable_id(*parts: str) -> str:
    joined = "::".join(parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def document_id_for_pdf(pdf: Path) -> str:
    pdf = pdf.resolve()
    return stable_id(
        pdf.name,
        str(pdf.stat().st_size),
        str(pdf.stat().st_mtime),
    )


def output_dir(document_id: str) -> Path:
    return PROCESSED_DIR / document_id


def document_path(document_id: str) -> Path:
    return output_dir(document_id) / "document.json"


def chunks_path(document_id: str) -> Path:
    return output_dir(document_id) / "chunks.json"


def text_spans_path(document_id: str) -> Path:
    return output_dir(document_id) / "text_spans.json"


def chroma_path(document_id: str) -> Path:
    return INDEX_DIR / document_id / "chroma"


def save_document(document: Document) -> Path:
    path = document_path(document.document_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        document.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return path


def load_document(document_id: str | None = None) -> Document:
    if document_id is None:
        document_id = document_id_for_pdf(PDF_PATH)
    path = document_path(document_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the previous phase first."
        )
    return Document.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def save_chunks(chunks: list[Chunk], document_id: str) -> Path:
    path = chunks_path(document_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [chunk.model_dump() for chunk in chunks]
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return path


def load_chunks(document_id: str | None = None) -> list[Chunk]:
    if document_id is None:
        document_id = document_id_for_pdf(PDF_PATH)
    path = chunks_path(document_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the previous phase first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk.model_validate(item) for item in payload]


def load_text_spans(document_id: str | None = None) -> list[dict]:
    if document_id is None:
        document_id = document_id_for_pdf(PDF_PATH)
    path = text_spans_path(document_id)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_text_spans(
    spans: list[dict],
    document_id: str,
) -> Path:
    path = text_spans_path(document_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spans, indent=2), encoding="utf-8")
    return path


def pdf_path_for_document(document_id: str | None = None) -> Path:
    document = load_document(document_id)
    source = Path(document.source_file)
    if source.exists():
        return source
    if PDF_PATH.exists():
        return PDF_PATH.resolve()
    raise FileNotFoundError(
        f"PDF not found for document {document.document_id}."
    )


def chunk_map(document_id: str | None = None) -> dict[str, Chunk]:
    return {chunk.chunk_id: chunk for chunk in load_chunks(document_id)}


def flatten_sections(sections: list[Section]) -> list[Section]:
    result: list[Section] = []
    for section in sections:
        result.append(section)
        result.extend(flatten_sections(section.children))
    return result


def heading_path(
    target: Section,
    roots: list[Section],
) -> list[str]:
    path: list[str] = []

    def dfs(node: Section, current: list[str]) -> bool:
        current = [*current, node.title]
        if node.section_id == target.section_id:
            path.extend(current)
            return True
        for child in node.children:
            if dfs(child, current):
                return True
        return False

    for root in roots:
        if dfs(root, []):
            break

    return path
