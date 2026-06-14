"""Index figures in Chroma with Azure OpenAI embeddings."""

import chromadb

from service_manual_rag.clients.azure_openai import embed_texts, get_azure_openai_client
from service_manual_rag.domain.models import Document, Figure
from service_manual_rag.storage import load_document
from service_manual_rag.storage.paths import chroma_path

COLLECTION_NAME = "figure_chunks"


def build_embed_text(figure: Figure) -> str:
    hierarchy = " > ".join(figure.heading_path)
    return (
        f"Page: {figure.page_number}\n"
        f"Section: {hierarchy}\n\n"
        f"{figure.context_text}"
    )


def filter_indexable_figures(figures: list[Figure]) -> list[Figure]:
    return [figure for figure in figures if figure.context_text.strip()]


def figure_metadata(
    figure: Figure,
    document_id: str,
) -> dict[str, str | int]:
    return {
        "document_id": document_id,
        "figure_id": figure.figure_id,
        "page_number": figure.page_number,
        "image_path": str(figure.image_path),
        "heading_path": " > ".join(figure.heading_path),
        "procedure_title": (
            figure.heading_path[-1] if figure.heading_path else ""
        ),
    }


def index_figures(
    document: Document,
) -> tuple[int, int, str]:
    figures = document.figures
    indexable = filter_indexable_figures(figures)
    skipped = len(figures) - len(indexable)
    if not indexable:
        raise ValueError("No indexable figures with context_text.")

    texts = [build_embed_text(figure) for figure in indexable]
    client = get_azure_openai_client()
    embeddings = embed_texts(client, texts)

    db_path = chroma_path(document.document_id)
    db_path.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(db_path))
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.upsert(
        ids=[figure.figure_id for figure in indexable],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            figure_metadata(figure, document.document_id)
            for figure in indexable
        ],
    )

    return len(indexable), skipped, str(db_path)


def main() -> None:
    print("Indexing figures in Chroma")
    document = load_document()
    indexed, skipped, path = index_figures(document)

    print(f"  total figures: {len(document.figures)}")
    print(f"  indexed: {indexed}")
    print(f"  skipped (no context): {skipped}")
    print(f"  collection: {COLLECTION_NAME}")
    print(f"  chroma path: {path}")
    if document.figures:
        sample = next(
            (f for f in document.figures if f.context_text.strip()),
            document.figures[0],
        )
        print(f"  sample: {sample.figure_id} | p{sample.page_number}")
        print(f"    {sample.heading_path[-1] if sample.heading_path else '(no heading)'}")

