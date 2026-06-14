"""Index text chunks in Chroma with Azure OpenAI embeddings."""

import chromadb

from service_manual_rag.clients.azure_openai import embed_texts, get_azure_openai_client
from service_manual_rag.domain.models import Chunk
from service_manual_rag.config import get_settings
from service_manual_rag.storage import load_chunks
from service_manual_rag.storage.paths import chroma_path, document_id_for_pdf

COLLECTION_NAME = "text_chunks"


def build_embed_text(chunk: Chunk) -> str:
    meta = chunk.metadata
    hierarchy = meta.get("hierarchy_context") or " > ".join(
        chunk.heading_path
    )
    provenance = meta.get("provenance") or {}
    page_range = provenance.get(
        "page_range",
        f"{chunk.page_start}-{chunk.page_end}",
    )
    return (
        f"Title: {chunk.title}\n"
        f"Section: {hierarchy}\n"
        f"Type: {chunk.chunk_type.value}\n"
        f"Pages: {page_range}\n\n"
        f"{chunk.content}"
    )


def filter_indexable_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return [chunk for chunk in chunks if chunk.content.strip()]


def chunk_metadata(chunk: Chunk) -> dict[str, str | int]:
    provenance = chunk.metadata.get("provenance") or {}
    return {
        "document_id": chunk.document_id,
        "chunk_type": chunk.chunk_type.value,
        "title": chunk.title,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "page_range": str(
            provenance.get(
                "page_range",
                f"{chunk.page_start}-{chunk.page_end}",
            )
        ),
        "source_document": str(
            chunk.metadata.get("source_document", "")
        ),
        "image_ids": ",".join(chunk.image_ids),
        "word_count": int(chunk.metadata.get("word_count", 0)),
    }


def index_chunks(
    chunks: list[Chunk],
    document_id: str,
) -> tuple[int, int, str]:
    indexable = filter_indexable_chunks(chunks)
    skipped = len(chunks) - len(indexable)
    if not indexable:
        raise ValueError("No indexable chunks with non-empty content.")

    texts = [build_embed_text(chunk) for chunk in indexable]
    client = get_azure_openai_client()
    embeddings = embed_texts(client, texts)

    db_path = chroma_path(document_id)
    db_path.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(db_path))
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.upsert(
        ids=[chunk.chunk_id for chunk in indexable],
        embeddings=embeddings,
        documents=texts,
        metadatas=[chunk_metadata(chunk) for chunk in indexable],
    )

    return len(indexable), skipped, str(db_path)


def main() -> None:
    print("Indexing text chunks in Chroma")
    chunks = load_chunks()
    document_id = chunks[0].document_id if chunks else document_id_for_pdf(
        get_settings().default_pdf
    )
    indexed, skipped, path = index_chunks(chunks, document_id)

    print(f"  total chunks: {len(chunks)}")
    print(f"  indexed: {indexed}")
    print(f"  skipped (empty content): {skipped}")
    print(f"  collection: {COLLECTION_NAME}")
    print(f"  chroma path: {path}")

