"""Phase 13: Unified retrieval across text chunks and figures."""

import argparse
from dataclasses import dataclass

import chromadb

from src.azure_openai_client import embed_texts, get_azure_openai_client
from src.phases.phase11_index_text import COLLECTION_NAME as TEXT_COLLECTION
from src.phases.phase12_index_figures import COLLECTION_NAME as FIGURE_COLLECTION
from src.utils import PDF_PATH, chroma_path, document_id_for_pdf

DEFAULT_QUERIES = [
    "How do I fix error 200 paper jam?",
    "What causes gray background or toner fog?",
    "paper jam sensor diagram",
]


@dataclass
class RetrievalResult:
    query: str
    chunks: list[dict]
    figures: list[dict]


def _get_chroma_client(document_id: str) -> chromadb.ClientAPI:
    path = chroma_path(document_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing index at {path}. Run phases 11 and 12 first."
        )
    return chromadb.PersistentClient(path=str(path))


def _query_collection(
    collection: chromadb.Collection,
    embedding: list[float],
    *,
    top_k: int,
    id_key: str,
) -> list[dict]:
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    for item_id, metadata, document, distance in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0],
    ):
        hit = {
            id_key: item_id,
            "distance": distance,
            "snippet": document[:300],
            **metadata,
        }
        hits.append(hit)

    return hits


def _dedupe_figures(figures: list[dict]) -> list[dict]:
    seen: set[tuple[str, int]] = set()
    deduped: list[dict] = []
    for figure in sorted(figures, key=lambda f: f["distance"]):
        key = (figure["procedure_title"], figure["page_number"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(figure)
    return deduped


def retrieve(
    query: str,
    *,
    document_id: str | None = None,
    top_k_chunks: int = 5,
    top_k_figures: int = 5,
    dedupe_figures: bool = True,
) -> RetrievalResult:
    document_id = document_id or document_id_for_pdf(PDF_PATH)
    chroma = _get_chroma_client(document_id)
    text_collection = chroma.get_collection(TEXT_COLLECTION)
    figure_collection = chroma.get_collection(FIGURE_COLLECTION)

    client = get_azure_openai_client()
    embedding = embed_texts(client, [query])[0]

    chunks = _query_collection(
        text_collection,
        embedding,
        top_k=top_k_chunks,
        id_key="chunk_id",
    )
    figures = _query_collection(
        figure_collection,
        embedding,
        top_k=top_k_figures,
        id_key="figure_id",
    )
    if dedupe_figures:
        figures = _dedupe_figures(figures)

    return RetrievalResult(query=query, chunks=chunks, figures=figures)


def print_result(result: RetrievalResult) -> None:
    print(f"\nQuery: {result.query}")
    print("=" * 60)

    print(f"\nText chunks ({len(result.chunks)}):")
    print("-" * 60)
    if not result.chunks:
        print("  (no results)")
    for i, chunk in enumerate(result.chunks, start=1):
        print(
            f"  {i}. {chunk['title']} "
            f"| p{chunk['page_range']} "
            f"| {chunk['chunk_type']} "
            f"| dist={chunk['distance']:.4f}"
        )
        print(f"     {chunk['snippet'][:200]}...")

    print(f"\nFigures ({len(result.figures)}):")
    print("-" * 60)
    if not result.figures:
        print("  (no results)")
    for i, figure in enumerate(result.figures, start=1):
        print(
            f"  {i}. {figure['procedure_title']} "
            f"| p{figure['page_number']} "
            f"| dist={figure['distance']:.4f}"
        )
        print(f"     image: {figure['image_path']}")
        print(f"     {figure['snippet'][:200]}...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified retrieval over text chunks and figures"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query (omit to run default test queries)",
    )
    parser.add_argument(
        "--chunk-k",
        type=int,
        default=3,
        help="Number of text chunks to retrieve (default: 3)",
    )
    parser.add_argument(
        "--figure-k",
        type=int,
        default=3,
        help="Number of figures to retrieve (default: 3)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run default test queries",
    )
    args = parser.parse_args()

    document_id = document_id_for_pdf(PDF_PATH)
    chroma = _get_chroma_client(document_id)
    text_count = chroma.get_collection(TEXT_COLLECTION).count()
    figure_count = chroma.get_collection(FIGURE_COLLECTION).count()
    print(
        f"Phase 13: {text_count} text chunks, "
        f"{figure_count} figures"
    )

    queries = DEFAULT_QUERIES if args.test or not args.query else [args.query]
    for query in queries:
        result = retrieve(
            query,
            document_id=document_id,
            top_k_chunks=args.chunk_k,
            top_k_figures=args.figure_k,
        )
        print_result(result)


if __name__ == "__main__":
    main()
