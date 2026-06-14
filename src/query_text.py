"""Query text chunks indexed in Phase 11."""

import argparse

import chromadb

from src.azure_openai_client import embed_texts, get_azure_openai_client
from src.phases.phase11_index_text import COLLECTION_NAME
from src.utils import PDF_PATH, chroma_path, document_id_for_pdf

DEFAULT_QUERIES = [
    "How do I fix error 200 paper jam?",
    "How do I replace the toner cartridge?",
    "What causes gray background or toner fog?",
]


def get_collection(document_id: str) -> chromadb.Collection:
    path = chroma_path(document_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing index at {path}. Run phase 11 first."
        )
    chroma = chromadb.PersistentClient(path=str(path))
    return chroma.get_collection(COLLECTION_NAME)


def search_chunks(
    query: str,
    *,
    document_id: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    document_id = document_id or document_id_for_pdf(PDF_PATH)
    collection = get_collection(document_id)
    client = get_azure_openai_client()
    embedding = embed_texts(client, [query])[0]

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]

    for chunk_id, metadata, document, distance in zip(
        ids,
        metadatas,
        documents,
        distances,
    ):
        hits.append(
            {
                "chunk_id": chunk_id,
                "title": metadata["title"],
                "chunk_type": metadata["chunk_type"],
                "page_range": metadata["page_range"],
                "distance": distance,
                "snippet": document[:300],
            }
        )

    return hits


def print_hits(query: str, hits: list[dict]) -> None:
    print(f"\nQuery: {query}")
    print("-" * 60)
    if not hits:
        print("  (no results)")
        return

    for i, hit in enumerate(hits, start=1):
        print(
            f"  {i}. {hit['title']} "
            f"| p{hit['page_range']} "
            f"| {hit['chunk_type']} "
            f"| dist={hit['distance']:.4f}"
        )
        print(f"     {hit['snippet'][:200]}...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Phase 11 text chunk index"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query (omit to run default test queries)",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=3,
        help="Number of results per query (default: 3)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run default test queries",
    )
    args = parser.parse_args()

    document_id = document_id_for_pdf(PDF_PATH)
    collection = get_collection(document_id)
    print(f"Phase 11 index: {collection.count()} chunks in {COLLECTION_NAME}")

    queries = DEFAULT_QUERIES if args.test or not args.query else [args.query]
    for query in queries:
        hits = search_chunks(query, document_id=document_id, top_k=args.k)
        print_hits(query, hits)


if __name__ == "__main__":
    main()
