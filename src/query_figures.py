"""Query figures indexed in Phase 12."""

import argparse

import chromadb

from src.azure_openai_client import embed_texts, get_azure_openai_client
from src.phases.phase12_index_figures import COLLECTION_NAME
from src.utils import PDF_PATH, chroma_path, document_id_for_pdf

DEFAULT_QUERIES = [
    "security jumper location on controller board",
    "gray background toner fog print quality",
    "paper jam sensor diagram",
]


def get_collection(document_id: str) -> chromadb.Collection:
    path = chroma_path(document_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing index at {path}. Run phase 12 first."
        )
    chroma = chromadb.PersistentClient(path=str(path))
    return chroma.get_collection(COLLECTION_NAME)


def search_figures(
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

    for figure_id, metadata, document, distance in zip(
        ids,
        metadatas,
        documents,
        distances,
    ):
        hits.append(
            {
                "figure_id": figure_id,
                "procedure_title": metadata["procedure_title"],
                "page_number": metadata["page_number"],
                "image_path": metadata["image_path"],
                "heading_path": metadata["heading_path"],
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
            f"  {i}. {hit['procedure_title']} "
            f"| p{hit['page_number']} "
            f"| dist={hit['distance']:.4f}"
        )
        print(f"     image: {hit['image_path']}")
        print(f"     {hit['snippet'][:200]}...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Phase 12 figure index"
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
    print(
        f"Phase 12 index: {collection.count()} figures in {COLLECTION_NAME}"
    )

    queries = DEFAULT_QUERIES if args.test or not args.query else [args.query]
    for query in queries:
        hits = search_figures(query, document_id=document_id, top_k=args.k)
        print_hits(query, hits)


if __name__ == "__main__":
    main()
