"""Hybrid retrieval over text chunks."""

from dataclasses import dataclass

import chromadb

from service_manual_rag.clients.azure_openai import embed_texts, get_azure_openai_client
from service_manual_rag.config import get_settings
from service_manual_rag.indexing.text import COLLECTION_NAME as TEXT_COLLECTION
from service_manual_rag.retrieval.bm25_index import Bm25Index, get_text_bm25_index
from service_manual_rag.retrieval.rrf import reciprocal_rank_fusion
from service_manual_rag.storage.paths import chroma_path, document_id_for_pdf

CANDIDATE_MULTIPLIER = 3


@dataclass
class RetrievalResult:
    query: str
    chunks: list[dict]


def _get_chroma_client(document_id: str) -> chromadb.ClientAPI:
    path = chroma_path(document_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing index at {path}. Run the index-text step first."
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


def _fusion_distance(fusion_score: float) -> float:
    return 1.0 / (fusion_score + 1.0)


def _hit_from_bm25(
    bm25_index: Bm25Index,
    item_id: str,
    *,
    id_key: str,
    fusion_score: float,
) -> dict:
    record = bm25_index.get(item_id)
    return {
        id_key: item_id,
        "distance": _fusion_distance(fusion_score),
        "fusion_score": fusion_score,
        "snippet": record.text[:300],
        **record.metadata,
    }


def _apply_fusion_score(hit: dict, fusion_score: float) -> dict:
    updated = dict(hit)
    updated["fusion_score"] = fusion_score
    updated["distance"] = _fusion_distance(fusion_score)
    return updated


def _hybrid_query(
    query: str,
    *,
    collection: chromadb.Collection,
    embedding: list[float],
    bm25_index: Bm25Index,
    top_k: int,
    id_key: str,
    semantic_weight: float,
    keyword_weight: float,
    rrf_k: int,
) -> list[dict]:
    candidate_k = max(top_k * CANDIDATE_MULTIPLIER, top_k)

    semantic_hits = _query_collection(
        collection,
        embedding,
        top_k=candidate_k,
        id_key=id_key,
    )
    semantic_ids = [hit[id_key] for hit in semantic_hits]
    bm25_ids = [item_id for item_id, _ in bm25_index.search(query, top_k=candidate_k)]

    fused = reciprocal_rank_fusion(
        [semantic_ids, bm25_ids],
        weights=[semantic_weight, keyword_weight],
        k=rrf_k,
    )

    semantic_by_id = {hit[id_key]: hit for hit in semantic_hits}
    hits: list[dict] = []
    for item_id, fusion_score in fused[:top_k]:
        if item_id in semantic_by_id:
            hits.append(_apply_fusion_score(semantic_by_id[item_id], fusion_score))
        else:
            hits.append(
                _hit_from_bm25(
                    bm25_index,
                    item_id,
                    id_key=id_key,
                    fusion_score=fusion_score,
                )
            )
    return hits


def retrieve(
    query: str,
    *,
    document_id: str | None = None,
    top_k_chunks: int = 5,
) -> RetrievalResult:
    settings = get_settings()
    document_id = document_id or document_id_for_pdf(settings.default_pdf)
    chroma = _get_chroma_client(document_id)
    text_collection = chroma.get_collection(TEXT_COLLECTION)

    client = get_azure_openai_client()
    embedding = embed_texts(client, [query])[0]

    text_bm25 = get_text_bm25_index(document_id)

    if settings.hybrid_search_enabled:
        chunks = _hybrid_query(
            query,
            collection=text_collection,
            embedding=embedding,
            bm25_index=text_bm25,
            top_k=top_k_chunks,
            id_key="chunk_id",
            semantic_weight=settings.hybrid_semantic_weight,
            keyword_weight=settings.hybrid_keyword_weight,
            rrf_k=settings.hybrid_rrf_k,
        )
    else:
        chunks = _query_collection(
            text_collection,
            embedding,
            top_k=top_k_chunks,
            id_key="chunk_id",
        )

    return RetrievalResult(query=query, chunks=chunks)
