"""In-memory BM25 indexes for keyword retrieval."""

from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from service_manual_rag.indexing.figures import (
    build_embed_text as build_figure_embed_text,
    figure_metadata,
    filter_indexable_figures,
)
from service_manual_rag.indexing.text import (
    build_embed_text as build_chunk_embed_text,
    chunk_metadata,
    filter_indexable_chunks,
)
from service_manual_rag.storage import load_chunks, load_document
from service_manual_rag.storage.paths import chunks_path, document_path


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _cache_key(document_id: str, source_path: Path) -> tuple[str, float]:
    mtime = source_path.stat().st_mtime if source_path.exists() else 0.0
    return (document_id, mtime)


@dataclass(frozen=True)
class Bm25Record:
    item_id: str
    text: str
    metadata: dict


@dataclass
class Bm25Index:
    records: dict[str, Bm25Record]
    _bm25: BM25Okapi
    _ids: list[str]

    def search(self, query: str, *, top_k: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self._ids, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return [(item_id, score) for item_id, score in ranked[:top_k]]

    def get(self, item_id: str) -> Bm25Record:
        return self.records[item_id]


def _build_index(records: list[Bm25Record]) -> Bm25Index:
    ids = [record.item_id for record in records]
    tokenized = [_tokenize(record.text) for record in records]
    return Bm25Index(
        records={record.item_id: record for record in records},
        _bm25=BM25Okapi(tokenized),
        _ids=ids,
    )


_text_indexes: dict[tuple[str, float], Bm25Index] = {}
_figure_indexes: dict[tuple[str, float], Bm25Index] = {}


def get_text_bm25_index(document_id: str) -> Bm25Index:
    key = _cache_key(document_id, chunks_path(document_id))
    cached = _text_indexes.get(key)
    if cached is not None:
        return cached

    chunks = filter_indexable_chunks(load_chunks(document_id))
    records = [
        Bm25Record(
            item_id=chunk.chunk_id,
            text=build_chunk_embed_text(chunk),
            metadata=chunk_metadata(chunk),
        )
        for chunk in chunks
    ]
    index = _build_index(records)
    _text_indexes[key] = index
    return index


def get_figure_bm25_index(document_id: str) -> Bm25Index:
    key = _cache_key(document_id, document_path(document_id))
    cached = _figure_indexes.get(key)
    if cached is not None:
        return cached

    document = load_document(document_id)
    figures = filter_indexable_figures(document.figures)
    records = [
        Bm25Record(
            item_id=figure.figure_id,
            text=build_figure_embed_text(figure),
            metadata=figure_metadata(figure, document.document_id),
        )
        for figure in figures
    ]
    index = _build_index(records)
    _figure_indexes[key] = index
    return index
