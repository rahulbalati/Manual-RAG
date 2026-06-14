"""Chunk and text-span persistence."""

import json
from pathlib import Path

from service_manual_rag.domain.models import Chunk
from service_manual_rag.storage.paths import (
    chunks_path,
    default_document_id,
    text_spans_path,
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
        document_id = default_document_id()
    path = chunks_path(document_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the previous ingestion step first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk.model_validate(item) for item in payload]


def chunk_map(document_id: str | None = None) -> dict[str, Chunk]:
    return {chunk.chunk_id: chunk for chunk in load_chunks(document_id)}


def load_text_spans(document_id: str | None = None) -> list[dict]:
    if document_id is None:
        document_id = default_document_id()
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
