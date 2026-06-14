"""Generate RAG answers with Azure OpenAI chat."""

import base64
from dataclasses import dataclass
from pathlib import Path

from service_manual_rag.clients.azure_openai import chat_completion, chat_completion_stream
from service_manual_rag.config import get_settings
from service_manual_rag.retrieval.search import RetrievalResult, retrieve
from service_manual_rag.storage import load_chunks, load_document

MAX_CHUNK_CHARS = 4000
MAX_FIGURE_CHARS = 2000

SYSTEM_PROMPT = """You are a technical service manual assistant for the MX-B468P printer.
Answer the user's question using ONLY the retrieved manual excerpts and figures provided.
Always cite page numbers when referencing information.
If the excerpts do not contain enough information, say you don't know.
Be concise and practical for a field technician."""


@dataclass
class AnswerResult:
    query: str
    answer: str
    chunks: list[dict]
    figures: list[dict]


def _chunk_content_map(document_id: str) -> dict[str, str]:
    return {chunk.chunk_id: chunk.content for chunk in load_chunks(document_id)}


def _figure_context_map(document_id: str) -> dict[str, str]:
    document = load_document(document_id)
    return {
        figure.figure_id: figure.context_text
        for figure in document.figures
    }


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def build_context(
    result: RetrievalResult,
    *,
    document_id: str,
) -> str:
    chunk_map = _chunk_content_map(document_id)
    figure_map = _figure_context_map(document_id)
    parts: list[str] = []

    if result.chunks:
        parts.append("## Retrieved text excerpts")
        for i, chunk in enumerate(result.chunks, start=1):
            content = _truncate(
                chunk_map.get(chunk["chunk_id"], chunk["snippet"]),
                MAX_CHUNK_CHARS,
            )
            parts.append(
                f"\n### [{i}] {chunk['title']}\n"
                f"Pages: {chunk['page_range']}\n"
                f"Type: {chunk['chunk_type']}\n"
                f"{content}"
            )

    if result.figures:
        parts.append("\n## Retrieved figures")
        for i, figure in enumerate(result.figures, start=1):
            context = _truncate(
                figure_map.get(figure["figure_id"], figure["snippet"]),
                MAX_FIGURE_CHARS,
            )
            parts.append(
                f"\n### [Figure {i}] {figure['procedure_title']}\n"
                f"Page: {figure['page_number']}\n"
                f"Image path: {figure['image_path']}\n"
                f"{context}"
            )

    return "\n".join(parts)


def _resolve_image_path(image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return get_settings().project_root / path


def _image_data_url(image_path: str) -> str | None:
    path = _resolve_image_path(image_path)
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_messages(
    query: str,
    context: str,
    figures: list[dict],
    *,
    include_images: bool,
) -> list[dict]:
    user_text = (
        f"Question: {query}\n\n"
        f"Manual excerpts:\n{context}"
    )
    content: list[dict] = [{"type": "text", "text": user_text}]

    if include_images:
        for figure in figures[:2]:
            data_url = _image_data_url(figure["image_path"])
            if data_url:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    }
                )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def prepare_answer(
    query: str,
    *,
    document_id: str | None = None,
    top_k_chunks: int = 5,
    top_k_figures: int = 3,
    include_images: bool = False,
) -> tuple[str, RetrievalResult, list[dict]]:
    from service_manual_rag.config import get_settings
    from service_manual_rag.storage.paths import document_id_for_pdf

    document_id = document_id or document_id_for_pdf(get_settings().default_pdf)
    result = retrieve(
        query,
        document_id=document_id,
        top_k_chunks=top_k_chunks,
        top_k_figures=top_k_figures,
    )
    context = build_context(result, document_id=document_id)
    messages = build_messages(
        query,
        context,
        result.figures,
        include_images=include_images,
    )
    return document_id, result, messages


def answer(
    query: str,
    *,
    document_id: str | None = None,
    top_k_chunks: int = 5,
    top_k_figures: int = 3,
    include_images: bool = False,
) -> AnswerResult:
    _document_id, result, messages = prepare_answer(
        query,
        document_id=document_id,
        top_k_chunks=top_k_chunks,
        top_k_figures=top_k_figures,
        include_images=include_images,
    )
    response = chat_completion(messages)
    return AnswerResult(
        query=query,
        answer=response,
        chunks=result.chunks,
        figures=result.figures,
        )
