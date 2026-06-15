"""Generate RAG answers with Azure OpenAI chat."""

from dataclasses import dataclass

from service_manual_rag.clients.azure_openai import chat_completion, chat_completion_stream
from service_manual_rag.config import get_settings
from service_manual_rag.retrieval.search import RetrievalResult, retrieve
from service_manual_rag.storage import load_chunks
from service_manual_rag.storage.paths import document_id_for_pdf

MAX_CHUNK_CHARS = 4000

SYSTEM_PROMPT = """You are a technical service manual assistant for the MX-B468P printer.
Answer the user's question using ONLY the retrieved manual excerpts provided.
Always cite page numbers when referencing information.
If the excerpts do not contain enough information, say you don't know.
Be concise and practical for a field technician."""


@dataclass
class AnswerResult:
    query: str
    answer: str
    chunks: list[dict]


def _chunk_content_map(document_id: str) -> dict[str, str]:
    return {chunk.chunk_id: chunk.content for chunk in load_chunks(document_id)}


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

    return "\n".join(parts)


def build_messages(query: str, context: str) -> list[dict]:
    user_text = (
        f"Question: {query}\n\n"
        f"Manual excerpts:\n{context}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def prepare_answer(
    query: str,
    *,
    document_id: str | None = None,
    top_k_chunks: int = 5,
) -> tuple[str, RetrievalResult, list[dict]]:
    document_id = document_id or document_id_for_pdf(get_settings().default_pdf)
    result = retrieve(
        query,
        document_id=document_id,
        top_k_chunks=top_k_chunks,
    )
    context = build_context(result, document_id=document_id)
    messages = build_messages(query, context)
    return document_id, result, messages


def answer(
    query: str,
    *,
    document_id: str | None = None,
    top_k_chunks: int = 5,
) -> AnswerResult:
    _document_id, result, messages = prepare_answer(
        query,
        document_id=document_id,
        top_k_chunks=top_k_chunks,
    )
    response = chat_completion(messages)
    return AnswerResult(
        query=query,
        answer=response,
        chunks=result.chunks,
    )
