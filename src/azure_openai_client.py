"""Shared Azure OpenAI client for embeddings and chat."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

EMBED_BATCH_SIZE = 16


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Missing {name}. Copy .env.example to .env and fill in values."
        )
    return value


def get_azure_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_require_env("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get(
            "AZURE_OPENAI_API_VERSION",
            "2024-02-15-preview",
        ),
    )


def embedding_deployment() -> str:
    return _require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


def chat_deployment() -> str:
    return _require_env("AZURE_OPENAI_CHAT_DEPLOYMENT")


def chat_completion(
    messages: list[dict],
    *,
    deployment: str | None = None,
    temperature: float = 0.2,
) -> str:
    client = get_azure_openai_client()
    response = client.chat.completions.create(
        model=deployment or chat_deployment(),
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def chat_completion_stream(
    messages: list[dict],
    *,
    deployment: str | None = None,
    temperature: float = 0.2,
):
    client = get_azure_openai_client()
    stream = client.chat.completions.create(
        model=deployment or chat_deployment(),
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def embed_texts(
    client: AzureOpenAI,
    texts: list[str],
    *,
    deployment: str | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    deployment = deployment or embedding_deployment()
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        response = client.embeddings.create(
            model=deployment,
            input=batch,
        )
        embeddings.extend(
            item.embedding
            for item in sorted(response.data, key=lambda d: d.index)
        )

    return embeddings
