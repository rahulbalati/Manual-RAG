# Service Manual RAG

Multimodal retrieval-augmented generation over technical service manual PDFs.

## Project structure

```
service-manual-rag/
├── pyproject.toml
├── README.md
├── .env.example
│
├── src/
│   └── service_manual_rag/
│       ├── config.py                # Settings via pydantic-settings
│       ├── domain/                  # Models and section tree utilities
│       ├── clients/                 # Azure OpenAI, Docling
│       ├── storage/                 # Persistence and path resolution
│       ├── ingestion/               # PDF → structured document + pipeline
│       ├── enrichment/              # Document → chunks + metadata + highlights
│       ├── indexing/                # Chroma vector indexing
│       ├── retrieval/               # Unified search
│       ├── generation/              # RAG answer generation
│       ├── processors/              # text_spans, error_table_split
│       └── api/                     # FastAPI app, routes, schemas
│
├── frontend/                        # Web UI (served at /ui)
├── data/
│   ├── raw/                         # Source PDFs
│   ├── processed/                   # Parsed output (gitignored)
│   └── index/                       # Vector stores (gitignored)
└── assets/                          # Extracted figures (gitignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env   # fill in Azure OpenAI credentials
```

## Run the server

```bash
python -m service_manual_rag.api.app
```

Open the UI at http://localhost:8000 (or http://localhost:8000/ui).

## Ingest a manual

Run the ingestion pipeline before querying. From Python:

```python
from pathlib import Path
from service_manual_rag.ingestion.pipeline import run_pipeline

run_pipeline(pdf=Path("data/raw/MX-B468P-Service-Manual.pdf"))
```

## Pipeline steps

| Step | Module |
|------|--------|
| Parse PDF | `ingestion/parse.py` |
| Build hierarchy | `ingestion/hierarchy.py` |
| Assign pages | `ingestion/pages.py` |
| Extract figures | `ingestion/figures.py` |
| Associate figures | `enrichment/associate.py` |
| Detect procedures | `enrichment/procedures.py` |
| Generate chunks | `enrichment/chunks.py` |
| Enrich metadata | `enrichment/metadata.py` |
| Figure context | `enrichment/image_context.py` |
| PDF highlights | `enrichment/highlights.py` |
| Index text | `indexing/text.py` |
| Index figures | `indexing/figures.py` |

Run a single step with `run_pipeline(step="chunks")`.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/documents` | List ingested documents |
| GET | `/documents/{id}/pdf` | Serve source PDF |
| POST | `/retrieve` | Vector search |
| POST | `/ask` | RAG answer |
| POST | `/ask/stream` | Streaming answer (SSE) |

## Configuration

Settings are loaded from environment variables and `.env` via `pydantic-settings`:

| Variable | Default |
|----------|---------|
| `DEFAULT_PDF` | `data/raw/MX-B468P-Service-Manual.pdf` |
| `PROCESSED_DIR` | `data/processed` |
| `ASSETS_DIR` | `assets` |
| `INDEX_DIR` | `data/index` |

Azure OpenAI credentials: see `.env.example`.
