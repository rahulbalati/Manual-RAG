"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from service_manual_rag.api.routes import api_router
from service_manual_rag.config import get_settings

app = FastAPI(
    title="Service Manual RAG API",
    description="Retrieval and Q&A over technical service manuals",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()

if settings.assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=settings.assets_dir), name="assets")

if settings.frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=settings.frontend_dir), name="ui")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(settings.frontend_dir / "index.html")


app.include_router(api_router)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "service_manual_rag.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
