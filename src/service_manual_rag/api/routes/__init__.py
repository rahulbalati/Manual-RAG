from fastapi import APIRouter

from service_manual_rag.api.routes import ask, documents, health, retrieve

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(retrieve.router)
api_router.include_router(ask.router)

__all__ = ["api_router"]
