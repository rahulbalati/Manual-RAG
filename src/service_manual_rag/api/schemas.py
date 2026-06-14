from pydantic import BaseModel, Field


class HighlightRect(BaseModel):
    page: int
    rect: list[float] = Field(min_length=4, max_length=4)


class ChunkSource(BaseModel):
    chunk_id: str
    title: str
    page_start: int
    page_end: int
    page_range: str
    snippet: str = ""
    chunk_type: str
    distance: float
    pdf_url: str = ""
    highlights: list[HighlightRect] = Field(default_factory=list)


class FigureSource(BaseModel):
    figure_id: str
    procedure_title: str
    page_number: int
    page_start: int
    page_end: int
    image_path: str
    image_url: str
    heading_path: str
    distance: float
    pdf_url: str = ""


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    document_id: str | None = None
    chunk_k: int = Field(default=3, ge=1, le=20)
    figure_k: int = Field(default=3, ge=1, le=20)


class RetrieveResponse(BaseModel):
    query: str
    document_id: str
    chunks: list[ChunkSource]
    figures: list[FigureSource]


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    document_id: str | None = None
    chunk_k: int = Field(default=3, ge=1, le=20)
    figure_k: int = Field(default=2, ge=0, le=20)
    include_images: bool = False


class Sources(BaseModel):
    chunks: list[ChunkSource]
    figures: list[FigureSource]


class AskResponse(BaseModel):
    query: str
    document_id: str
    answer: str
    sources: Sources


class DocumentInfo(BaseModel):
    document_id: str
    title: str
    indexed: bool
    pdf_url: str = ""


class HealthResponse(BaseModel):
    status: str
    documents: int
    default_document_id: str | None
    index_ready: bool
    pdf_url: str | None = None
