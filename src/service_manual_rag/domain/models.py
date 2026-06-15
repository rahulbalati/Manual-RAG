from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    PROCEDURE = "procedure"
    TROUBLESHOOTING = "troubleshooting"
    ERROR_CODE = "error_code"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class Section(BaseModel):
    section_id: str
    title: str
    level: int
    page_start: int | None = None
    page_end: int | None = None
    content: str = ""
    children: list["Section"] = Field(default_factory=list)


class Document(BaseModel):
    document_id: str
    source_file: Path
    title: str
    sections: list[Section] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_type: ChunkType
    title: str
    heading_path: list[str]
    content: str
    page_start: int
    page_end: int
    metadata: dict = Field(default_factory=dict)


Section.model_rebuild()
