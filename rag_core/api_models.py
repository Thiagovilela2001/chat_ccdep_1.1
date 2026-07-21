"""Contratos HTTP compartilhados pelas engines RAG."""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)


class SourceInfo(BaseModel):
    file: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    page: str | int | None = None


class ValidationInfo(BaseModel):
    verified: int = Field(ge=0)
    total: int = Field(ge=0)
    unverified: list[str]


class CitationValidationInfo(BaseModel):
    verified: int = Field(ge=0)
    total: int = Field(ge=0)
    unverified: list[str]


class QueryResponse(BaseModel):
    answer: str = Field(min_length=1)
    sources_used: list[str]
    rewritten_query: str
    sources: list[SourceInfo]
    validation: ValidationInfo
    citation_validation: CitationValidationInfo
    rag_type: str
    rag_label: str
