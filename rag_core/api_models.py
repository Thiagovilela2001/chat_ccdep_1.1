"""Contratos HTTP compartilhados pelas engines RAG."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4_000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    conversation_id: str | None = Field(
        default=None,
        min_length=16,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)


class SourceInfo(BaseModel):
    file: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    page: str | int | None = None
    excerpt: str = Field(default="", max_length=4_000)


class ValidationInfo(BaseModel):
    verified: int = Field(ge=0)
    total: int = Field(ge=0)
    unverified: list[str]


class CitationValidationInfo(BaseModel):
    verified: int = Field(ge=0)
    total: int = Field(ge=0)
    unverified: list[str]


class NumericCitationInfo(BaseModel):
    value: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    source_index: int = Field(ge=0)
    file: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    page: str | int | None = None
    snippet: str = Field(default="", max_length=1_000)
    content_type: str = Field(default="text", min_length=1, max_length=32)
    claim: str = Field(default="", max_length=1_000)
    explanation: str = Field(default="", max_length=400)


class QueryResponse(BaseModel):
    answer: str = Field(min_length=1)
    sources_used: list[str]
    rewritten_query: str
    sources: list[SourceInfo]
    validation: ValidationInfo
    citation_validation: CitationValidationInfo
    numeric_citations: list[NumericCitationInfo] = Field(default_factory=list)
    rag_type: str
    rag_label: str
    conversation_id: str
    memory_turns: int = Field(default=0, ge=0)
