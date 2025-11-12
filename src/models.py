"""Pydantic models for API request/response schemas."""

from typing import Any

from pydantic import BaseModel, Field


class HFModel(BaseModel):
    """HuggingFace model representation."""

    repo_id: str = Field(..., description="Repository identifier")
    visibility: str = Field(..., description="Repository visibility")
    gated: bool = Field(False, description="Whether the model is gated")
    tags: list[str] = Field(default_factory=list, description="Repository tags")
    cached: bool = Field(False, description="Whether model is cached locally")
    last_modified: str | None = Field(None, description="Last modification timestamp")


class IdResponse(BaseModel):
    """Individual item in list response."""

    id: str
    value: HFModel


class ListResponse(BaseModel):
    """Response for list endpoints."""

    status: str
    data: list[IdResponse] | None = None


class ModelResponse(BaseModel):
    """Response for single model endpoints."""

    status: str
    data: HFModel | None = None


class BasicResponse(BaseModel):
    """Basic response for health checks."""

    status: str
    data: dict[str, Any] | None = None


class FilterParams(BaseModel):
    """Query parameters for filtering."""

    filter: str | None = Field(None, description="Filter query")
    limit: int = Field(100, gt=0, le=100, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Offset for pagination")
