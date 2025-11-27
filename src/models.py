"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field


class HFModel(BaseModel):
    """HuggingFace model representation."""

    id: str = Field(..., description="Repository identifier")
    visibility: str = Field(..., description="Repository visibility")
    gated: bool = Field(False, description="Whether the model is gated")
    tags: list[str] = Field(default_factory=list, description="Repository tags")
    cached: bool = Field(False, description="Whether model is cached locally")
    last_modified: str | None = Field(None, description="Last modification timestamp")


class ModelResponse(BaseModel):
    """Response for single model endpoints."""

    status: str
    data: HFModel | None = None


class HFModelsData(BaseModel):
    """Container for HuggingFace models list."""

    hf_models: list[dict[str, HFModel]] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    """Response for model list endpoints."""

    status: str
    data: HFModelsData | None = None
