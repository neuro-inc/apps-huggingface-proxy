"""Pydantic models for API request/response schemas."""

from apolo_app_types.dynamic_outputs import DynamicAppIdResponse, DynamicAppListResponse
from apolo_app_types.protocols.common.hugging_face import HuggingFaceToken
from apolo_app_types.protocols.common.storage import ApoloFilesPath
from pydantic import BaseModel, Field


class HFModelDetail(BaseModel):
    """Detailed HuggingFace model representation."""

    id: str = Field(..., description="Repository identifier")
    name: str = Field(..., description="Model name")
    visibility: str = Field(..., description="Repository visibility")
    gated: bool = Field(False, description="Whether the model is gated")
    tags: list[str] = Field(default_factory=list, description="Repository tags")
    cached: bool = Field(False, description="Whether model is cached locally")
    last_modified: str | None = Field(None, description="Last modification timestamp")
    files_path: ApoloFilesPath | None = Field(None, description="Path to cached model files")
    hf_token: HuggingFaceToken | None = Field(None, description="HuggingFace token")


class HFModel(DynamicAppIdResponse):
    """HuggingFace model representation."""

    id: str = Field(..., description="Repository identifier")
    value: HFModelDetail = Field(..., description="Detailed model information")


class ModelResponse(BaseModel):
    """Response for single model endpoints."""

    status: str
    data: HFModelDetail | None = None


class ModelListResponse(DynamicAppListResponse):
    """Response for model list endpoints."""

    status: str
    data: list[HFModel] | None = None
