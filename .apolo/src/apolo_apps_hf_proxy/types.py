"""Type definitions for HuggingFace Proxy App."""

from apolo_app_types.protocols.common import AppInputs, AppOutputs
from apolo_app_types.protocols.common.hugging_face import HuggingFaceCache, HuggingFaceToken
from pydantic import Field


class HfProxyInputs(AppInputs):
    """Input configuration for HuggingFace Proxy deployment."""

    cache_config: HuggingFaceCache = Field(
        ...,
        description="Configuration for the HuggingFace cache storage",
    )

    token: HuggingFaceToken = Field(
        ...,
        description="HuggingFace API token for authentication",
    )


class HfProxyOutputs(AppOutputs):
    """Output information from HuggingFace Proxy deployment."""

    cache_config: HuggingFaceCache = Field(
        ...,
        description="Configuration for the HuggingFace cache storage",
    )

    token: HuggingFaceToken = Field(
        ...,
        description="HuggingFace API token",
    )

    internal_url: str = Field(
        ...,
        description="Internal service URL accessible within the cluster",
    )
