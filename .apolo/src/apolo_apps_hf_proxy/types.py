"""Type definitions for HuggingFace Proxy App."""

from apolo_app_types.protocols.common import AppInputs, AppOutputs
from apolo_app_types.protocols.common.hugging_face import (
    HuggingFaceModelDetailDynamic,
    HuggingFaceToken,
)
from apolo_app_types.protocols.common.schema_extra import SchemaExtraMetadata, SchemaMetaType
from apolo_app_types.protocols.common.storage import ApoloFilesPath
from pydantic import Field


class HfProxyInputs(AppInputs):
    """Input configuration for HuggingFace Proxy deployment."""

    files_path: ApoloFilesPath = Field(
        default=ApoloFilesPath(path="storage:.apps/hugging-face-cache"),
        json_schema_extra=SchemaExtraMetadata(
            description="The path to the Apolo Files directory where Hugging Face artifacts are cached.",  # noqa: E501
            title="Files Path",
        ).as_json_schema_extra(),
    )

    token: HuggingFaceToken


class HfProxyOutputs(AppOutputs):
    """Output information from HuggingFace Proxy deployment."""

    files_path: ApoloFilesPath

    token: HuggingFaceToken

    huggingface_models: list[HuggingFaceModelDetailDynamic] = Field(
        default_factory=list,
        json_schema_extra=SchemaExtraMetadata(
            title="HuggingFace Models",
            description="List of available HuggingFace models.",
            meta_type=SchemaMetaType.DYNAMIC,
        ).as_json_schema_extra(),
    )
