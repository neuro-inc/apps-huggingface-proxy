"""Type definitions for HuggingFace Proxy App."""

import typing as t

from apolo_app_types.protocols.common import AbstractAppFieldType, AppInputs, AppOutputs
from apolo_app_types.protocols.common.hugging_face import HuggingFaceToken
from apolo_app_types.protocols.common.schema_extra import SchemaExtraMetadata, SchemaMetaType
from apolo_app_types.protocols.common.storage import ApoloFilesPath
from pydantic import ConfigDict, Field


class HuggingFaceModelDetailDynamic(AbstractAppFieldType):
    """Detailed HuggingFace model information."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra=SchemaExtraMetadata(
            title="HuggingFace Model Detail",
            description="Detailed information about a HuggingFace model.",
            meta_type=SchemaMetaType.DYNAMIC,
        ).as_json_schema_extra(),
    )

    id: t.Annotated[
        str,
        Field(
            json_schema_extra=SchemaExtraMetadata(
                title="Repository ID",
                description="The HuggingFace repository identifier.",
            ).as_json_schema_extra()
        ),
    ]

    name: t.Annotated[
        str,
        Field(
            json_schema_extra=SchemaExtraMetadata(
                title="Model Name",
                description="The model name.",
            ).as_json_schema_extra()
        ),
    ]

    visibility: t.Annotated[
        str,
        Field(
            json_schema_extra=SchemaExtraMetadata(
                title="Visibility",
                description="Repository visibility (public or private).",
            ).as_json_schema_extra()
        ),
    ]

    gated: bool = Field(
        default=False,
        json_schema_extra=SchemaExtraMetadata(
            title="Gated",
            description="Whether the model requires access approval.",
        ).as_json_schema_extra(),
    )

    tags: list[str] = Field(
        default_factory=list,
        json_schema_extra=SchemaExtraMetadata(
            title="Tags",
            description="Tags associated with the model.",
        ).as_json_schema_extra(),
    )

    cached: bool = Field(
        default=False,
        json_schema_extra=SchemaExtraMetadata(
            title="Cached",
            description="Whether the model is cached locally.",
        ).as_json_schema_extra(),
    )

    last_modified: (
        t.Annotated[
            str,
            Field(
                json_schema_extra=SchemaExtraMetadata(
                    title="Last Modified",
                    description="Timestamp when the model was last modified.",
                ).as_json_schema_extra()
            ),
        ]
        | None
    ) = None

    files_path: (
        t.Annotated[
            ApoloFilesPath,
            Field(
                json_schema_extra=SchemaExtraMetadata(
                    title="Files Path",
                    description="Path to the cached model files in Apolo storage.",
                ).as_json_schema_extra()
            ),
        ]
        | None
    ) = None


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
