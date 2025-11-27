"""Type definitions for HuggingFace Proxy App."""

import typing as t

from apolo_app_types.protocols.common import AbstractAppFieldType, AppInputs, AppOutputs
from apolo_app_types.protocols.common.hugging_face import HuggingFaceCache, HuggingFaceToken
from apolo_app_types.protocols.common.schema_extra import SchemaExtraMetadata, SchemaMetaType
from pydantic import ConfigDict, Field


class HuggingFaceModelDynamic(AbstractAppFieldType):
    """HuggingFace model/repository information."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra=SchemaExtraMetadata(
            title="HuggingFace Model",
            description="A HuggingFace model or repository.",
            meta_type=SchemaMetaType.DYNAMIC,
        ).as_json_schema_extra(),
    )

    id: t.Annotated[
        str,
        Field(
            json_schema_extra=SchemaExtraMetadata(
                title="Repository ID",
                description="The HuggingFace repository "
                "identifier (e.g., 'meta-llama/Llama-2-7b').",
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


class HuggingFaceModelsDynamic(AbstractAppFieldType):
    """Mapping of HuggingFace model IDs to their information."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra=SchemaExtraMetadata(
            title="HuggingFace Models",
            description="Mapping of HuggingFace model IDs to their information.",
            meta_type=SchemaMetaType.DYNAMIC,
        ).as_json_schema_extra(),
    )
    hf_models: list[HuggingFaceModelDynamic] = Field(
        default_factory=list,
        json_schema_extra=SchemaExtraMetadata(
            title="HuggingFace Models List",
            description="List of HuggingFace models.",
        ).as_json_schema_extra(),
    )


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

    huggingface_models: HuggingFaceModelsDynamic = Field(
        default_factory=HuggingFaceModelsDynamic,
        json_schema_extra=SchemaExtraMetadata(
            title="HuggingFace Models",
            description="List of available HuggingFace models.",
            meta_type=SchemaMetaType.DYNAMIC,
        ).as_json_schema_extra(),
    )
