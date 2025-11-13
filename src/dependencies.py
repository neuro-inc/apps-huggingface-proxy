"""Dependency injection for FastAPI endpoints."""

from typing import Annotated

from fastapi import Depends, Request

from src.services import HuggingFaceService

_hf_service: HuggingFaceService | None = None


def get_hf_service(request: Request) -> HuggingFaceService:
    """Get the singleton HuggingFace service instance.

    Args:
        request: FastAPI request object to access app state

    Returns:
        HuggingFaceService instance
    """
    global _hf_service
    if _hf_service is None:
        config = request.app.config
        _hf_service = HuggingFaceService(
            token=config.hf_token,
            base_url=config.hf_api_base_url,
            timeout=config.hf_timeout,
        )
    return _hf_service


DepHFService = Annotated[HuggingFaceService, Depends(get_hf_service)]
