"""Dependency injection for FastAPI endpoints."""

import logging
from typing import Annotated

from fastapi import Depends, Query

from src.services import HuggingFaceService

logger = logging.getLogger(__name__)


async def get_hf_service(
    hf_app_name: Annotated[str, Query()],
) -> HuggingFaceService:
    """Get HuggingFace service with token for the given app name.

    Args:
        hf_app_name: Application name from query params

    Returns:
        HuggingFaceService instance configured with the app's token
    """
    logger.info("Creating HF service", extra={"hf_app_name": hf_app_name})

    # hf_token = lib.get_token(hf_app_name)
    hf_token = None

    service = HuggingFaceService(token=hf_token)
    return service


DepHFService = Annotated[HuggingFaceService, Depends(get_hf_service)]
