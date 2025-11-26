"""Service layer for HuggingFace API interactions."""

import asyncio
import logging
from functools import partial
from typing import Any

from huggingface_hub import HfApi, scan_cache_dir

logger = logging.getLogger(__name__)


class HuggingFaceService:
    """Service for interacting with HuggingFace API."""

    def __init__(
        self,
        token: str | None = None,
        base_url: str = "https://huggingface.co/api",
        timeout: int = 30,
        cache_dir: str = "/root/.cache/huggingface",
    ) -> None:
        self.token = token
        self.cache_dir = cache_dir
        # Extract endpoint from base_url if provided (remove /api suffix)
        endpoint = base_url.replace("/api", "") if base_url else None
        self.api = HfApi(token=token, endpoint=endpoint)

    async def close(self) -> None:
        """Close is a no-op for HfApi but kept for compatibility."""
        pass

    async def search_models(self, limit: int = 100) -> list[dict[str, Any]]:
        """Search for models on HuggingFace Hub.

        Args:
            limit: Maximum number of models to return

        Returns:
            List of model dictionaries
        """
        try:
            logger.info("Searching HuggingFace models", extra={"limit": limit})

            # Run the blocking API call in a thread pool
            models_iter = await asyncio.to_thread(
                partial(self.api.list_models, limit=limit, full=True)
            )

            # Convert ModelInfo objects to dictionaries
            models = []
            for model in models_iter:
                model_dict = {
                    "id": model.id,
                    "modelId": model.id,
                    "private": getattr(model, "private", False),
                    "gated": getattr(model, "gated", False) in ("auto", "manual"),
                    "tags": model.tags or [],
                    "lastModified": model.lastModified.isoformat() if model.lastModified else None,
                }
                models.append(model_dict)

            return models
        except Exception as e:
            logger.error("HuggingFace API error", extra={"error": str(e)})
            raise

    async def get_repo_details(self, repo_id: str) -> dict[str, Any]:
        """Get details for a specific model repository.

        Checks local cache first. If cached, returns basic info without API call.
        If not cached, fetches full details from HuggingFace Hub.

        Args:
            repo_id: Repository identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")

        Returns:
            Model details dictionary with 'cached' field indicating source
        """
        try:
            logger.info("Fetching repo details", extra={"repo_id": repo_id})

            # Check if model exists in local cache first
            cached = await self.is_model_cached(repo_id)
            if cached:
                logger.info(
                    "Model found in cache, returning without API call",
                    extra={"repo_id": repo_id},
                )
                # Return basic info from cache without making API call
                return {
                    "id": repo_id,
                    "modelId": repo_id,
                    "private": False,  # Default, can't determine from cache alone
                    "gated": False,  # Default, can't determine from cache alone
                    "tags": [],  # Not available in cache
                    "lastModified": None,  # Cache has local mtime, not HF Hub timestamp
                    "cached": True,
                }

            # Model not cached, fetch from HuggingFace Hub
            logger.info(
                "Model not cached, fetching from HuggingFace Hub",
                extra={"repo_id": repo_id},
            )
            model_info = await asyncio.to_thread(self.api.model_info, repo_id)

            # Convert ModelInfo to dictionary
            model_dict = {
                "id": model_info.id,
                "modelId": model_info.id,
                "private": getattr(model_info, "private", False),
                "gated": getattr(model_info, "gated", False),
                "tags": model_info.tags or [],
                "lastModified": (
                    model_info.lastModified.isoformat() if model_info.lastModified else None
                ),
                "cached": False,
            }

            return model_dict
        except Exception as e:
            logger.error(
                "Failed to fetch repo details",
                extra={"repo_id": repo_id, "error": str(e)},
            )
            raise

    async def search_cache(self, model_name_prefix: str | None = None) -> set[str]:
        """Search for cached models in local storage using HF Hub utilities.

        Args:
            model_name_prefix: Optional prefix to filter model names

        Returns:
            Set of cached model IDs (repo_id format)
        """
        if not self.cache_dir:
            logger.debug("Cache directory not configured")
            return set()

        try:
            # Run cache scan in thread pool (it's a blocking operation)
            cache_info = await asyncio.to_thread(scan_cache_dir, self.cache_dir)

            cached_models = set()
            for repo in cache_info.repos:
                repo_id = repo.repo_id
                # Filter by prefix if provided
                if model_name_prefix is None or repo_id.startswith(model_name_prefix):
                    cached_models.add(repo_id)

            logger.info(
                "Found cached models",
                extra={"count": len(cached_models), "prefix": model_name_prefix},
            )
            return cached_models
        except Exception as e:
            logger.debug("Error scanning cache directory", extra={"error": str(e)})
            return set()

    async def get_cached_models(self, model_name_prefix: str | None = None) -> list[dict[str, Any]]:
        """Get list of cached models with basic info without API calls.

        Args:
            model_name_prefix: Optional prefix to filter model names

        Returns:
            List of model dictionaries with basic cached info
        """
        if not self.cache_dir:
            logger.debug("Cache directory not configured")
            return []

        try:
            # Run cache scan in thread pool (it's a blocking operation)
            cache_info = await asyncio.to_thread(scan_cache_dir, self.cache_dir)

            cached_models = []
            for repo in cache_info.repos:
                repo_id = repo.repo_id
                # Filter by prefix if provided
                if model_name_prefix is None or repo_id.startswith(model_name_prefix):
                    cached_models.append(
                        {
                            "id": repo_id,
                            "modelId": repo_id,
                            "private": False,  # Default, can't determine from cache alone
                            "gated": False,  # Default, can't determine from cache alone
                            "tags": [],  # Not available in cache
                            "lastModified": None,  # Cache has local mtime, not HF Hub timestamp
                            "cached": True,
                        }
                    )

            logger.info(
                "Found cached models",
                extra={"count": len(cached_models), "prefix": model_name_prefix},
            )
            return cached_models
        except Exception as e:
            logger.debug("Error scanning cache directory", extra={"error": str(e)})
            return []

    async def is_model_cached(self, repo_id: str) -> bool:
        """Check if a specific model is cached locally using HF Hub utilities.

        Args:
            repo_id: Repository identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")

        Returns:
            True if the model is found in the cache, False otherwise
        """
        if not self.cache_dir:
            return False

        try:
            # Run cache scan in thread pool (it's a blocking operation)
            cache_info = await asyncio.to_thread(scan_cache_dir, self.cache_dir)

            # Check if repo_id exists in cached repos
            for repo in cache_info.repos:
                if repo.repo_id == repo_id:
                    return True

            return False
        except Exception as e:
            logger.debug(
                "Error checking cache for model", extra={"repo_id": repo_id, "error": str(e)}
            )
            return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
