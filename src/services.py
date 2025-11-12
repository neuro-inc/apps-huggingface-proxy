"""Service layer for HuggingFace API interactions."""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class HuggingFaceService:
    """Service for interacting with HuggingFace API."""

    def __init__(
        self,
        token: str | None = None,
        base_url: str = "https://huggingface.co/api",
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.token = token
        self._client: httpx.AsyncClient | None = None

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=headers,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search_models(self, limit: int = 100) -> list[dict[str, Any]]:
        client = await self.get_client()
        params: dict[str, Any] = {"limit": limit}

        try:
            logger.info("Searching HuggingFace models", extra={"limit": limit})
            response = await client.get("/models", params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("HuggingFace API error", extra={"error": str(e)})
            raise

    async def get_repo_details(self, repo_id: str) -> dict[str, Any]:
        client = await self.get_client()

        try:
            logger.info("Fetching repo details", extra={"repo_id": repo_id})
            response = await client.get(f"/models/{repo_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(
                "Failed to fetch repo details",
                extra={"repo_id": repo_id, "error": str(e)},
            )
            raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
