"""HuggingFace Proxy Service - Main application."""

import asyncio
import logging
import os
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from apolo_app_types import (
    DynamicAppBasicResponse,
    DynamicAppFilterParams,
    DynamicAppIdResponse,
    DynamicAppListResponse,
)
from fastapi import FastAPI, Query

from src.config import Config
from src.dependencies import DepHFService
from src.logging import setup_logging
from src.models import HFModel, ModelResponse


class App(FastAPI):
    config: Config


shutdown_event = asyncio.Event()


def handle_shutdown_signal(signum: int, frame: Any) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Received shutdown signal", extra={"signal": signum})
    shutdown_event.set()


@asynccontextmanager
async def lifespan(_: App) -> AsyncIterator[None]:
    logger = logging.getLogger(__name__)
    logger.info("Starting HuggingFace Proxy Service", extra={"version": "0.1.0"})

    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    signal.signal(signal.SIGINT, handle_shutdown_signal)

    yield

    logger.info("Shutting down gracefully...")
    logger.info("Shutdown complete")


app = App(
    title="HuggingFace Proxy Service",
    version="0.1.0",
    description="Production-ready proxy for HuggingFace API",
    lifespan=lifespan,
)

app.config = Config(
    hf_api_base_url=os.getenv("HF_API_BASE_URL", "https://huggingface.co/api"),
    hf_timeout=int(os.getenv("HF_TIMEOUT", "30")),
    hf_token=os.getenv("HF_TOKEN"),
    hf_cache_dir=os.getenv("HF_CACHE_DIR", "/root/.cache/huggingface"),
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_json=os.getenv("LOG_JSON", "true").lower() == "true",
    port=int(os.getenv("PORT", "8080")),
    host=os.getenv("HOST", "0.0.0.0"),
)

setup_logging(app.config)
logger = logging.getLogger(__name__)


@app.get("/")
@app.get("/health")
@app.get("/healthz")
async def root() -> DynamicAppBasicResponse:
    return DynamicAppBasicResponse(status="healthy")


@app.get("/outputs")
async def list_outputs(
    filter_params: Annotated[DynamicAppFilterParams, Query()],
    hf_service: DepHFService,
) -> DynamicAppListResponse:
    """List available models from HuggingFace."""
    try:
        logger.info("Fetching outputs list")

        async with hf_service:
            # Search HF Hub and local cache in parallel
            hf_search_task = asyncio.create_task(
                hf_service.search_models(limit=filter_params.limit)
            )
            local_search_task = asyncio.create_task(
                hf_service.search_cache(model_name_prefix=filter_params.filter)
            )

            hf_response, cached_models = await asyncio.gather(hf_search_task, local_search_task)

        # Only include models that exist in HF Hub (not just local finetuned models)
        models = []
        for model in hf_response:
            if isinstance(model, dict):
                repo_id = model.get("id", model.get("modelId", ""))
                hf_model = HFModel(
                    repo_id=repo_id,
                    visibility="private" if model.get("private") else "public",
                    gated=model.get("gated", False),
                    tags=model.get("tags", []),
                    cached=repo_id in cached_models,
                    last_modified=model.get("lastModified"),
                )
                models.append(hf_model)

        if filter_params.filter:
            models = [
                m
                for m in models
                if filter_params.filter.lower() in m.repo_id.lower()
                or any(filter_params.filter.lower() in tag.lower() for tag in m.tags)
            ]

        if filter_params.limit:
            models = models[filter_params.offset : filter_params.offset + filter_params.limit]

        return DynamicAppListResponse(
            status="success",
            data=[DynamicAppIdResponse(id=model.repo_id, value=model) for model in models],
        )

    except Exception as e:
        logger.error("Failed to fetch outputs", extra={"error": str(e)})
        return DynamicAppListResponse(status="error", data=None)


@app.get("/outputs/{repo_id:path}")
async def get_output_detail(
    repo_id: str,
    hf_service: DepHFService,
) -> ModelResponse:
    """Get details for a specific repository."""
    try:
        logger.info("Fetching output details", extra={"repo_id": repo_id})

        async with hf_service:
            hf_response = await hf_service.get_repo_details(repo_id)
            model_repo_id = hf_response.get("id", hf_response.get("modelId", repo_id))
            cached = await hf_service.is_model_cached(model_repo_id)

        model = HFModel(
            repo_id=model_repo_id,
            visibility="private" if hf_response.get("private") else "public",
            gated=hf_response.get("gated", False),
            tags=hf_response.get("tags", []),
            cached=cached,
            last_modified=hf_response.get("lastModified"),
        )

        return ModelResponse(status="success", data=model)

    except Exception as e:
        logger.error("Failed to fetch output details", extra={"repo_id": repo_id, "error": str(e)})
        return ModelResponse(status="error", data=None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=app.config.host,
        port=app.config.port,
        log_config=None,
    )
