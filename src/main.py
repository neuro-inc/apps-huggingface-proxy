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
)
from fastapi import Depends, FastAPI

from src.config import Config
from src.dependencies import DepHFService
from src.filters import ModelFilter
from src.logging import setup_logging
from src.models import HFModel, HFModelDetail, ModelListResponse, ModelResponse


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
    filter_params: Annotated[DynamicAppFilterParams, Depends()],
    hf_service: DepHFService,
) -> ModelListResponse:
    """List available models from HuggingFace.

    Supports filtering with syntax: field:operator:value,field2:operator2:value2

    Operators:
        - eq: Exact match (e.g., visibility:eq:public)
        - ne: Not equal (e.g., gated:ne:true)
        - like: Contains substring (e.g., name:like:llama)
        - in: Value in list (e.g., tags:in:text-generation)

    Special filters:
        - cached_only: Return only cached models

    Examples:
        - /outputs?filter=visibility:eq:public
        - /outputs?filter=name:like:llama,gated:eq:false
        - /outputs?filter=cached_only
    """
    try:
        # Parse filter string
        model_filter = ModelFilter(filter_params.filter)

        logger.info(
            "Fetching outputs list",
            extra={"cached_only": model_filter.cached_only, "filter": repr(model_filter)},
        )

        # Extract API-level filters (propagated to HF Hub) and local filters
        api_filters = model_filter.get_api_filters()
        local_conditions = model_filter.get_local_conditions()

        async with hf_service:
            if model_filter.cached_only:
                # Only return cached models without HF Hub API call
                logger.info("Fetching cached models only, skipping HF Hub API")
                hf_response = await hf_service.get_cached_models()
            else:
                # Search HF Hub and local cache in parallel
                # Propagate supported filters to HF API for server-side filtering
                hf_search_task = asyncio.create_task(
                    hf_service.search_models(
                        limit=filter_params.limit or 100,
                        search=api_filters.search,
                        author=api_filters.author,
                        tags=api_filters.tags if api_filters.tags else None,
                    )
                )
                local_search_task = asyncio.create_task(hf_service.search_cache())

                hf_response, cached_models = await asyncio.gather(hf_search_task, local_search_task)

                # Mark which models are cached
                for model in hf_response:
                    if isinstance(model, dict):
                        repo_id = model.get("id", model.get("modelId", ""))
                        model["cached"] = repo_id in cached_models

        # Convert to HFModel objects
        models = []
        for model in hf_response:
            if isinstance(model, dict):
                repo_id = model.get("id", model.get("modelId", ""))
                # Extract model name from repo_id (e.g., "org/model-name" -> "model-name")
                model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
                hf_model = HFModel(
                    id=repo_id,
                    value=HFModelDetail(
                        id=repo_id,
                        name=model_name,
                        visibility="private" if model.get("private") else "public",
                        gated=model.get("gated", False),
                        tags=model.get("tags", []),
                        cached=model.get("cached", False),
                        last_modified=model.get("lastModified"),
                    ),
                )
                models.append(hf_model)

        # Apply local filters (those not supported by HF API)
        models = model_filter.apply_local(models, local_conditions)

        # Apply pagination after filtering
        if filter_params.limit:
            models = models[filter_params.offset : filter_params.offset + filter_params.limit]

        return ModelListResponse(
            status="success",
            data=models,
        )

    except Exception as e:
        logger.error("Failed to fetch outputs", extra={"error": str(e)})
        return ModelListResponse(status="error", data=None)


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

        # Extract model name from repo_id (e.g., "org/model-name" -> "model-name")
        model_name = model_repo_id.split("/")[-1] if "/" in model_repo_id else model_repo_id
        model = HFModel(
            id=model_repo_id,
            value=HFModelDetail(
                id=model_repo_id,
                name=model_name,
                visibility="private" if hf_response.get("private") else "public",
                gated=hf_response.get("gated", False),
                tags=hf_response.get("tags", []),
                cached=hf_response.get("cached", False),
                last_modified=hf_response.get("lastModified"),
            ),
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
