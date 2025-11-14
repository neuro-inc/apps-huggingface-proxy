"""Cache management utilities for HuggingFace models."""

import asyncio

from huggingface_hub import scan_cache_dir


async def is_model_cached(repo_id: str, cache_dir: str) -> bool:
    """Check if a HuggingFace model is cached locally using HF Hub utilities.

    Args:
        repo_id: Repository identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        cache_dir: Path to the HuggingFace cache directory

    Returns:
        True if the model is found in the cache, False otherwise
    """
    if not cache_dir:
        return False

    try:
        # Run cache scan in thread pool (it's a blocking operation)
        cache_info = await asyncio.to_thread(scan_cache_dir, cache_dir)

        # Check if repo_id exists in cached repos
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                return True

        return False
    except Exception:
        # Cache directory doesn't exist or can't be scanned
        return False
