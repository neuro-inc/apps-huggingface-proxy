"""Cache management utilities for HuggingFace models."""

import os
from pathlib import Path


def is_model_cached(repo_id: str, cache_dir: str) -> bool:
    """Check if a HuggingFace model is cached locally.

    Args:
        repo_id: Repository identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        cache_dir: Path to the HuggingFace cache directory

    Returns:
        True if the model is found in the cache, False otherwise
    """
    if not cache_dir or not os.path.exists(cache_dir):
        return False

    cache_path = Path(cache_dir)
    models_dir = cache_path / "hub"

    if not models_dir.exists():
        return False

    # HuggingFace stores models with "models--" prefix and "--" instead of "/"
    normalized_repo_id = repo_id.replace("/", "--")
    model_cache_name = f"models--{normalized_repo_id}"

    # Check if the model directory exists in the cache
    model_path = models_dir / model_cache_name
    return model_path.exists() and model_path.is_dir()
