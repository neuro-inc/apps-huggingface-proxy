"""Application configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Application configuration."""

    hf_api_base_url: str = "https://huggingface.co/api"
    hf_timeout: int = 30
    hf_token: str | None = None
    hf_cache_dir: str = "/root/.cache/huggingface"
    log_level: str = "INFO"
    log_json: bool = True
    port: int = 8080
    host: str = "0.0.0.0"
