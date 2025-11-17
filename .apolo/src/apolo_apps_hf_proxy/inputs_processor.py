"""Input processor for HuggingFace Proxy App."""

import json
from typing import Any

from .types import HfProxyInputs


class HfProxyChartValueProcessor:
    """Processes HuggingFace Proxy inputs into Helm chart values."""

    def __init__(
        self,
        inputs: HfProxyInputs,
        app_instance_id: str,
        cluster_domain: str,
    ):
        """Initialize the processor."""
        self.inputs = inputs
        self.app_instance_id = app_instance_id
        self.cluster_domain = cluster_domain

    def gen_extra_values(self) -> dict[str, Any]:
        """Generate Helm chart values from user inputs."""
        inputs = self.inputs

        # Get storage URI from cache config
        storage_uri = inputs.cache_config.files_path.path

        # Storage injection configuration
        storage_config = [
            {
                "storage_uri": storage_uri,
                "mount_path": "/root/.cache/huggingface",
                "mount_mode": "rw",  # Read-write for caching
            }
        ]

        # Pod annotations for storage injection
        pod_annotations = {
            "platform.apolo.us/inject-storage": json.dumps(storage_config),
        }

        # Pod labels for storage injection
        pod_labels = {
            "platform.apolo.us/inject-storage": "true",
            "application": "hf-proxy",
        }

        # Environment variables
        env_vars = {
            "HF_TIMEOUT": "30",
            "HF_CACHE_DIR": "/root/.cache/huggingface",
            "PORT": "8080",
        }

        # Build Helm values
        values: dict[str, Any] = {
            # Image configuration
            "image": {
                "repository": "hf-proxy",
                "tag": "latest",
                "pullPolicy": "Always",
            },
            # Resource limits - minimal CPU requirements
            "resources": {
                "limits": {"cpu": "0.5", "memory": "1Gi"},
                "requests": {"cpu": "0.25", "memory": "512Mi"},
            },
            # Pod configuration
            "podAnnotations": pod_annotations,
            "podLabels": pod_labels,
            # Environment variables (will be added to container env)
            "env": env_vars,
            # Service configuration
            "service": {
                "type": "ClusterIP",
                "port": 8080,
            },
            # Remove built-in emptyDir volume since we're using storage injection
            "volumes": [],
            "volumeMounts": [],
            # Apolo app ID for platform integration
            "apolo_app_id": self.app_instance_id,
        }

        # Create secret for HF token using the token key from ApoloSecret
        values["hf_token_secret"] = {
            "name": inputs.token.token_name,
            "key": inputs.token.token.key,
        }

        return values
