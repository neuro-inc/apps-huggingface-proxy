"""Input processor for HuggingFace Proxy App."""

import json
import logging
from typing import Any

import apolo_sdk
from apolo_app_types.helm.apps.common import get_component_values

from .types import HfProxyInputs

logger = logging.getLogger(__name__)


class HfProxyChartValueProcessor:
    """Processes HuggingFace Proxy inputs into Helm chart values."""

    def __init__(
        self,
        inputs: HfProxyInputs,
        app_instance_id: str,
        cluster_domain: str,
        client: apolo_sdk.Client,
    ):
        """Initialize the processor."""
        self.inputs = inputs
        self.app_instance_id = app_instance_id
        self.cluster_domain = cluster_domain
        self.client = client

    async def _get_preset(self) -> str:
        """Auto-select the best CPU preset for hf-proxy deployment.

        Returns:
            str: The name of the selected preset.

        Raises:
            RuntimeError: If no suitable CPU preset is found.
        """
        # Get available presets from cluster
        available_presets = dict(self.client.config.presets)
        jobs_capacity = await self.client.jobs.get_capacity()

        candidates = []

        for preset_name, preset in available_presets.items():
            # Filter 1: Must be CPU-only (no GPU)
            if preset.nvidia_gpu or preset.amd_gpu:
                logger.debug(f"Skipping preset {preset_name}: has GPU")
                continue

            # Filter 2: Must have capacity
            capacity = jobs_capacity.get(preset_name, 0)
            if capacity <= 0:
                logger.debug(f"Skipping preset {preset_name}: no capacity")
                continue

            # Filter 3: Must meet minimum requirements (0.5 CPU, 1Gi RAM)
            cpu = preset.cpu or 0
            memory_bytes = preset.memory or 0
            memory_gb = memory_bytes / 1e9

            if cpu < 0.5:
                logger.debug(f"Skipping preset {preset_name}: insufficient CPU ({cpu} < 0.5)")
                continue

            if memory_gb < 1:
                logger.debug(
                    f"Skipping preset {preset_name}: insufficient memory ({memory_gb}Gi < 1Gi)"
                )
                continue

            # Add to candidates: (cost, -capacity, cpu, preset_name)
            # Sorted to prefer: cheaper, more capacity, less CPU (smallest viable)
            candidates.append(
                (
                    preset.credits_per_hour,
                    -capacity,
                    cpu,
                    preset_name,
                )
            )
            logger.debug(
                f"Preset {preset_name} eligible: "
                f"cpu={cpu}, memory={memory_gb:.1f}Gi, "
                f"cost={preset.credits_per_hour}, capacity={capacity}"
            )

        if not candidates:
            msg = (
                "No suitable CPU preset found for hf-proxy. "
                "Requirements: CPU >= 0.5, Memory >= 1Gi, no GPU, capacity > 0"
            )
            raise RuntimeError(msg)

        # Select cheapest preset with most capacity
        selected = min(candidates)
        preset_name = selected[-1]
        logger.info(f"Selected preset: {preset_name}")
        return preset_name

    async def gen_extra_values(self) -> dict[str, Any]:
        """Generate Helm chart values from user inputs."""
        inputs = self.inputs

        # Auto-select the best CPU preset
        preset_name = await self._get_preset()
        preset = self.client.config.presets[preset_name]

        # Get component values (resources, labels, tolerations, affinity)
        component_vals = await get_component_values(preset, preset_name)

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

        # Pod labels for storage injection (merge with component labels)
        pod_labels = {
            **component_vals["labels"],  # Component and preset labels
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
            # Resource limits from component values (properly formatted)
            "resources": component_vals["resources"],
            # Tolerations from component values
            "tolerations": component_vals["tolerations"],
            # Affinity from component values
            "affinity": component_vals["affinity"],
            # Preset name for reference
            "preset_name": preset_name,
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
