"""Input processor for HuggingFace Proxy App."""

import logging
import typing as t

from apolo_app_types.helm.apps.base import BaseChartValueProcessor
from apolo_app_types.helm.apps.common import (
    append_apolo_storage_integration_annotations,
    gen_apolo_storage_integration_labels,
    get_component_values,
)
from apolo_app_types.protocols.common import (
    ApoloFilesMount,
    ApoloMountMode,
    MountPath,
)
from apolo_app_types.protocols.common.secrets_ import serialize_optional_secret
from apolo_app_types.protocols.common.storage import ApoloMountModes

from .types import HfProxyInputs

logger = logging.getLogger(__name__)


class HfProxyChartValueProcessor(BaseChartValueProcessor[HfProxyInputs]):
    """Processes HuggingFace Proxy inputs into Helm chart values."""

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
            has_nvidia_gpu = preset.nvidia_gpu and preset.nvidia_gpu.count > 0
            has_amd_gpu = preset.amd_gpu and preset.amd_gpu.count > 0
            if has_nvidia_gpu or has_amd_gpu:
                logger.debug(f"Skipping preset {preset_name}: has GPU")
                continue

            # Filter 2: Must have capacity
            capacity = jobs_capacity.get(preset_name, 0)
            if capacity <= 0:
                logger.debug(f"Skipping preset {preset_name}: no capacity")
                continue

            # Filter 3: Must meet minimum requirements (0.1 CPU, 0.5Gi RAM)
            cpu = preset.cpu or 0
            memory_bytes = preset.memory or 0
            memory_gb = memory_bytes / 1e9

            if cpu < 0.1:
                logger.debug(f"Skipping preset {preset_name}: insufficient CPU ({cpu} < 0.1)")
                continue

            if memory_gb < 0.5:
                logger.debug(
                    f"Skipping preset {preset_name}: insufficient memory ({memory_gb}Gi < 0.5Gi)"
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
                "Requirements: CPU >= 0.1, Memory >= 0.5Gi, no GPU, capacity > 0"
            )
            raise RuntimeError(msg)

        # Select cheapest preset with most capacity
        selected = min(candidates)
        preset_name = selected[-1]
        logger.info(f"Selected preset: {preset_name}")
        return preset_name

    async def gen_extra_values(
        self,
        input_: HfProxyInputs,
        app_name: str,
        namespace: str,
        app_id: str,
        app_secrets_name: str,
        *_: t.Any,
        **kwargs: t.Any,
    ) -> dict[str, t.Any]:
        """Generate Helm chart values from user inputs."""
        inputs = input_

        # Auto-select the best CPU preset
        preset_name = await self._get_preset()
        preset = self.client.config.presets[preset_name]

        # Get component values (resources, labels, tolerations, affinity)
        component_vals = await get_component_values(preset, preset_name)

        # Storage injection configuration using proper helper functions
        storage_mount = ApoloFilesMount(
            storage_uri=inputs.files_path,
            mount_path=MountPath(path="/root/.cache/huggingface"),
            mode=ApoloMountMode(mode=ApoloMountModes.RW),
        )

        # Pod annotations for storage injection (using helper to get proper format)
        pod_annotations = append_apolo_storage_integration_annotations(
            {}, [storage_mount], self.client
        )

        # Pod labels for storage injection (merge with component labels + org/project)
        pod_labels = {
            **component_vals["labels"],  # Component and preset labels
            **gen_apolo_storage_integration_labels(client=self.client, inject_storage=True),
            "application": "hf-proxy",
        }

        # Environment variables with HF token from app secrets
        env_vars = {
            "HF_TIMEOUT": "30",
            "HF_CACHE_DIR": "/root/.cache/huggingface",
            "PORT": "8080",
            "HF_TOKEN": serialize_optional_secret(inputs.token.token, secret_name=app_secrets_name),
        }

        # Build Helm values
        values: dict[str, t.Any] = {
            # Image configuration
            "image": {
                "repository": "ghcr.io/neuro-inc/apps-huggingface-proxy",
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
            "apolo_app_id": app_id,
        }

        return values
