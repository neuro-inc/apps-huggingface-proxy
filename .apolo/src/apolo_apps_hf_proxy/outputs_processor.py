"""Output processor for HuggingFace Proxy App."""

import typing as t

from apolo_app_types.outputs.base import BaseAppOutputsProcessor
from apolo_app_types.protocols.common.hugging_face import HuggingFaceToken
from apolo_app_types.protocols.common.secrets_ import ApoloSecret
from apolo_app_types.protocols.common.storage import ApoloFilesPath

from .types import HfProxyOutputs


class HfProxyOutputProcessor(BaseAppOutputsProcessor[HfProxyOutputs]):
    """Processes Helm deployment outputs into app outputs."""

    async def _generate_outputs(
        self,
        helm_values: dict[str, t.Any],
        app_instance_id: str,
    ) -> HfProxyOutputs:
        """Generate outputs from Helm deployment.

        Args:
            helm_values: The Helm chart values used for deployment
            app_instance_id: The app instance identifier

        Returns:
            HfProxyOutputs with configuration
        """
        # Reconstruct files_path from helm_values
        # The storage URI is in the pod annotations as JSON
        storage_uri = "storage:.apps/hugging-face-cache"  # Default
        if "podAnnotations" in helm_values:
            import json

            storage_annotation = helm_values["podAnnotations"].get(
                "platform.apolo.us/inject-storage"
            )
            if storage_annotation:
                storage_config = json.loads(storage_annotation)
                if storage_config and len(storage_config) > 0:
                    storage_uri = storage_config[0].get("storage_uri", storage_uri)

        files_path = ApoloFilesPath(path=storage_uri)

        # Reconstruct token from helm_values
        token_secret = helm_values.get("hf_token_secret", {})
        token_name = token_secret.get("name", "hf-token")
        token_key = token_secret.get("key", "HF_TOKEN")

        token = HuggingFaceToken(
            token_name=token_name,
            token=ApoloSecret(key=token_key),
        )

        return HfProxyOutputs(
            files_path=files_path,
            token=token,
        )
