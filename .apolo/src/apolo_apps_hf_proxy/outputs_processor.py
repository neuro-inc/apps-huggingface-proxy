"""Output processor for HuggingFace Proxy App."""

from typing import Any

from .types import HfProxyInputs, HfProxyOutputs


class HfProxyOutputProcessor:
    """Processes Helm deployment outputs into app outputs."""

    def __init__(
        self,
        k8s_client: Any,
        app_instance_id: str,
        app_namespace: str,
        inputs: HfProxyInputs,
    ):
        """Initialize the processor."""
        self.k8s_client = k8s_client
        self.app_instance_id = app_instance_id
        self.app_namespace = app_namespace
        self.inputs = inputs

    async def gen_outputs(self) -> HfProxyOutputs:
        """Extract outputs from Kubernetes deployment."""
        # Get service information
        service_name = await self._get_service_name()
        namespace = self.app_namespace

        # Construct internal URL
        # Format: http://{service-name}.{namespace}.svc.cluster.local:{port}
        internal_url = f"http://{service_name}.{namespace}.svc.cluster.local:8080"

        return HfProxyOutputs(
            cache_config=self.inputs.cache_config,
            token=self.inputs.token,
            internal_url=internal_url,
        )

    async def _get_service_name(self) -> str:
        """Get the service name from Kubernetes."""
        # Query Kubernetes for the service with label output-server=true
        services = await self.k8s_client.list_namespaced_service(
            namespace=self.app_namespace,
            label_selector=f"app.kubernetes.io/instance={self.app_instance_id},output-server=true",
        )

        if not services.items:
            # Fallback to default naming
            return f"hf-proxy-{self.app_instance_id}"

        return services.items[0].metadata.name
