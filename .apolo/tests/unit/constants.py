"""Test constants for HuggingFace Proxy tests."""

from decimal import Decimal

from apolo_sdk import Preset
from neuro_config_client import NvidiaGPUPreset

# Pool names
CPU_POOL = "cpu_pool"
GPU_POOL = "gpu_pool"
DEFAULT_POOL = "default"
DEFAULT_NAMESPACE = "default"
DEFAULT_CLUSTER_NAME = "cluster"
DEFAULT_ORG_NAME = "test-org"
DEFAULT_PROJECT_NAME = "test-project"
APP_SECRETS_NAME = "apps-secrets"
APP_ID = "test-app-instance-id"

# CPU Presets for testing
CPU_PRESETS = {
    "cpu-small": Preset(
        cpu=1.0,
        memory=2e9,  # 2GB
        nvidia_gpu=NvidiaGPUPreset(count=0),
        credits_per_hour=Decimal("1.0"),
        available_resource_pool_names=(CPU_POOL,),
    ),
    "cpu-medium": Preset(
        cpu=2.0,
        memory=4e9,  # 4GB
        nvidia_gpu=NvidiaGPUPreset(count=0),
        credits_per_hour=Decimal("2.0"),
        available_resource_pool_names=(CPU_POOL,),
    ),
    "cpu-large": Preset(
        cpu=4.0,
        memory=8e9,  # 8GB
        nvidia_gpu=NvidiaGPUPreset(count=0),
        credits_per_hour=Decimal("4.0"),
        available_resource_pool_names=(CPU_POOL,),
    ),
    "cpu-tiny": Preset(
        cpu=0.25,
        memory=512e6,  # 512MB - below minimum requirements
        nvidia_gpu=NvidiaGPUPreset(count=0),
        credits_per_hour=Decimal("0.5"),
        available_resource_pool_names=(CPU_POOL,),
    ),
}

# GPU Presets for testing (should be filtered out)
GPU_PRESETS = {
    "gpu-1x-a100": Preset(
        cpu=8.0,
        memory=16e9,
        nvidia_gpu=NvidiaGPUPreset(count=1, memory=80e9),
        credits_per_hour=Decimal("10.0"),
        available_resource_pool_names=(GPU_POOL,),
    ),
}

# Combined test presets
TEST_PRESETS = {
    **CPU_PRESETS,
    **GPU_PRESETS,
}
