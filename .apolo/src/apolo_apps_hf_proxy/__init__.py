"""Apolo HuggingFace Proxy App."""

from apolo_apps_hf_proxy.inputs_processor import HfProxyChartValueProcessor
from apolo_apps_hf_proxy.outputs_processor import HfProxyOutputProcessor
from apolo_apps_hf_proxy.types import HfProxyInputs, HfProxyOutputs

__version__ = "0.1.0"

APOLO_APP_TYPE = "hf-proxy"

__all__ = [
    "APOLO_APP_TYPE",
    "HfProxyChartValueProcessor",
    "HfProxyOutputProcessor",
    "HfProxyInputs",
    "HfProxyOutputs",
]
