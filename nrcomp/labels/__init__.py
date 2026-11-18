"""Quality metric labelling package."""

from nrcomp.labels.metrics import QualityMetrics, compute_psnr
from nrcomp.labels.registry import (
    METRIC_REGISTRY,
    MetricSpec,
    get_metric_spec,
    prepare_target_value,
)

__all__ = [
    "QualityMetrics",
    "compute_psnr",
    "METRIC_REGISTRY",
    "MetricSpec",
    "get_metric_spec",
    "prepare_target_value",
]
