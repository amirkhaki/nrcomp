"""Metric registry defining directionality, transforms, and metadata.

Ensures that all quality metrics are standardized so that the optimization and
prediction target is always monotonically 'higher = better (less degraded)'.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


@dataclass(frozen=True)
class MetricSpec:
    name: str
    higher_better: bool
    transform: str = "identity"  # "identity" or "one_minus"
    description: str = ""
    range_min: float = 0.0
    range_max: float = 1.0


# Supported full-reference and reference proxy metrics
METRIC_REGISTRY: Dict[str, MetricSpec] = {
    "haarpsi": MetricSpec(
        name="haarpsi",
        higher_better=True,
        transform="identity",
        description="Haar wavelet-based perceptual similarity index (higher = better)",
        range_min=0.0,
        range_max=1.0,
    ),
    "dists": MetricSpec(
        name="dists",
        higher_better=False,
        transform="one_minus",
        description="Deep Image Structure and Texture Similarity (raw lower = better; mapped to 1 - DISTS for training)",
        range_min=0.0,
        range_max=1.0,
    ),
    "psnr": MetricSpec(
        name="psnr",
        higher_better=True,
        transform="identity",
        description="Peak signal-to-noise ratio in dB (higher = better)",
        range_min=0.0,
        range_max=100.0,
    ),
    "ms_ssim": MetricSpec(
        name="ms_ssim",
        higher_better=True,
        transform="identity",
        description="Multi-Scale Structural Similarity Index",
        range_min=0.0,
        range_max=1.0,
    ),
    "iw_ssim": MetricSpec(
        name="iw_ssim",
        higher_better=True,
        transform="identity",
        description="Information Content Weighted SSIM",
        range_min=0.0,
        range_max=1.0,
    ),
    "lpips": MetricSpec(
        name="lpips",
        higher_better=False,
        transform="one_minus",
        description="Learned Perceptual Image Patch Similarity (mapped to 1 - LPIPS)",
        range_min=0.0,
        range_max=1.0,
    ),
    "cvvdp": MetricSpec(
        name="cvvdp",
        higher_better=False,
        transform="one_minus",
        description="Color Visual Voice/Difference Predictor (mapped to higher = better)",
        range_min=0.0,
        range_max=10.0,
    ),
}


def get_metric_spec(metric_name: str) -> MetricSpec:
    """Retrieve MetricSpec for a given metric name (case-insensitive)."""
    key = str(metric_name).lower().strip()
    if key not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric '{metric_name}'. Registered metrics: {list(METRIC_REGISTRY.keys())}"
        )
    return METRIC_REGISTRY[key]


def prepare_target_value(raw_value: Optional[float], metric_name: str, clip: bool = True) -> Optional[float]:
    """Transform raw metric value to standardized higher-better scalar target."""
    if raw_value is None:
        return None
    val = float(raw_value)
    spec = get_metric_spec(metric_name)
    if spec.transform == "one_minus":
        val = 1.0 - val
    if clip and spec.range_min is not None and spec.range_max is not None:
        val = max(spec.range_min, min(spec.range_max, val))
    return val
