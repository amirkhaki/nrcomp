"""Quality metric computation for proxy label generation.

Implements HaarPSI, DISTS, and PSNR over full-resolution RGB images in [0, 1] (NCHW float32).
Uses PyIQA library on the execution server, with graceful mock/fallback for offline unit testing.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union


def compute_psnr(orig: Any, recon: Any, max_val: Optional[float] = None) -> float:
    """Compute PSNR between two images in [0, 255] (uint8/float) or [0, 1] (float)."""
    try:
        import numpy as np
        if isinstance(orig, np.ndarray) and isinstance(recon, np.ndarray):
            orig_f = orig.astype(np.float64)
            recon_f = recon.astype(np.float64)
            if max_val is None:
                peak = 255.0 if (orig_f.max() > 1.0 or recon_f.max() > 1.0 or orig.dtype == np.uint8) else 1.0
            else:
                peak = float(max_val)
            mse = np.mean((orig_f - recon_f) ** 2)
            if mse <= 1e-15:
                return float("inf")
            return float(10.0 * math.log10(peak ** 2 / mse))
    except ImportError:
        pass

    # Pure Python fallback for nested lists / iterables
    def flatten(x):
        if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
            for item in x:
                yield from flatten(item)
        else:
            yield x

    f1 = list(flatten(orig))
    f2 = list(flatten(recon))
    if len(f1) == 0 or len(f1) != len(f2):
        return 0.0

    if max_val is None:
        peak = 255.0 if (max(f1) > 1.0 or max(f2) > 1.0 or any(isinstance(x, int) for x in f1)) else 1.0
    else:
        peak = float(max_val)

    mse = sum((float(a) - float(b)) ** 2 for a, b in zip(f1, f2)) / len(f1)
    if mse <= 1e-15:
        return float("inf")
    return float(10.0 * math.log10(peak ** 2 / mse))


class QualityMetrics:
    """Manages PyIQA metric models and calculates HaarPSI, DISTS, and PSNR."""

    def __init__(self, device: str = "cuda", enable_metrics: Optional[list] = None):
        self.device = device
        self.enable_metrics = enable_metrics or ["haarpsi", "dists", "psnr"]
        self.models: Dict[str, Any] = {}
        self._init_models()

    def _init_models(self):
        """Lazy or eager initialization of PyIQA metric models."""
        try:
            import pyiqa
            import torch

            actual_device = self.device if (torch.cuda.is_available() and self.device == "cuda") else "cpu"

            if "haarpsi" in self.enable_metrics:
                self.models["haarpsi"] = pyiqa.create_metric("haarpsi", device=actual_device)
            if "dists" in self.enable_metrics:
                self.models["dists"] = pyiqa.create_metric("dists", device=actual_device)
        except (ImportError, Exception):
            # Graceful fallback for offline environment without pyiqa/torch
            pass

    def _to_tensor(self, img: Any) -> Any:
        """Convert input uint8 or float image to (1, 3, H, W) float32 tensor in [0, 1]."""
        import torch

        if isinstance(img, torch.Tensor):
            t = img.float()
            if t.dim() == 3:
                t = t.unsqueeze(0)
            if t.shape[1] != 3 and t.shape[-1] == 3:
                t = t.permute(0, 3, 1, 2)
            if t.max() > 1.0:
                t = t / 255.0
            return t.to(self.device)

        # Convert from numpy / PIL / list
        try:
            import numpy as np
            np_arr = np.asarray(img)
            if np_arr.dtype == np.uint8:
                np_arr = np_arr.astype(np.float32) / 255.0
            else:
                np_arr = np_arr.astype(np.float32)
                if np_arr.max() > 1.0:
                    np_arr = np_arr / 255.0

            if np_arr.ndim == 3:
                # HWC -> CHW -> NCHW
                np_arr = np.transpose(np_arr, (2, 0, 1))[np.newaxis, ...]
            elif np_arr.ndim == 4 and np_arr.shape[-1] == 3:
                np_arr = np.transpose(np_arr, (0, 3, 1, 2))

            return torch.from_numpy(np_arr).float().to(self.device)
        except ImportError:
            return torch.tensor(img, dtype=torch.float32).to(self.device)

    def compute_all(self, orig: Any, recon: Any) -> Dict[str, float]:
        """Compute all configured metrics for orig and recon."""
        results: Dict[str, float] = {}

        # 1. PSNR
        if "psnr" in self.enable_metrics:
            try:
                results["psnr"] = compute_psnr(orig, recon)
            except Exception:
                results["psnr"] = 0.0

        # PyIQA metrics (HaarPSI, DISTS)
        try:
            import torch
            with torch.no_grad():
                t_orig = self._to_tensor(orig)
                t_recon = self._to_tensor(recon)

                if "haarpsi" in self.enable_metrics and "haarpsi" in self.models:
                    val = self.models["haarpsi"](t_recon, t_orig)
                    results["haarpsi"] = float(val.item() if hasattr(val, "item") else val)

                if "dists" in self.enable_metrics and "dists" in self.models:
                    val = self.models["dists"](t_recon, t_orig)
                    results["dists"] = float(val.item() if hasattr(val, "item") else val)
        except (ImportError, Exception):
            pass

        return results
