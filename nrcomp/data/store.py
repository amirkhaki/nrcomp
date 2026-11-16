"""Latent storage I/O and pre-flight disk capacity checks.

Handles saving/loading quantized latents (y_hat, optionally z_hat) in fp16
and ensures sufficient free disk space is available before starting generation.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


def check_disk_space(
    target_dir: Union[str, Path],
    num_samples: int,
    bytes_per_sample: int = 2 * 1024 * 1024,  # ~2 MB per fp16 latent
    safety_margin: float = 1.1,
) -> Tuple[bool, int, int]:
    """Check if target filesystem has >= safety_margin * estimated required storage.

    Returns:
        (has_enough_space, free_bytes, required_bytes)
    """
    path = Path(target_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)

    stat = shutil.disk_usage(path)
    free_bytes = stat.free
    estimated_required = int(num_samples * bytes_per_sample * safety_margin)

    return (free_bytes >= estimated_required), free_bytes, estimated_required


class LatentStore:
    """I/O interface for saving and loading codec latents in fp16."""

    def __init__(self, root_dir: Union[str, Path], store_format: str = "npz"):
        self.root_dir = Path(root_dir)
        self.store_format = store_format

    def get_path(
        self, dataset: str, image_id: str, lambda_idx: int
    ) -> Path:
        """Construct deterministic file path for latent tensor."""
        clean_id = Path(image_id).stem
        ext = ".pt" if self.store_format == "pt" else ".npz"
        return self.root_dir / dataset / f"lambda_{lambda_idx}" / f"{clean_id}{ext}"

    def save_latents(
        self,
        dest_path: Path,
        y_hat: Any,
        z_hat: Optional[Any] = None,
        valid_mask: Optional[Any] = None,
    ):
        """Save latent arrays/tensors to disk."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if self.store_format == "pt":
            import torch
            data_dict = {"y_hat": y_hat.half() if hasattr(y_hat, "half") else y_hat}
            if z_hat is not None:
                data_dict["z_hat"] = z_hat.half() if hasattr(z_hat, "half") else z_hat
            if valid_mask is not None:
                data_dict["valid_mask"] = valid_mask.bool() if hasattr(valid_mask, "bool") else valid_mask
            tmp_path = dest_path.with_name(f"{dest_path.stem}.tmp.pt")
            torch.save(data_dict, tmp_path)
            os.replace(tmp_path, dest_path)
            return

        # Handle torch tensors if available
        try:
            import torch
            if isinstance(y_hat, torch.Tensor):
                y_hat = y_hat.detach().cpu().half().numpy()
            if z_hat is not None and isinstance(z_hat, torch.Tensor):
                z_hat = z_hat.detach().cpu().half().numpy()
            if valid_mask is not None and isinstance(valid_mask, torch.Tensor):
                valid_mask = valid_mask.detach().cpu().bool().numpy()
        except ImportError:
            pass

        import numpy as np

        data_dict = {"y_hat": np.asarray(y_hat, dtype=np.float16)}
        if z_hat is not None:
            data_dict["z_hat"] = np.asarray(z_hat, dtype=np.float16)
        if valid_mask is not None:
            data_dict["valid_mask"] = np.asarray(valid_mask, dtype=bool)

        tmp_path = dest_path.with_name(f"{dest_path.stem}.tmp.npz")
        with open(tmp_path, "wb") as f:
            np.savez_compressed(f, **data_dict)
        os.replace(tmp_path, dest_path)

    def load_latents(self, latent_path: Path) -> Dict[str, Any]:
        """Load latent arrays from disk."""
        if latent_path.suffix == ".pt":
            import torch
            return torch.load(latent_path, map_location="cpu")

        import numpy as np

        with np.load(latent_path) as data:
            result = {"y_hat": data["y_hat"]}
            if "z_hat" in data:
                result["z_hat"] = data["z_hat"]
            if "valid_mask" in data:
                result["valid_mask"] = data["valid_mask"]
        return result
