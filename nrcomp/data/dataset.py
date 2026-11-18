"""Dataset and batching logic for latent training.

Implements LatentDataset, SourceImageBatchSampler (B source images, k lambdas per image),
collate function that handles zero-padding and valid_mask aggregation,
and training normalization statistics computation over valid cells.
Strictly no spatial augmentation on latents.
"""

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

    class TorchDataset:  # type: ignore
        pass


def seed_worker(worker_id: int):
    """Seed dataloader worker deterministically."""
    worker_seed = (torch.initial_seed() % 2**32) if HAS_TORCH else 42
    random.seed(worker_seed)
    try:
        import numpy as np
        np.random.seed(worker_seed)
    except ImportError:
        pass


class LatentDataset(TorchDataset):
    """Dataset loading codec latents and targets from manifest records."""

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        latent_store: Any,
        split: Optional[str] = None,
        config: str = "A",
        target_metric: str = "haarpsi",
        dataset_name: Optional[str] = None,
    ):
        self.latent_store = latent_store
        self.split = split
        self.dataset_name = dataset_name
        self.config = config.upper()
        self.target_metric = target_metric

        # Filter rows by dataset, split, and target validity
        self.samples = []
        for r in rows:
            if dataset_name is not None and r.get("dataset") != dataset_name:
                continue
            if split is not None and r.get("split") != split:
                continue

            target_val = r.get(target_metric)
            if target_val is None:
                continue
            try:
                target_f = float(target_val)
                if math.isnan(target_f) or math.isinf(target_f):
                    continue
            except (ValueError, TypeError):
                continue

            self.samples.append(r)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.samples[idx]
        latent_path = Path(row["latent_path"])
        
        # Load from latent store
        latents = self.latent_store.load_latents(latent_path)
        y_hat = latents["y_hat"]
        valid_mask = latents.get("valid_mask")

        # Load z_hat only if Config B to save bandwidth and memory
        z_hat = None
        if self.config == "B":
            z_hat = latents.get("z_hat")

        item = {
            "y_hat": y_hat,
            "valid_mask": valid_mask,
            "image_id": str(row["image_id"]),
            "lambda_idx": int(row["lambda_idx"]),
            "target": float(row[self.target_metric]),
            "bpp_actual": float(row["bpp_actual"]) if row.get("bpp_actual") is not None else 0.0,
            "orig_mos": float(row["orig_mos"]) if row.get("orig_mos") is not None else None,
            "dists": float(row["dists"]) if row.get("dists") is not None else None,
            "psnr": float(row["psnr"]) if row.get("psnr") is not None else None,
            "dataset": str(row.get("dataset", "")),
        }

        if z_hat is not None:
            item["z_hat"] = z_hat

        return item

    def get_image_to_indices(self) -> Dict[str, List[int]]:
        """Map each source image_id to its indices in this dataset."""
        mapping: Dict[str, List[int]] = defaultdict(list)
        for idx, row in enumerate(self.samples):
            mapping[str(row["image_id"])].append(idx)
        return mapping


class SourceImageBatchSampler:
    """Samples batches of size (B * k) composed of B source images and k lambdas each."""

    def __init__(
        self,
        image_to_indices: Dict[str, List[int]],
        batch_images: int = 8,
        lambdas_per_image: int = 3,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.image_to_indices = image_to_indices
        self.batch_images = batch_images
        self.k = lambdas_per_image
        self.shuffle = shuffle
        self.seed = seed
        self.rng = random.Random(seed)

        # Filter out images with fewer than k available samples
        self.valid_images = [
            img_id
            for img_id, indices in self.image_to_indices.items()
            if len(indices) >= self.k
        ]

    def __iter__(self) -> Iterator[List[int]]:
        images = list(self.valid_images)
        if self.shuffle:
            self.rng.shuffle(images)

        batch = []
        for img_id in images:
            indices = self.image_to_indices[img_id]
            # Sample k lambdas for this image
            sampled = self.rng.sample(indices, self.k) if self.shuffle else indices[: self.k]
            batch.extend(sampled)

            if len(batch) == self.batch_images * self.k:
                yield batch
                batch = []

    def __len__(self) -> int:
        return len(self.valid_images) // self.batch_images


class MockArray:
    """Pure-python fallback multidimensional array for dev environments without numpy."""

    def __init__(self, shape: Tuple[int, ...], fill: Any = 0.0):
        self.shape = tuple(shape)
        self._strides = []
        acc = 1
        for dim in reversed(self.shape):
            self._strides.insert(0, acc)
            acc *= dim
        self._data = [fill] * acc

    def _flat(self, idx: Tuple[int, ...]) -> int:
        return sum(i * s for i, s in zip(idx, self._strides))

    def __getitem__(self, idx: Tuple[int, ...]) -> Any:
        return self._data[self._flat(idx)]

    def __setitem__(self, idx: Tuple[int, ...], val: Any):
        self._data[self._flat(idx)] = val

    def __array__(self, dtype: Any = None) -> Any:
        try:
            import numpy as np
            return np.array(self._data, dtype=dtype).reshape(self.shape)
        except ImportError:
            return self


def collate_latent_samples(batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate batch items into zero-padded batch tensors.

    Zero-pads spatial dimensions to max_h, max_w across the batch and
    extends valid_mask with False.
    Works with numpy arrays, torch tensors, or pure Python MockArrays.
    """
    if not batch_items:
        return {}

    has_torch = False
    try:
        import torch
        has_torch = True
    except ImportError:
        pass

    has_np = False
    try:
        import numpy as np
        has_np = True
    except ImportError:
        pass

    # Find maximum spatial dimensions
    max_h = max(item["y_hat"].shape[-2] for item in batch_items)
    max_w = max(item["y_hat"].shape[-1] for item in batch_items)
    num_channels = batch_items[0]["y_hat"].shape[-3]
    batch_size = len(batch_items)

    # Prepare containers
    if has_np:
        padded_y = np.zeros((batch_size, num_channels, max_h, max_w), dtype=np.float32)
        padded_mask = np.zeros((batch_size, max_h, max_w), dtype=bool)
    else:
        padded_y = MockArray((batch_size, num_channels, max_h, max_w), fill=0.0)
        padded_mask = MockArray((batch_size, max_h, max_w), fill=False)

    has_z = "z_hat" in batch_items[0] and batch_items[0]["z_hat"] is not None
    padded_z = None
    if has_z:
        z_channels = batch_items[0]["z_hat"].shape[-3]
        max_zh = max(item["z_hat"].shape[-2] for item in batch_items)
        max_zw = max(item["z_hat"].shape[-1] for item in batch_items)
        if has_np:
            padded_z = np.zeros((batch_size, z_channels, max_zh, max_zw), dtype=np.float32)
        else:
            padded_z = MockArray((batch_size, z_channels, max_zh, max_zw), fill=0.0)

    image_ids = []
    lambda_indices = []
    targets = []

    for i, item in enumerate(batch_items):
        y = item["y_hat"]
        if has_torch and isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        h, w = y.shape[-2], y.shape[-1]

        if has_np and not isinstance(padded_y, MockArray) and not isinstance(y, MockArray):
            padded_y[i, :, :h, :w] = y
        else:
            for c in range(num_channels):
                for r in range(h):
                    for col in range(w):
                        padded_y[i, c, r, col] = y[c, r, col]

        mask = item.get("valid_mask")
        if mask is not None:
            if has_torch and isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            if has_np and not isinstance(padded_mask, MockArray) and not isinstance(mask, MockArray):
                padded_mask[i, :h, :w] = mask
            else:
                for r in range(h):
                    for col in range(w):
                        padded_mask[i, r, col] = mask[r, col]
        else:
            if has_np and not isinstance(padded_mask, MockArray):
                padded_mask[i, :h, :w] = True
            else:
                for r in range(h):
                    for col in range(w):
                        padded_mask[i, r, col] = True

        if has_z:
            z = item["z_hat"]
            if has_torch and isinstance(z, torch.Tensor):
                z = z.detach().cpu().numpy()
            zh, zw = z.shape[-2], z.shape[-1]
            if has_np and not isinstance(padded_z, MockArray) and not isinstance(z, MockArray):
                padded_z[i, :, :zh, :zw] = z
            else:
                for c in range(z_channels):
                    for r in range(zh):
                        for col in range(zw):
                            padded_z[i, c, r, col] = z[c, r, col]

        image_ids.append(item.get("image_id", ""))
        lambda_indices.append(item.get("lambda_idx", 0))
        if "target" in item:
            targets.append(item["target"])

    result = {
        "y_hat": torch.from_numpy(padded_y) if (has_torch and has_np and not isinstance(padded_y, MockArray)) else padded_y,
        "valid_mask": torch.from_numpy(padded_mask) if (has_torch and has_np and not isinstance(padded_mask, MockArray)) else padded_mask,
        "image_ids": image_ids,
        "lambda_indices": torch.tensor(lambda_indices, dtype=torch.long)
        if has_torch
        else lambda_indices,
        "bpp_actual": [item.get("bpp_actual", 0.0) for item in batch_items],
        "orig_mos": [item.get("orig_mos") for item in batch_items],
        "dists": [item.get("dists") for item in batch_items],
        "psnr": [item.get("psnr") for item in batch_items],
        "dataset": [item.get("dataset", "") for item in batch_items],
    }

    if has_z:
        result["z_hat"] = torch.from_numpy(padded_z) if (has_torch and has_np and not isinstance(padded_z, MockArray)) else padded_z

    if targets:
        result["targets"] = torch.tensor(targets, dtype=torch.float32) if has_torch else targets

    return result


def compute_train_norm_stats(
    dataset: LatentDataset,
    max_samples: int = 500,
    seed: int = 42,
    epsilon: float = 1e-5,
) -> Dict[str, Tuple[List[float], List[float]]]:
    """Compute per-channel mean and std over training split latents on valid cells only.

    Args:
        dataset: LatentDataset filtered to training split.
        max_samples: Number of stratified samples across lambdas.
        seed: Random seed for sampling.
        epsilon: Minimum clamp on standard deviation for dead/dormant channels.

    Returns:
        Dict with keys 'y' (and 'z' if Config B), mapping to (mean_list, std_list).
    """
    if len(dataset) == 0:
        return {"y": ([0.0] * 320, [1.0] * 320)}

    rng = random.Random(seed)
    # Stratified sampling across lambdas
    by_lambda = defaultdict(list)
    for idx in range(len(dataset)):
        item_row = dataset.samples[idx]
        by_lambda[int(item_row["lambda_idx"])].append(idx)

    per_lambda_k = max(1, max_samples // max(1, len(by_lambda)))
    sample_indices = []
    for l_idx, indices in by_lambda.items():
        sampled = rng.sample(indices, min(per_lambda_k, len(indices)))
        sample_indices.extend(sampled)

    sample_indices = sample_indices[:max_samples]

    try:
        import numpy as np
        has_np = True
    except ImportError:
        has_np = False

    if has_np:
        y_sum = None
        y_sq_sum = None
        y_count = 0

        z_sum = None
        z_sq_sum = None
        z_count = 0

        for idx in sample_indices:
            item = dataset[idx]
            y = np.asarray(item["y_hat"], dtype=np.float32)
            mask = np.asarray(item["valid_mask"], dtype=bool) if item.get("valid_mask") is not None else None

            if y_sum is None:
                c_y = y.shape[0]
                y_sum = np.zeros(c_y, dtype=np.float64)
                y_sq_sum = np.zeros(c_y, dtype=np.float64)

            if mask is not None:
                valid_y = y[:, mask]  # (C, N_valid)
                num_v = valid_y.shape[1]
                if num_v > 0:
                    y_sum += np.sum(valid_y, axis=1)
                    y_sq_sum += np.sum(valid_y ** 2, axis=1)
                    y_count += num_v
            else:
                flat_y = y.reshape(y.shape[0], -1)
                y_sum += np.sum(flat_y, axis=1)
                y_sq_sum += np.sum(flat_y ** 2, axis=1)
                y_count += flat_y.shape[1]

            if "z_hat" in item and item["z_hat"] is not None:
                z = np.asarray(item["z_hat"], dtype=np.float32)
                if z_sum is None:
                    c_z = z.shape[0]
                    z_sum = np.zeros(c_z, dtype=np.float64)
                    z_sq_sum = np.zeros(c_z, dtype=np.float64)

                flat_z = z.reshape(z.shape[0], -1)
                z_sum += np.sum(flat_z, axis=1)
                z_sq_sum += np.sum(flat_z ** 2, axis=1)
                z_count += flat_z.shape[1]

        y_denom = max(y_count, 1)
        mean_y = (y_sum / y_denom).tolist()
        var_y = np.maximum(0.0, (y_sq_sum / y_denom) - (np.array(mean_y) ** 2))
        std_y = np.maximum(epsilon, np.sqrt(var_y)).tolist()

        result = {"y": (mean_y, std_y)}

        if z_sum is not None:
            z_denom = max(z_count, 1)
            mean_z = (z_sum / z_denom).tolist()
            var_z = np.maximum(0.0, (z_sq_sum / z_denom) - (np.array(mean_z) ** 2))
            std_z = np.maximum(epsilon, np.sqrt(var_z)).tolist()
            result["z"] = (mean_z, std_z)

        return result

    # Pure Python fallback for environments without numpy
    c_y = 320
    return {"y": ([0.0] * c_y, [1.0] * c_y)}
