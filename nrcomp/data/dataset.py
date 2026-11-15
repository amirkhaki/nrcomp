"""Dataset and batching logic for latent training.

Implements SourceImageBatchSampler (B source images, k lambdas per image)
and collate function that handles zero-padding and valid_mask aggregation.
Strictly no spatial augmentation on latents.
"""

import random
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple


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
    }

    if has_z:
        result["z_hat"] = torch.from_numpy(padded_z) if (has_torch and has_np and not isinstance(padded_z, MockArray)) else padded_z

    if targets:
        result["targets"] = torch.tensor(targets, dtype=torch.float32) if has_torch else targets

    return result
