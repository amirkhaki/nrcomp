"""Composite loss: SmoothL1 regression + intra-image pairwise ranking loss.

Per design doc Section 9.3 and Milestone M4 specification:
- SmoothL1(s, y) with beta=0.05.
- Ranking loss over pairs (i, j) of the same source image with:
    lambda_i < lambda_j AND y_j > y_i
  penalizing ReLU(m - (s_j - s_i)) with margin m = 0.0 and weight alpha = 0.5.
All calculations executed in float32.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    BaseModule = nn.Module
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False

    class BaseModule:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)


class CompositeLoss(BaseModule):
    """Combines SmoothL1 regression and pairwise ranking loss over same-image pairs."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.05, margin: float = 0.0):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.margin = float(margin)

    def forward(
        self,
        predictions: "torch.Tensor",
        targets: "torch.Tensor",
        lambda_indices: "torch.Tensor",
        image_ids: List[str],
    ) -> Tuple["torch.Tensor", Dict[str, float]]:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for CompositeLoss.")

        # Enforce float32 for loss precision
        s = predictions.float().view(-1)
        y = targets.float().view(-1)
        lambdas = lambda_indices.view(-1)
        n = s.size(0)

        # 1. SmoothL1 Regression Loss
        smooth_l1 = F.smooth_l1_loss(s, y, beta=self.beta)

        # 2. Intra-Source-Image Pairwise Ranking Loss
        # Build boolean mask for pairs (i, j)
        device = s.device
        same_image = torch.tensor(
            [[image_ids[i] == image_ids[j] for j in range(n)] for i in range(n)],
            dtype=torch.bool,
            device=device,
        )
        lambda_order = lambdas.unsqueeze(1) < lambdas.unsqueeze(0)  # lambda_i < lambda_j
        target_order = y.unsqueeze(0) > y.unsqueeze(1)              # y_j > y_i

        valid_pairs = same_image & lambda_order & target_order

        if valid_pairs.any():
            diff = s.unsqueeze(0) - s.unsqueeze(1)  # diff[i, j] = s_j - s_i
            violations = F.relu(self.margin - diff[valid_pairs])
            rank_loss = violations.mean()
            num_pairs = int(valid_pairs.sum().item())
        else:
            rank_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            num_pairs = 0

        total_loss = smooth_l1 + self.alpha * rank_loss

        loss_dict = {
            "loss": float(total_loss.detach().item()),
            "smooth_l1": float(smooth_l1.detach().item()),
            "rank_loss": float(rank_loss.detach().item()),
            "num_pairs": float(num_pairs),
        }

        return total_loss, loss_dict


# ---------------------------------------------------------------------------
# Pure-Python Reference Oracle (for offline testing & verification)
# ---------------------------------------------------------------------------

def smooth_l1_oracle(pred: float, target: float, beta: float = 0.05) -> float:
    """Pure-Python SmoothL1 loss for a single value pair."""
    diff = abs(pred - target)
    if diff < beta:
        return 0.5 * (diff ** 2) / beta
    return diff - 0.5 * beta


def composite_loss_oracle(
    predictions: List[float],
    targets: List[float],
    lambda_indices: List[int],
    image_ids: List[str],
    alpha: float = 0.5,
    beta: float = 0.05,
    margin: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """Pure-Python reference oracle for CompositeLoss."""
    n = len(predictions)
    if n == 0:
        return 0.0, {"loss": 0.0, "smooth_l1": 0.0, "rank_loss": 0.0, "num_pairs": 0.0}

    # Mean SmoothL1
    smooth_l1_vals = [smooth_l1_oracle(p, t, beta=beta) for p, t in zip(predictions, targets)]
    smooth_l1 = sum(smooth_l1_vals) / n

    # Ranking loss over valid pairs
    violations = []
    for i in range(n):
        for j in range(n):
            if (
                image_ids[i] == image_ids[j]
                and lambda_indices[i] < lambda_indices[j]
                and targets[j] > targets[i]
            ):
                diff = predictions[j] - predictions[i]
                v = max(0.0, margin - diff)
                violations.append(v)

    if violations:
        rank_loss = sum(violations) / len(violations)
        num_pairs = len(violations)
    else:
        rank_loss = 0.0
        num_pairs = 0

    total_loss = smooth_l1 + alpha * rank_loss
    return total_loss, {
        "loss": total_loss,
        "smooth_l1": smooth_l1,
        "rank_loss": rank_loss,
        "num_pairs": float(num_pairs),
    }
