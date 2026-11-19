"""Composite loss: point regression + pairwise margin ranking (design.md Section 8).

    L = (1 - alpha) * L_reg(s, y) + alpha * L_rank(s, y)

    L_reg  : Smooth L1 with parameter beta (beta <= 0 degrades to plain L1).
    L_rank : mean over all unordered batch pairs (i, j) with |y_i - y_j| > min_target_diff of
             max(0, margin - sign(y_i - y_j) * (s_i - s_j)).

Consequences of the exact formula (the previous implementation used L_reg + alpha * L_rank
over same-image pairs only, which made "alpha = 1.0" NOT a pure-ranking ablation and never
trained cross-image ordering at a fixed lambda -- the headline metric):
    alpha = 0.0 -> regression only        (ablation A2 "no rank")
    alpha = 1.0 -> ranking only           (ablation A2 "pure rank"; output scale is arbitrary)
All math runs in float32.

`lambda_indices` and `image_ids` are accepted for API compatibility and diagnostics but do not
influence the loss: pairs are formed over the whole batch, as the design specifies.
"""

from typing import Any, Dict, List, Optional, Tuple

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


DEFAULT_MIN_TARGET_DIFF = 1e-6


class CompositeLoss(BaseModule):
    """(1 - alpha) * SmoothL1 + alpha * pairwise margin ranking over all batch pairs."""

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.01,
        margin: float = 0.0,
        min_target_diff: float = DEFAULT_MIN_TARGET_DIFF,
    ):
        super().__init__()
        if not (0.0 <= float(alpha) <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.margin = float(margin)
        self.min_target_diff = float(min_target_diff)

    def forward(
        self,
        predictions: "torch.Tensor",
        targets: "torch.Tensor",
        lambda_indices: Optional["torch.Tensor"] = None,
        image_ids: Optional[List[str]] = None,
    ) -> Tuple["torch.Tensor", Dict[str, float]]:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for CompositeLoss.")

        s = predictions.float().view(-1)
        y = targets.float().view(-1)
        n = s.size(0)
        device = s.device

        # 1. Regression
        if self.beta <= 0.0:
            reg = F.l1_loss(s, y)
        else:
            reg = F.smooth_l1_loss(s, y, beta=self.beta)

        # 2. Pairwise ranking over all unordered pairs with distinguishable targets
        diff_y = y.unsqueeze(1) - y.unsqueeze(0)  # [i, j] = y_i - y_j
        diff_s = s.unsqueeze(1) - s.unsqueeze(0)  # [i, j] = s_i - s_j
        upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
        valid = upper & (diff_y.abs() > self.min_target_diff)
        num_pairs = int(valid.sum().item())
        if num_pairs > 0:
            violations = F.relu(self.margin - torch.sign(diff_y) * diff_s)[valid]
            rank = violations.mean()
        else:
            rank = torch.zeros((), device=device, dtype=torch.float32)

        total = (1.0 - self.alpha) * reg + self.alpha * rank

        return total, {
            "loss": float(total.detach().item()),
            "smooth_l1": float(reg.detach().item()),
            "rank_loss": float(rank.detach().item()),
            "num_pairs": float(num_pairs),
        }


# ---------------------------------------------------------------------------
# Pure-Python Reference Oracle (for offline testing & verification)
# ---------------------------------------------------------------------------

def smooth_l1_oracle(pred: float, target: float, beta: float = 0.01) -> float:
    """Pure-Python SmoothL1 for one value pair (beta <= 0 -> L1)."""
    diff = abs(pred - target)
    if beta <= 0.0:
        return diff
    if diff < beta:
        return 0.5 * (diff ** 2) / beta
    return diff - 0.5 * beta


def composite_loss_oracle(
    predictions: List[float],
    targets: List[float],
    lambda_indices: Optional[List[int]] = None,
    image_ids: Optional[List[str]] = None,
    alpha: float = 0.5,
    beta: float = 0.01,
    margin: float = 0.0,
    min_target_diff: float = DEFAULT_MIN_TARGET_DIFF,
) -> Tuple[float, Dict[str, float]]:
    """Pure-Python reference oracle for CompositeLoss."""
    n = len(predictions)
    if n == 0:
        return 0.0, {"loss": 0.0, "smooth_l1": 0.0, "rank_loss": 0.0, "num_pairs": 0.0}

    reg = sum(smooth_l1_oracle(p, t, beta=beta) for p, t in zip(predictions, targets)) / n

    violations = []
    for i in range(n):
        for j in range(i + 1, n):
            dy = targets[i] - targets[j]
            if abs(dy) <= min_target_diff:
                continue
            sign = 1.0 if dy > 0 else -1.0
            ds = predictions[i] - predictions[j]
            violations.append(max(0.0, margin - sign * ds))

    rank = (sum(violations) / len(violations)) if violations else 0.0
    total = (1.0 - alpha) * reg + alpha * rank
    return total, {
        "loss": total,
        "smooth_l1": reg,
        "rank_loss": rank,
        "num_pairs": float(len(violations)),
    }
