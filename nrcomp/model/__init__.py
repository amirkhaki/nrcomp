"""Model architecture package for latent student and losses."""

from nrcomp.model.loss import CompositeLoss, composite_loss_oracle
from nrcomp.model.pooling import (
    MaskedAttentionPool,
    MaskedMeanPool,
    masked_attention_pool_oracle,
    masked_mean_pool_oracle,
)
from nrcomp.model.student import LatentStudent
from nrcomp.model.trunk import ConvStem, ResBlock, Trunk

__all__ = [
    "ConvStem",
    "ResBlock",
    "Trunk",
    "MaskedMeanPool",
    "MaskedAttentionPool",
    "masked_mean_pool_oracle",
    "masked_attention_pool_oracle",
    "LatentStudent",
    "CompositeLoss",
    "composite_loss_oracle",
]
