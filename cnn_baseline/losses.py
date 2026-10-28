"""Velocity MSE plus time-normalized cumulative residual penalties.

The residual is between predictions and overlapping window-mean labels.
This is a persistent-bias penalty, not exact ground-truth displacement loss.
"""
import math

import torch


def velocity_bias_loss(prediction, target, delta, weight=1., horizons=(16, 32, 64)):
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 2:
        raise ValueError("Matching [segments,windows,2] predictions and labels required")
    if delta.shape != prediction.shape[:2] or not len(prediction):
        raise ValueError("Prediction spacing must have [segments,windows] shape")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Nonnegative finite weight required")
    if not horizons or any(h <= 0 or h > prediction.shape[1] for h in horizons):
        raise ValueError("Horizons must fit the segment length")
    if not torch.isfinite(delta).all() or not (delta > 0).all():
        raise ValueError("Finite positive prediction spacing required")
    error = prediction-target
    velocity = error.square().mean()
    weighted_error = error*delta.unsqueeze(-1)
    # Only the declared horizons are needed. Prefix reductions also avoid CUDA
    # cumsum implementations that reject deterministic-algorithm mode.
    mean_bias = torch.stack([weighted_error[:, :h].sum(dim=1)/
                             delta[:, :h].sum(dim=1, keepdim=True)
                             for h in horizons], dim=1)
    bias = mean_bias.square().mean()
    total = velocity if weight == 0 else velocity+weight*bias
    return total, velocity, bias
