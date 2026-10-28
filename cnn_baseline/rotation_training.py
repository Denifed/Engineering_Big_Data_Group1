"""Differentiable per-window SO(2) transforms and paired training objective."""
import torch
import torch.nn.functional as F


def rotate_velocity(v, cosine, sine):
    """Active horizontal rotation; positive angle maps +X towards +Y."""
    return torch.stack((cosine*v[:, 0]-sine*v[:, 1],
                        sine*v[:, 0]+cosine*v[:, 1]), dim=1)


def rotate_features(x, cosine, sine):
    if x.ndim != 3 or x.shape[1:] != (6, 200):
        raise ValueError("Expected [B,6,200] global gyro/acceleration windows")
    if cosine.shape != (len(x),) or sine.shape != (len(x),):
        raise ValueError("Expected one rotation per window")
    c, s = cosine[:, None], sine[:, None]
    result = x.clone()
    for a, b in ((0, 1), (3, 4)):
        result[:, a] = c*x[:, a]-s*x[:, b]
        result[:, b] = s*x[:, a]+c*x[:, b]
    return result


def paired_loss(prediction, rotated_prediction, target, cosine, sine, weight):
    if prediction.shape != target.shape or rotated_prediction.shape != target.shape or target.shape[1:] != (2,):
        raise ValueError("Expected matching [B,2] predictions and target")
    if weight < 0:
        raise ValueError("Nonnegative consistency weight required")
    rotated_target = rotate_velocity(target, cosine, sine)
    supervised = .5*(F.mse_loss(prediction, target)+F.mse_loss(rotated_prediction, rotated_target))
    consistency = F.mse_loss(rotated_prediction, rotate_velocity(prediction, cosine, sine))
    total = supervised if weight == 0 else supervised+weight*consistency
    return total, supervised, consistency
