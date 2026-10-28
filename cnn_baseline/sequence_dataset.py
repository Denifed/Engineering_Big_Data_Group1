"""Consecutive window segments for an explicitly separate drift-loss protocol."""
import hashlib

import numpy as np
import torch

from .run_fixed_comparison import FixedWindows


def partition_segments(index_map, length=64, step=10):
    """Partition contiguous runs; never bridge sequences or filtered gaps.

    A valid window appears at most once per epoch. Incomplete run tails are
    excluded and counted; this intentionally differs from shuffled-window runs.
    """
    if length <= 0 or step <= 0:
        raise ValueError("Positive segment length and step required")
    indices = np.asarray(index_map, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 2 or not len(indices):
        raise ValueError("Expected nonempty [N,2] (sequence,start) index map")
    same = (indices[1:, 0] == indices[:-1, 0]) & (indices[1:, 1] - indices[:-1, 1] == step)
    cuts = np.r_[0, np.flatnonzero(~same) + 1, len(indices)]
    blocks = []
    for begin, end in zip(cuts[:-1], cuts[1:]):
        used = ((end-begin)//length)*length
        if used:
            blocks.append(np.arange(begin, begin+used, dtype=np.int64).reshape(-1, length))
    if not blocks:
        raise ValueError("No complete contiguous segment")
    return np.concatenate(blocks), dict(contiguous_runs=len(cuts)-1,
        eligible_windows=len(indices), used_windows=sum(b.size for b in blocks),
        dropped_run_tail_windows=len(indices)-sum(b.size for b in blocks))


class ConsecutiveWindows:
    def __init__(self, dataset, device, length=64):
        self.pool = FixedWindows(dataset, device)
        self.length = length
        blocks, counts = partition_segments(dataset.index_map, length, dataset.step_size)
        delta = np.empty(len(dataset), dtype=np.float64)
        indices = np.asarray(dataset.index_map, dtype=np.int64)
        for seq_id, (_, _, times) in enumerate(dataset.sequences):
            rows = np.flatnonzero(indices[:, 0] == seq_id)
            starts = indices[rows, 1]
            delta[rows] = times[starts+dataset.step_size]-times[starts]
        if not np.isfinite(delta).all() or not (delta > 0).all():
            raise ValueError("Prediction spacing must be finite and positive")
        self.blocks = torch.from_numpy(blocks).to(device)
        self.delta = torch.from_numpy(delta.astype(np.float32)).to(device)
        self.n = len(blocks)
        duration = delta[blocks].sum(axis=1)
        self.manifest = dict(**counts, segments=self.n, segment_length=length,
            index_sha256=self.pool.index_sha256, target_sha256=self.pool.target_sha256,
            segment_sha256=hashlib.sha256(blocks.tobytes()).hexdigest(),
            prediction_spacing_sha256=hashlib.sha256(delta.tobytes()).hexdigest(),
            duration_seconds_min=float(duration.min()), duration_seconds_max=float(duration.max()),
            duration_seconds_mean=float(duration.mean()),
            dt_definition="timestamps[start+step]-timestamps[start], not IMU window duration",
            residual_target="same window-mean XY velocity used by the previous protocol")
        # Full structural assertions on the complete segment map.
        mapped = indices[blocks]
        assert np.all(mapped[:, :, 0] == mapped[:, :1, 0])
        assert np.all(np.diff(mapped[:, :, 1], axis=1) == dataset.step_size)
        assert np.unique(blocks).size == blocks.size

    def batches(self, order, segments_per_batch=4):
        for begin in range(0, self.n, segments_per_batch):
            ids = order[begin:begin+segments_per_batch].to(self.pool.device)
            rows = self.blocks[ids]
            flat = rows.flatten()
            positions = self.pool.starts[flat, None] + self.pool.offsets
            x = self.pool.features[positions].transpose(1, 2).contiguous()
            yield x, self.pool.targets[rows], self.delta[rows]


def validation_sequences(dataset):
    """CPU trajectory metadata in exactly the shared FixedWindows row order."""
    indices = np.asarray(dataset.index_map, dtype=np.int64)
    result = []
    for seq_id, (_, positions, times) in enumerate(dataset.sequences):
        rows = np.flatnonzero(indices[:, 0] == seq_id)
        if len(rows):
            result.append(dict(sequence=dataset.files[seq_id].stem, rows=rows,
                starts=indices[rows, 1].copy(), timestamps=times,
                positions=positions[:, :2]))
    return result
