from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_PROTOCOL = "window_mean_xy_velocity_v1"
LABEL_FORMULA = "(positions[end,:2]-positions[start,:2])/(timestamps[end]-timestamps[start])"


def validate_checkpoint(checkpoint, model_name, window_size, step_size):
    """Reject legacy single-frame labels even if weight tensor shapes match."""
    expected = {"label_protocol": LABEL_PROTOCOL, "model_name": model_name,
                "window_size": window_size, "step_size": step_size}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Checkpoint {key}={checkpoint.get(key)!r}; expected {value!r}. "
                             "Legacy single-frame checkpoints must be retrained.")


class RoNINNPZDataset(Dataset):
    """x=[6,W]，y 为 [start,end] 的平均 XY 速度，end=start+W。

    输入 features[start:end]，标签从位置端点及真实时间差计算。
    NPZ targets 仅用于过滤窗口内异常位移段，不作为模型监督标签。
    """
    label_protocol = LABEL_PROTOCOL

    def __init__(
        self,
        root_dir,
        list_path,
        window_size=200,
        step_size=10,
        mode="train",
        filter_outliers=False,
        outlier_threshold=10.0,
    ):
        self.root_dir = Path(root_dir)
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.mode = mode
        self.files = []
        self.sequences = []
        self.index_map = []
        self.filtered_count = 0

        if self.window_size <= 0 or self.step_size <= 0:
            raise ValueError("window_size 和 step_size 必须为正整数")
        if not np.isfinite(outlier_threshold) or outlier_threshold <= 0:
            raise ValueError("outlier_threshold 必须为有限正数")

        names = [
            line.strip()
            for line in Path(list_path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not names or len(names) != len(set(names)):
            raise ValueError("划分名单不能为空或包含重复条目")

        for name in names:
            path = self.root_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
            self.files.append(path)

        for seq_id, path in enumerate(self.files):
            with np.load(path, allow_pickle=False) as data:
                features = data["features"].astype(np.float32)
                positions = data["positions"].astype(np.float64)
                timestamps = data["timestamps"].astype(np.float64)
                targets = data["targets"].astype(np.float32)

            if features.ndim != 2 or features.shape[1] != 6:
                raise ValueError(f"{path}: features 必须为 (N, 6)")
            if targets.ndim != 2 or targets.shape[1] != 2:
                raise ValueError(f"{path}: targets 必须为 (N, 2)")
            if positions.shape != (len(features), 3) or timestamps.shape != (len(features),):
                raise ValueError(f"{path}: positions/timestamps 必须为 (N,3)/(N,)")
            if not (
                len(features) == len(positions)
                == len(timestamps) == len(targets)
            ):
                raise ValueError(f"{path}: features/targets/positions/timestamps 长度不一致")
            if not (
                np.isfinite(features).all()
                and np.isfinite(targets).all()
                and np.isfinite(positions).all()
                and np.isfinite(timestamps).all()
            ):
                raise ValueError(f"{path}: 数据包含 NaN 或 Inf")

            if np.any(np.diff(timestamps) <= 0):
                raise ValueError(f"{path}: 时间戳必须严格递增")
            self.sequences.append((features, positions, timestamps))

            # 需要 positions[end]；输入窗口内只有 [start,end) 的 IMU。
            speed = np.linalg.norm(targets, axis=1)
            bad = speed > outlier_threshold
            bad_prefix = np.concatenate(
                ([0], np.cumsum(bad.astype(np.int64)))
            )

            for end in range(self.window_size, len(features), self.step_size):
                start = end - self.window_size

                # 平均速度覆盖的位移区间恰为 targets[start:end]。
                if filter_outliers and bad_prefix[end] - bad_prefix[start] > 0:
                    self.filtered_count += 1
                    continue

                self.index_map.append((seq_id, start))

        if not self.index_map:
            raise ValueError(f"{mode}: 没有有效窗口")
        print(
            f"{mode}: sequences={len(self.files)}, "
            f"windows={len(self.index_map)}, "
            f"filtered={self.filtered_count}, label={LABEL_PROTOCOL}"
        )

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        seq_id, start = self.index_map[index]
        features, positions, timestamps = self.sequences[seq_id]
        end = start + self.window_size

        x = features[start:end].T.copy()
        y = ((positions[end, :2] - positions[start, :2]) /
             (timestamps[end] - timestamps[start])).astype(np.float32)

        # 保留现有评估接口使用的辅助字段。
        aux = np.array(
            [
                timestamps[start],
                timestamps[end],
                positions[start, 0],
                positions[start, 1],
                positions[end, 0],
                positions[end, 1],
            ],
            dtype=np.float64,
        )

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(aux),
            torch.tensor(seq_id, dtype=torch.long),
        )
